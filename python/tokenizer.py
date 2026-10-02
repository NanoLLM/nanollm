"""
BPE (Byte Pair Encoding) tokenizer for NanoLLM.
Uses tokenizers library for efficient BPE implementation.
"""
import hashlib
import json
import os
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from tokenizers.normalizers import NFD, StripAccents, Sequence


class BPETokenizer:
    """BPE tokenizer wrapper for NanoLLM."""
    
    def __init__(self, vocab_size=500):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self._is_trained = False
        self._fingerprint = None
    
    def train(self, text_files, output_dir=None):
        """
        Train BPE tokenizer on text files.
        
        Args:
            text_files: List of text file paths or single path
            output_dir: Directory to save tokenizer files
        """
        if isinstance(text_files, str):
            text_files = [text_files]
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.normalizer = Sequence([NFD(), StripAccents()])
        
        # Trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<UNK>", "<PAD>", "<BOS>", "<EOS>"],
            min_frequency=2
        )
        
        # Train on files
        print(f"Training BPE tokenizer on {len(text_files)} file(s)...")
        self.tokenizer.train(files=text_files, trainer=trainer)
        
        # Add post-processor
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.enable_padding(pad_id=1, pad_token="<PAD>")
        self._fingerprint = self._compute_fingerprint()
        
        self._is_trained = True
        
        # Save if output_dir provided
        if output_dir:
            self.save(output_dir)
        
        print(f"✓ BPE tokenizer trained with vocab_size={self.vocab_size}")
        return self
    
    def encode(self, text):
        """Encode text to token IDs."""
        if not self._is_trained:
            raise ValueError("Tokenizer not trained. Call train() first.")
        encoding = self.tokenizer.encode(text)
        return encoding.ids
    
    def decode(self, token_ids):
        """Decode token IDs to text."""
        if not self._is_trained:
            raise ValueError("Tokenizer not trained. Call train() first.")
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def save(self, output_dir):
        """Save tokenizer to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tokenizer
        tokenizer_path = os.path.join(output_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        
        # Save vocab info
        vocab_info = {
            "vocab_size": self.vocab_size,
            "tokenizer_path": tokenizer_path,
            "tokenizer_fingerprint": self.fingerprint(),
        }
        vocab_info_path = os.path.join(output_dir, "vocab_info.json")
        with open(vocab_info_path, 'w') as f:
            json.dump(vocab_info, f, indent=2)
        
        print(f"✓ Tokenizer saved to {output_dir}")
    
    def load(self, tokenizer_path):
        """Load tokenizer from file."""
        try:
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        except Exception:
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                tokenizer_json = json.load(f)

            model_cfg = tokenizer_json.get('model', {})
            merges = model_cfg.get('merges', [])
            repaired = False

            if isinstance(merges, list):
                repaired_merges = []
                for merge in merges:
                    if isinstance(merge, str):
                        repaired_merges.append(merge)
                    elif isinstance(merge, (list, tuple)) and len(merge) == 2:
                        left, right = merge
                        repaired_merges.append(f"{left} {right}")
                        repaired = True
                    else:
                        continue

                if repaired and repaired_merges:
                    model_cfg['merges'] = repaired_merges
                    tokenizer_json['model'] = model_cfg

            if not repaired:
                raise

            serialized = json.dumps(tokenizer_json, ensure_ascii=False)
            self.tokenizer = Tokenizer.from_str(serialized)

            try:
                with open(tokenizer_path, 'w', encoding='utf-8') as f:
                    json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)
            except OSError:
                pass
        self._fingerprint = self._compute_fingerprint()
        self._is_trained = True
        
        # Get vocab size from tokenizer
        self.vocab_size = self.tokenizer.get_vocab_size()
        
        # Ensure decoder is configured when loading existing files.
        if self.tokenizer.decoder is None:
            self.tokenizer.decoder = decoders.ByteLevel()
        
        return self

    def _compute_fingerprint(self):
        if self.tokenizer is None:
            return None
        serialized = self.tokenizer.to_str()
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def fingerprint(self):
        """Return a stable fingerprint for cache invalidation and provenance."""
        if self._fingerprint is None:
            self._fingerprint = self._compute_fingerprint()
        return self._fingerprint
    
    @classmethod
    def from_file(cls, tokenizer_path):
        """Create tokenizer from saved file."""
        tokenizer = cls()
        tokenizer.load(tokenizer_path)
        return tokenizer
    
    def get_vocab_size(self):
        """Get vocabulary size."""
        if self.tokenizer:
            return self.tokenizer.get_vocab_size()
        return self.vocab_size


def train_tokenizer_from_data(data_file, vocab_size=500, output_dir=None):
    """
    Convenience function to train tokenizer from a data file.
    
    Args:
        data_file: Path or list of paths used to train the tokenizer
        vocab_size: Vocabulary size
        output_dir: Directory to save tokenizer (default: same as data file)
    """
    data_files = data_file if isinstance(data_file, (list, tuple)) else [data_file]
    if not data_files:
        raise ValueError("At least one tokenizer training file is required")
    if output_dir is None:
        output_dir = os.path.dirname(data_files[0]) or "."
    
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(list(data_files), output_dir=output_dir)
    
    return tokenizer

