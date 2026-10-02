#ifndef BPE_TOKENIZER_H
#define BPE_TOKENIZER_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class BPETokenizer {
public:
    BPETokenizer();
    ~BPETokenizer();
    
    // Load tokenizer from JSON file (tokenizer.json format)
    bool loadFromFile(const std::string& tokenizer_path);
    
    // Load tokenizer from simplified vocab JSON (exported format)
    bool loadFromVocabFile(const std::string& vocab_path);
    
    // Encode text to token IDs
    std::vector<int> encode(const std::string& text);
    
    // Decode token IDs to text
    std::string decode(const std::vector<int>& token_ids);
    
    // Get vocabulary size
    size_t getVocabSize() const { return vocab_size; }
    
    // Check if loaded
    bool isLoaded() const { return loaded; }

private:
    bool parseTokenizerJson(const std::string& json_content);
    bool parseVocabJson(const std::string& json_content);

    void buildByteEncoder();
    size_t decodeUtf8(const std::string& input, size_t offset, uint32_t& cp) const;
    std::string normalizeCodepoint(uint32_t cp) const;
    std::string normalizeText(const std::string& input) const;
    std::string bytesToUnicode(const std::vector<uint8_t>& bytes) const;
    std::vector<uint8_t> unicodeToBytes(const std::string& token) const;
    std::vector<std::string> applyBPE(const std::string& token);
    void encodeBytesChunk(const std::vector<uint8_t>& bytes, std::vector<int>& out);

    bool loaded;
    size_t vocab_size;
    int unk_id;
    bool byte_encoder_ready;
    std::unordered_map<std::string, int> token_to_id;
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> merge_ranks;
    std::vector<std::string> byte_encoder;
    std::vector<uint16_t> byte_decoder_lut;
};

#endif // BPE_TOKENIZER_H

