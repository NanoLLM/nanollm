#ifndef BPE_TOKENIZER_ESP32_H
#define BPE_TOKENIZER_ESP32_H

#include <Arduino.h>
#include <FS.h>
#include <vector>
#include <string>
#include <string_view>
#include <unordered_map>

class BPETokenizer {
public:
    BPETokenizer();
    ~BPETokenizer();
    
    // Load tokenizer from SPIFFS
    bool loadFromFile(const char* vocab_path);
    
    // Load tokenizer from embedded header (firmware)
    bool loadFromEmbedded();
    
    // Encode text to token IDs
    std::vector<int> encode(const String& text);
    
    // Decode token IDs to text
    String decode(const std::vector<int>& token_ids);
    
    // Get vocabulary size
    size_t getVocabSize() const { return vocab_size; }
    
    // Check if loaded
    bool isLoaded() const { return loaded; }

private:
    bool loaded;
    size_t vocab_size;
    int unk_id;
    bool byte_encoder_ready;
    bool progmem_vocab_;
    
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string_view, int> token_to_id;
    std::unordered_map<uint32_t, int> merge_ranks;
    std::vector<std::string> byte_encoder;
    std::vector<uint16_t> byte_decoder_lut;

    bool parseVocabJson(File& file);
    bool initializeFromEmbedded();
    void buildByteEncoder();
    std::string normalizeText(const std::string& input) const;
    std::string normalizeCodepoint(uint32_t cp) const;
    size_t decodeUtf8(const std::string& input, size_t offset, uint32_t& cp) const;
    std::string bytesToUnicode(const std::vector<uint8_t>& bytes) const;
    std::vector<uint8_t> unicodeToBytes(const std::string& token) const;
    std::vector<std::string> applyBPE(const std::string& token);
    void encodeBytesChunk(const std::vector<uint8_t>& bytes, std::vector<int>& out);
    int lookupTokenId(std::string_view token) const;
    int lookupMergeRank(uint32_t key) const;
    std::string tokenStringById(int id) const;
};

#endif // BPE_TOKENIZER_ESP32_H

