#include "bpe_tokenizer_esp32.h"
#include <ArduinoJson.h>
#include <SPIFFS.h>
#include <algorithm>
#include <cctype>
#include <limits>
#include <new>
#include <set>
#include <utility>
#include <pgmspace.h>

#ifdef NANOLLM_USE_EMBEDDED_VOCAB
#include "vocab_weights.h"
using namespace nanollm;
#endif

namespace {

std::string utf8Encode(uint32_t cp) {
    std::string out;
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

} // namespace

BPETokenizer::BPETokenizer()
    : loaded(false), vocab_size(0), unk_id(0), byte_encoder_ready(false), progmem_vocab_(false) {}

BPETokenizer::~BPETokenizer() {
    id_to_token.clear();
    token_to_id.clear();
    merge_ranks.clear();
    byte_encoder.clear();
    byte_decoder_lut.clear();
}

bool BPETokenizer::loadFromFile(const char* vocab_path) {
#ifdef NANOLLM_USE_EMBEDDED_VOCAB
    Serial.println("Embedded vocab enabled; using embedded tokenizer");
    return loadFromEmbedded();
#else
    File file = SPIFFS.open(vocab_path, "r");
    if (!file) {
        Serial.printf("Failed to open vocab file: %s\n", vocab_path);
        return false;
    }
    bool ok = parseVocabJson(file);
    file.close();
    if (!ok) {
        return false;
    }
    buildByteEncoder();
    loaded = true;
    Serial.printf("Loaded tokenizer: %d tokens (SPIFFS)\n", vocab_size);
    return true;
#endif
}

bool BPETokenizer::loadFromEmbedded() {
#ifdef NANOLLM_USE_EMBEDDED_VOCAB
    if (!initializeFromEmbedded()) {
        return false;
    }
    try {
        buildByteEncoder();
    } catch (const std::bad_alloc&) {
        Serial.println("byte_encoder allocation failed (out of memory)");
        byte_encoder.clear();
        byte_decoder_lut.clear();
        byte_encoder_ready = false;
        return false;
    }
    loaded = true;
    Serial.printf("Loaded embedded tokenizer: %d tokens, free_heap=%u\n",
                  vocab_size, ESP.getFreeHeap());
    return true;
#else
    Serial.println("Embedded vocab not enabled. Define NANOLLM_USE_EMBEDDED_VOCAB and include vocab_weights.h");
    return false;
#endif
}

bool BPETokenizer::parseVocabJson(File& file) {
    Serial.println("Tokenizer JSON loading from SPIFFS is not supported on this build. Use embedded vocab instead.");
    return false;
}

bool BPETokenizer::initializeFromEmbedded() {
#ifdef NANOLLM_USE_EMBEDDED_VOCAB
    vocab_size = VOCAB_SIZE;
    progmem_vocab_ = true;
    id_to_token.clear();
    token_to_id.clear();
    merge_ranks.clear();

    int found_unk = lookupTokenId("<UNK>");
    unk_id = found_unk >= 0 ? found_unk : 0;

    Serial.printf("Embedded vocab: %u tokens in PROGMEM (no RAM copy), %u merges\n",
                  static_cast<unsigned>(vocab_size),
                  static_cast<unsigned>(MERGE_COUNT));
    return true;
#else
    return false;
#endif
}

int BPETokenizer::lookupTokenId(std::string_view token) const {
#ifdef NANOLLM_USE_EMBEDDED_VOCAB
    if (progmem_vocab_) {
        for (size_t i = 0; i < VOCAB_SIZE; ++i) {
            TokenData token_data;
            memcpy_P(&token_data, &TOKEN_LOOKUP[i], sizeof(TokenData));
            if (token_data.length != token.size()) {
                continue;
            }
            bool match = true;
            for (uint8_t j = 0; j < token_data.length; ++j) {
                char c;
                memcpy_P(&c, &token_data.data[j], sizeof(char));
                if (c != token[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }
#endif
    auto it = token_to_id.find(token);
    return it != token_to_id.end() ? it->second : -1;
}

int BPETokenizer::lookupMergeRank(uint32_t key) const {
#ifdef NANOLLM_USE_EMBEDDED_VOCAB
    if (progmem_vocab_) {
        uint16_t left = static_cast<uint16_t>(key >> 16);
        uint16_t right = static_cast<uint16_t>(key & 0xFFFFu);
        for (size_t idx = 0; idx < MERGE_COUNT; ++idx) {
            MergeEntry entry;
            memcpy_P(&entry, &MERGE_TABLE[idx], sizeof(MergeEntry));
            if (entry.left == left && entry.right == right) {
                return static_cast<int>(idx);
            }
        }
        return -1;
    }
#endif
    auto it = merge_ranks.find(key);
    return it != merge_ranks.end() ? it->second : -1;
}

std::string BPETokenizer::tokenStringById(int id) const {
#ifdef NANOLLM_USE_EMBEDDED_VOCAB
    if (progmem_vocab_) {
        if (id < 0 || static_cast<size_t>(id) >= VOCAB_SIZE) {
            return "";
        }
        TokenData token_data;
        memcpy_P(&token_data, &TOKEN_LOOKUP[id], sizeof(TokenData));
        std::string token;
        token.reserve(token_data.length);
        for (uint8_t j = 0; j < token_data.length; ++j) {
            char c;
            memcpy_P(&c, &token_data.data[j], sizeof(char));
            token.push_back(c);
        }
        return token;
    }
#endif
    if (id < 0 || static_cast<size_t>(id) >= id_to_token.size()) {
        return "";
    }
    return id_to_token[static_cast<size_t>(id)];
}

void BPETokenizer::buildByteEncoder() {
    if (byte_encoder_ready) {
        return;
    }

    std::vector<int> bs;
    bs.reserve(256);
    for (int i = 33; i <= 126; ++i) bs.push_back(i);
    for (int i = 161; i <= 172; ++i) bs.push_back(i);
    for (int i = 174; i <= 255; ++i) bs.push_back(i);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    byte_encoder.clear();
    byte_encoder.resize(256);
    int max_code = 0;
    for (size_t i = 0; i < bs.size(); ++i) {
        int b = bs[i];
        int c = cs[i];
        byte_encoder[b] = utf8Encode(static_cast<uint32_t>(c));
        if (c > max_code) {
            max_code = c;
        }
    }

    const uint16_t kInvalidByte = std::numeric_limits<uint16_t>::max();
    byte_decoder_lut.assign(static_cast<size_t>(max_code) + 1, kInvalidByte);
    for (size_t i = 0; i < bs.size(); ++i) {
        int c = cs[i];
        if (c >= 0) {
            byte_decoder_lut[static_cast<size_t>(c)] = static_cast<uint16_t>(bs[i]);
        }
    }

    byte_encoder_ready = true;
}

size_t BPETokenizer::decodeUtf8(const std::string& input, size_t offset, uint32_t& cp) const {
    if (offset >= input.size()) {
        return 0;
    }
    uint8_t lead = static_cast<uint8_t>(input[offset]);
    if ((lead & 0x80) == 0) {
        cp = lead;
        return 1;
    } else if ((lead & 0xE0) == 0xC0 && offset + 1 < input.size()) {
        cp = ((lead & 0x1F) << 6) | (static_cast<uint8_t>(input[offset + 1]) & 0x3F);
        return 2;
    } else if ((lead & 0xF0) == 0xE0 && offset + 2 < input.size()) {
        cp = ((lead & 0x0F) << 12) |
             ((static_cast<uint8_t>(input[offset + 1]) & 0x3F) << 6) |
             (static_cast<uint8_t>(input[offset + 2]) & 0x3F);
        return 3;
    } else if ((lead & 0xF8) == 0xF0 && offset + 3 < input.size()) {
        cp = ((lead & 0x07) << 18) |
             ((static_cast<uint8_t>(input[offset + 1]) & 0x3F) << 12) |
             ((static_cast<uint8_t>(input[offset + 2]) & 0x3F) << 6) |
             (static_cast<uint8_t>(input[offset + 3]) & 0x3F);
        return 4;
    }
    cp = lead;
    return 1;
}

std::string BPETokenizer::normalizeCodepoint(uint32_t cp) const {
    if (cp >= 0x0300 && cp <= 0x036F) {
        return std::string();
    }

    switch (cp) {
        case 0x00C0: case 0x00C1: case 0x00C2: case 0x00C3: case 0x00C4: case 0x00C5:
        case 0x00E0: case 0x00E1: case 0x00E2: case 0x00E3: case 0x00E4: case 0x00E5:
        case 0x0100: case 0x0101: case 0x0102: case 0x0103: case 0x0104: case 0x0105:
            return "a";
        case 0x00C7: case 0x00E7: case 0x0106: case 0x0107: case 0x0108: case 0x0109:
        case 0x010A: case 0x010B: case 0x010C: case 0x010D:
            return "c";
        case 0x00C8: case 0x00C9: case 0x00CA: case 0x00CB:
        case 0x00E8: case 0x00E9: case 0x00EA: case 0x00EB:
        case 0x0112: case 0x0113: case 0x0114: case 0x0115: case 0x0116: case 0x0117:
        case 0x0118: case 0x0119: case 0x011A: case 0x011B:
            return "e";
        case 0x00CC: case 0x00CD: case 0x00CE: case 0x00CF:
        case 0x00EC: case 0x00ED: case 0x00EE: case 0x00EF:
        case 0x0128: case 0x0129: case 0x012A: case 0x012B: case 0x012C: case 0x012D:
        case 0x012E: case 0x012F: case 0x0130: case 0x0131:
            return "i";
        case 0x00D1: case 0x00F1: case 0x0143: case 0x0144: case 0x0145: case 0x0146:
        case 0x0147: case 0x0148: case 0x0149: case 0x014B:
            return "n";
        case 0x00D2: case 0x00D3: case 0x00D4: case 0x00D5: case 0x00D6: case 0x00D8:
        case 0x00F2: case 0x00F3: case 0x00F4: case 0x00F5: case 0x00F6: case 0x00F8:
        case 0x014C: case 0x014D: case 0x014E: case 0x014F: case 0x0150: case 0x0151:
            return "o";
        case 0x00D9: case 0x00DA: case 0x00DB: case 0x00DC:
        case 0x00F9: case 0x00FA: case 0x00FB: case 0x00FC:
        case 0x0168: case 0x0169: case 0x016A: case 0x016B: case 0x016C: case 0x016D:
        case 0x016E: case 0x016F: case 0x0170: case 0x0171:
            return "u";
        case 0x00DD: case 0x00FD: case 0x00FF: case 0x0177:
            return "y";
        case 0x00DF:
            return "ss";
        case 0x017E: case 0x017D:
            return "z";
    }

    if (cp <= 0x7F) {
        return std::string(1, static_cast<char>(cp));
    }

    return utf8Encode(cp);
}

std::string BPETokenizer::normalizeText(const std::string& input) const {
    std::string output;
    output.reserve(input.size());
    size_t offset = 0;
    while (offset < input.size()) {
        uint32_t cp = 0;
        size_t len = decodeUtf8(input, offset, cp);
        if (len == 0) {
            break;
        }
        std::string normalized = normalizeCodepoint(cp);
        output += normalized;
        offset += len;
    }
    return output;
}

std::string BPETokenizer::bytesToUnicode(const std::vector<uint8_t>& bytes) const {
    std::string result;
    result.reserve(bytes.size() * 2);
    for (uint8_t b : bytes) {
        result += byte_encoder[b];
    }
    return result;
}

std::vector<uint8_t> BPETokenizer::unicodeToBytes(const std::string& token) const {
    std::vector<uint8_t> bytes;
    bytes.reserve(token.size());
    size_t offset = 0;
    while (offset < token.size()) {
        uint32_t cp = 0;
        size_t len = decodeUtf8(token, offset, cp);
        if (len == 0) {
            break;
        }
        if (cp < byte_decoder_lut.size()) {
            uint16_t mapped = byte_decoder_lut[static_cast<size_t>(cp)];
            if (mapped != std::numeric_limits<uint16_t>::max()) {
                bytes.push_back(static_cast<uint8_t>(mapped));
            }
        }
        offset += len;
    }
    return bytes;
}

std::vector<std::string> BPETokenizer::applyBPE(const std::string& token) {
    if (token.empty()) {
        return {};
    }

    std::vector<std::string> word;
    size_t offset = 0;
    while (offset < token.size()) {
        uint32_t cp = 0;
        size_t len = decodeUtf8(token, offset, cp);
        if (len == 0) {
            break;
        }
        word.emplace_back(token.substr(offset, len));
        offset += len;
    }

    if (word.size() <= 1) {
        return word;
    }

    auto get_pairs = [](const std::vector<std::string>& symbols) {
        std::set<std::pair<std::string, std::string>> pairs;
        if (symbols.size() < 2) {
            return pairs;
        }
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            pairs.emplace(symbols[i], symbols[i + 1]);
        }
        return pairs;
    };

    auto pairs = get_pairs(word);
    while (!pairs.empty()) {
        int min_rank = std::numeric_limits<int>::max();
        std::pair<std::string, std::string> best_pair;
        bool found = false;
        for (const auto& pair : pairs) {
            int left_id = lookupTokenId(pair.first);
            if (left_id < 0) {
                continue;
            }
            int right_id = lookupTokenId(pair.second);
            if (right_id < 0) {
                continue;
            }
            uint32_t key = (static_cast<uint32_t>(left_id) << 16) |
                           (static_cast<uint32_t>(right_id) & 0xFFFFu);
            int rank = lookupMergeRank(key);
            if (rank >= 0 && rank < min_rank) {
                min_rank = rank;
                best_pair = pair;
                found = true;
            }
        }
        if (!found) {
            break;
        }

        std::vector<std::string> new_word;
        size_t i = 0;
        while (i < word.size()) {
            if (i + 1 < word.size() && word[i] == best_pair.first && word[i + 1] == best_pair.second) {
                new_word.emplace_back(word[i] + word[i + 1]);
                i += 2;
            } else {
                new_word.emplace_back(word[i]);
                i += 1;
            }
        }
        word.swap(new_word);
        pairs = get_pairs(word);
    }

    return word;
}

void BPETokenizer::encodeBytesChunk(const std::vector<uint8_t>& bytes, std::vector<int>& out) {
    if (bytes.empty()) {
        return;
    }

    std::string unicode = bytesToUnicode(bytes);
    std::vector<std::string> bpe_tokens = applyBPE(unicode);
    for (const auto& token : bpe_tokens) {
        int token_id = lookupTokenId(token);
        if (token_id >= 0) {
            out.push_back(token_id);
        } else {
            std::vector<uint8_t> fallback = unicodeToBytes(token);
            for (uint8_t b : fallback) {
                const std::string& single = byte_encoder[b];
                int single_id = lookupTokenId(single);
                if (single_id >= 0) {
                    out.push_back(single_id);
                } else {
                    out.push_back(unk_id);
                }
            }
        }
    }
}

std::vector<int> BPETokenizer::encode(const String& text) {
    if (!loaded) {
        return {};
    }

    buildByteEncoder();

    std::string input = text.c_str();
    std::string normalized = normalizeText(input);

    std::vector<int> token_ids;
    token_ids.reserve(normalized.size());

    size_t idx = 0;
    int pending_spaces = 0;
    while (idx < normalized.size()) {
        uint8_t byte = static_cast<uint8_t>(normalized[idx]);
        if (byte == ' ') {
            pending_spaces++;
            idx++;
            continue;
        }

        if (byte == '\n' || byte == '\r' || byte == '\t') {
            if (pending_spaces > 0) {
                std::vector<uint8_t> spaces(static_cast<size_t>(pending_spaces), static_cast<uint8_t>(' '));
                encodeBytesChunk(spaces, token_ids);
                pending_spaces = 0;
            }
            std::vector<uint8_t> ws = {byte};
            encodeBytesChunk(ws, token_ids);
            idx++;
            continue;
        }

        size_t start = idx;
        while (idx < normalized.size()) {
            uint8_t b = static_cast<uint8_t>(normalized[idx]);
            if (b == ' ' || b == '\n' || b == '\r' || b == '\t') {
                break;
            }
            idx++;
        }

        std::vector<uint8_t> bytes;
        bytes.reserve(pending_spaces + (idx - start));
        for (int i = 0; i < pending_spaces; ++i) {
            bytes.push_back(static_cast<uint8_t>(' '));
        }
        pending_spaces = 0;
        for (size_t pos = start; pos < idx; ++pos) {
            bytes.push_back(static_cast<uint8_t>(normalized[pos]));
        }
        encodeBytesChunk(bytes, token_ids);
    }

    if (pending_spaces > 0) {
        std::vector<uint8_t> spaces(static_cast<size_t>(pending_spaces), static_cast<uint8_t>(' '));
        encodeBytesChunk(spaces, token_ids);
    }

    return token_ids;
}

String BPETokenizer::decode(const std::vector<int>& token_ids) {
    if (!loaded) {
        return "";
    }

    buildByteEncoder();

    std::string output;
    output.reserve(token_ids.size() * 4);

    for (int id : token_ids) {
        const std::string token = tokenStringById(id);
        if (token.empty()) {
            continue;
        }
        if (token == "<PAD>" || token == "<BOS>" || token == "<EOS>") {
            continue;
        }
        if (token == "<UNK>") {
            output += "<UNK>";
            continue;
        }

        size_t offset = 0;
        while (offset < token.size()) {
            uint32_t cp = 0;
            size_t len = decodeUtf8(token, offset, cp);
            if (len == 0) {
                break;
            }
            uint16_t mapped = (cp < byte_decoder_lut.size())
                                    ? byte_decoder_lut[static_cast<size_t>(cp)]
                                    : std::numeric_limits<uint16_t>::max();
            if (mapped != std::numeric_limits<uint16_t>::max()) {
                output.push_back(static_cast<char>(mapped));
            } else {
                output.append(token.substr(offset, len));
            }
            offset += len;
        }
    }

    return String(output.c_str());
}

