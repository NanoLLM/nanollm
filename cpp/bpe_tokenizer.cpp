#include "bpe_tokenizer.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <utility>

namespace {

std::string readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return {};
    }
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

std::string extractJsonString(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) {
        return {};
    }

    pos = json.find(":", pos);
    if (pos == std::string::npos) {
        return {};
    }
    ++pos;

    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        ++pos;
    }

    if (pos >= json.size() || json[pos] != '"') {
        return {};
    }
    ++pos;

    size_t start = pos;
    bool escape = false;
    while (pos < json.size()) {
        char ch = json[pos];
        if (!escape && ch == '\\') {
            escape = true;
            ++pos;
            continue;
        }
        if (!escape && ch == '"') {
            break;
        }
        escape = false;
        ++pos;
    }
    if (pos >= json.size()) {
        return {};
    }
    return json.substr(start, pos - start);
}

std::string decodeJsonToken(const std::string& token) {
    std::string decoded;
    decoded.reserve(token.size());
    for (size_t i = 0; i < token.size(); ++i) {
        char c = token[i];
        if (c == '\\' && i + 1 < token.size()) {
            char next = token[i + 1];
            switch (next) {
                case 'n': decoded.push_back('\n'); break;
                case 't': decoded.push_back('\t'); break;
                case 'r': decoded.push_back('\r'); break;
                case '\\': decoded.push_back('\\'); break;
                case '"': decoded.push_back('"'); break;
                case 'u': {
                    if (i + 5 < token.size()) {
                        std::string hex = token.substr(i + 2, 4);
                        int code = std::strtol(hex.c_str(), nullptr, 16);
                        if (code <= 0x7F) {
                            decoded.push_back(static_cast<char>(code));
                        } else if (code <= 0x7FF) {
                            decoded.push_back(static_cast<char>(0xC0 | ((code >> 6) & 0x1F)));
                            decoded.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                        } else if (code <= 0xFFFF) {
                            decoded.push_back(static_cast<char>(0xE0 | ((code >> 12) & 0x0F)));
                            decoded.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                            decoded.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                        } else {
                            decoded.push_back(static_cast<char>(0xF0 | ((code >> 18) & 0x07)));
                            decoded.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
                            decoded.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                            decoded.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                        }
                        i += 5;
                    }
                    break;
                }
                default:
                    decoded.push_back(next);
                    break;
            }
            ++i;
        } else {
            decoded.push_back(c);
        }
    }
    return decoded;
}

std::string directoryOf(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return "";
    }
    return path.substr(0, pos + 1);
}

bool isAbsolutePath(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    if (path[0] == '/' || path[0] == '\\') {
        return true;
    }
    if (path.size() > 1 && std::isalpha(static_cast<unsigned char>(path[0])) && path[1] == ':') {
        return true;
    }
    return false;
}

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
    : loaded(false), vocab_size(0), unk_id(0), byte_encoder_ready(false) {}

BPETokenizer::~BPETokenizer() {
    token_to_id.clear();
    id_to_token.clear();
    merge_ranks.clear();
    byte_encoder.clear();
    byte_decoder_lut.clear();
}

bool BPETokenizer::loadFromVocabFile(const std::string& vocab_path) {
    std::string directory = directoryOf(vocab_path);
    std::string info_path = directory + "tokenizer_info.json";
    std::string info_content = readFile(info_path);
    if (!info_content.empty()) {
        std::string tokenizer_path = extractJsonString(info_content, "tokenizer_path");
        if (!tokenizer_path.empty()) {
            auto try_load = [&](const std::string& candidate) -> bool {
                std::string data = readFile(candidate);
                if (data.empty()) {
                    return false;
                }
                if (!parseTokenizerJson(data)) {
                    return false;
                }
                buildByteEncoder();
                loaded = true;
                return true;
            };

            if (isAbsolutePath(tokenizer_path)) {
                if (try_load(tokenizer_path)) {
                    return true;
                }
            } else {
                if (try_load(directory + tokenizer_path)) {
                    return true;
                }
                if (try_load(tokenizer_path)) {
                    return true;
                }
            }
        }
    }

    std::string content = readFile(vocab_path);
    if (content.empty()) {
        return false;
    }

    if (!parseVocabJson(content)) {
        return false;
    }

    buildByteEncoder();
    loaded = true;
    return true;
}

bool BPETokenizer::parseVocabJson(const std::string& json_content) {
    token_to_id.clear();
    id_to_token.clear();
    merge_ranks.clear();
    byte_encoder_ready = false;
    vocab_size = 0;

    size_t vocab_pos = json_content.find("\"vocab_size\"");
    if (vocab_pos != std::string::npos) {
        size_t colon_pos = json_content.find(":", vocab_pos);
        if (colon_pos != std::string::npos) {
            size_t end_pos = json_content.find_first_of(",}", colon_pos);
            std::string vocab_str = json_content.substr(colon_pos + 1, end_pos - colon_pos - 1);
            vocab_size = static_cast<size_t>(std::max(0, std::stoi(vocab_str)));
        }
    }

    size_t id_to_token_pos = json_content.find("\"id_to_token\"");
    if (id_to_token_pos != std::string::npos) {
        size_t start = json_content.find("{", id_to_token_pos);
        size_t end = json_content.find("}", start);
        size_t pos = start + 1;
        while (pos < end) {
            size_t key_start = json_content.find("\"", pos);
            if (key_start == std::string::npos || key_start >= end) {
                break;
            }
            ++key_start;
            size_t key_end = json_content.find("\"", key_start);
            if (key_end == std::string::npos) {
                break;
            }
            std::string id_str = json_content.substr(key_start, key_end - key_start);
            int id = std::stoi(id_str);

            size_t val_start = json_content.find(":", key_end);
            if (val_start == std::string::npos) {
                break;
            }
            val_start = json_content.find("\"", val_start);
            if (val_start == std::string::npos) {
                break;
            }
            ++val_start;

            size_t val_end = val_start;
            bool escape = false;
            while (val_end < json_content.size()) {
                char ch = json_content[val_end];
                if (!escape && ch == '\\') {
                    escape = true;
                    ++val_end;
                    continue;
                }
                if (!escape && ch == '"') {
                    break;
                }
                escape = false;
                ++val_end;
            }
            std::string raw = json_content.substr(val_start, val_end - val_start);
            std::string decoded = decodeJsonToken(raw);

            if (id >= 0) {
                if (static_cast<size_t>(id) >= id_to_token.size()) {
                    id_to_token.resize(static_cast<size_t>(id) + 1);
                }
                id_to_token[static_cast<size_t>(id)] = decoded;
                token_to_id[decoded] = id;
            }

            pos = json_content.find(",", val_end);
            if (pos == std::string::npos || pos >= end) {
                break;
            }
            ++pos;
        }
    }

    if (vocab_size < id_to_token.size()) {
        vocab_size = id_to_token.size();
    }

    if (auto it = token_to_id.find("<UNK>"); it != token_to_id.end()) {
        unk_id = it->second;
    } else {
        unk_id = 0;
    }

    return !id_to_token.empty();
}

bool BPETokenizer::loadFromFile(const std::string& tokenizer_path) {
    std::string content = readFile(tokenizer_path);
    if (content.empty()) {
        return false;
    }

    if (!parseTokenizerJson(content)) {
        return false;
    }

    buildByteEncoder();
    loaded = true;
    return true;
}

bool BPETokenizer::parseTokenizerJson(const std::string& json_content) {
    token_to_id.clear();
    id_to_token.clear();
    merge_ranks.clear();
    byte_encoder_ready = false;
    vocab_size = 0;

    size_t model_pos = json_content.find("\"model\"");
    if (model_pos == std::string::npos) {
        return false;
    }

    size_t vocab_pos = json_content.find("\"vocab\"", model_pos);
    if (vocab_pos == std::string::npos) {
        return false;
    }

    size_t start = json_content.find("{", vocab_pos);
    if (start == std::string::npos) {
        return false;
    }
    size_t end = start + 1;
    int brace_depth = 1;
    while (end < json_content.size() && brace_depth > 0) {
        char ch = json_content[end];
        if (ch == '"') {
            ++end;
            bool escape = false;
            while (end < json_content.size()) {
                char c = json_content[end];
                if (!escape && c == '\\') {
                    escape = true;
                    ++end;
                    continue;
                }
                if (!escape && c == '"') {
                    ++end;
                    break;
                }
                escape = false;
                ++end;
            }
            continue;
        }
        if (ch == '{') {
            ++brace_depth;
        } else if (ch == '}') {
            --brace_depth;
        }
        ++end;
    }
    size_t pos = start + 1;
    while (pos < end) {
        size_t key_start = json_content.find("\"", pos);
        if (key_start == std::string::npos || key_start >= end) {
            break;
        }
        ++key_start;
        // Scan to the real closing quote, honoring backslash escapes so tokens
        // that contain escaped quotes (e.g. "\u0120\"") are read in full.
        size_t key_end = key_start;
        bool escape = false;
        while (key_end < json_content.size()) {
            char c = json_content[key_end];
            if (!escape && c == '\\') { escape = true; ++key_end; continue; }
            if (!escape && c == '"') { break; }
            escape = false;
            ++key_end;
        }
        if (key_end >= json_content.size()) {
            break;
        }
        std::string token_raw = json_content.substr(key_start, key_end - key_start);
        std::string token = decodeJsonToken(token_raw);

        size_t val_start = json_content.find(":", key_end);
        if (val_start == std::string::npos) {
            break;
        }
        ++val_start;
        while (val_start < json_content.size() && std::isspace(static_cast<unsigned char>(json_content[val_start]))) {
            ++val_start;
        }
        size_t val_end = val_start;
        while (val_end < json_content.size() && (std::isdigit(static_cast<unsigned char>(json_content[val_end])) || json_content[val_end] == '-')) {
            ++val_end;
        }
        int id = std::stoi(json_content.substr(val_start, val_end - val_start));
        if (id >= 0) {
            if (static_cast<size_t>(id) >= id_to_token.size()) {
                id_to_token.resize(static_cast<size_t>(id) + 1);
            }
            id_to_token[static_cast<size_t>(id)] = token;
            token_to_id[token] = id;
        }

        pos = json_content.find(",", val_end);
        if (pos == std::string::npos || pos >= end) {
            break;
        }
        ++pos;
    }

    vocab_size = id_to_token.size();
    if (auto it = token_to_id.find("<UNK>"); it != token_to_id.end()) {
        unk_id = it->second;
    } else {
        unk_id = 0;
    }

    auto parse_json_string = [&](size_t quote_pos) -> std::pair<std::string, size_t> {
        if (quote_pos == std::string::npos || quote_pos >= json_content.size() || json_content[quote_pos] != '"') {
            return {std::string(), std::string::npos};
        }
        size_t pos_cursor = quote_pos + 1;
        size_t start_cursor = pos_cursor;
        bool escape = false;
        while (pos_cursor < json_content.size()) {
            char ch = json_content[pos_cursor];
            if (!escape && ch == '\\') {
                escape = true;
                ++pos_cursor;
                continue;
            }
            if (!escape && ch == '"') {
                break;
            }
            escape = false;
            ++pos_cursor;
        }
        if (pos_cursor >= json_content.size()) {
            return {std::string(), std::string::npos};
        }
        std::string raw = json_content.substr(start_cursor, pos_cursor - start_cursor);
        std::string decoded = decodeJsonToken(raw);
        return {decoded, pos_cursor + 1};
    };

    size_t merges_pos = json_content.find("\"merges\"", model_pos);
    if (merges_pos != std::string::npos) {
        size_t array_start = json_content.find("[", merges_pos);
        if (array_start != std::string::npos) {
            size_t pos_cursor = array_start + 1;
            int depth = 1;
            int rank = 0;
            std::string pending_left;
            while (pos_cursor < json_content.size() && depth > 0) {
                char ch = json_content[pos_cursor];
                if (ch == '[') {
                    ++depth;
                    ++pos_cursor;
                    continue;
                }
                if (ch == ']') {
                    --depth;
                    ++pos_cursor;
                    pending_left.clear();
                    continue;
                }
                if (ch == '"') {
                    auto token_result = parse_json_string(pos_cursor);
                    if (token_result.second == std::string::npos) {
                        break;
                    }
                    const std::string& token = token_result.first;
                    pos_cursor = token_result.second;

                    if (depth >= 2) {
                        size_t space_pos = token.find(' ');
                        if (space_pos != std::string::npos) {
                            std::string left = token.substr(0, space_pos);
                            std::string right = token.substr(space_pos + 1);
                            merge_ranks[left + '\t' + right] = rank++;
                            pending_left.clear();
                        } else if (pending_left.empty()) {
                            pending_left = token;
                        } else {
                            merge_ranks[pending_left + '\t' + token] = rank++;
                            pending_left.clear();
                        }
                    }
                    continue;
                }
                ++pos_cursor;
            }
        }
    }

    return !id_to_token.empty();
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
            ++n;
        }
    }

    byte_encoder.assign(256, std::string());
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
    if ((lead & 0x80u) == 0) {
        cp = lead;
        return 1;
    }
    if ((lead & 0xE0u) == 0xC0u && offset + 1 < input.size()) {
        cp = ((lead & 0x1Fu) << 6) | (static_cast<uint8_t>(input[offset + 1]) & 0x3Fu);
        return 2;
    }
    if ((lead & 0xF0u) == 0xE0u && offset + 2 < input.size()) {
        cp = ((lead & 0x0Fu) << 12) |
             ((static_cast<uint8_t>(input[offset + 1]) & 0x3Fu) << 6) |
             (static_cast<uint8_t>(input[offset + 2]) & 0x3Fu);
        return 3;
    }
    if ((lead & 0xF8u) == 0xF0u && offset + 3 < input.size()) {
        cp = ((lead & 0x07u) << 18) |
             ((static_cast<uint8_t>(input[offset + 1]) & 0x3Fu) << 12) |
             ((static_cast<uint8_t>(input[offset + 2]) & 0x3Fu) << 6) |
             (static_cast<uint8_t>(input[offset + 3]) & 0x3Fu);
        return 4;
    }
    cp = lead;
    return 1;
}

std::string BPETokenizer::normalizeCodepoint(uint32_t cp) const {
    if (cp >= 0x0300 && cp <= 0x036F) {
        return {};
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
        default:
            break;
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
        if (b < byte_encoder.size()) {
            result += byte_encoder[b];
        }
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

    if (word.size() <= 1 || merge_ranks.empty()) {
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
            std::string key = pair.first + '\t' + pair.second;
            auto it = merge_ranks.find(key);
            if (it != merge_ranks.end() && it->second < min_rank) {
                min_rank = it->second;
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
                ++i;
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
        auto it = token_to_id.find(token);
        if (it != token_to_id.end()) {
            out.push_back(it->second);
        } else {
            std::vector<uint8_t> fallback = unicodeToBytes(token);
            if (fallback.empty()) {
                out.push_back(unk_id);
                continue;
            }
            for (uint8_t b : fallback) {
                const std::string& single = byte_encoder[b];
                auto single_it = token_to_id.find(single);
                if (single_it != token_to_id.end()) {
                    out.push_back(single_it->second);
                } else {
                    out.push_back(unk_id);
                }
            }
        }
    }
}

std::vector<int> BPETokenizer::encode(const std::string& text) {
    if (!loaded) {
        return {};
    }

    buildByteEncoder();

    std::string normalized = normalizeText(text);
    std::vector<int> token_ids;
    token_ids.reserve(normalized.size());

    size_t idx = 0;
    int pending_spaces = 0;
    while (idx < normalized.size()) {
        uint8_t byte = static_cast<uint8_t>(normalized[idx]);
        if (byte == ' ') {
            ++pending_spaces;
            ++idx;
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
            ++idx;
            continue;
        }

        size_t start = idx;
        while (idx < normalized.size()) {
            uint8_t b = static_cast<uint8_t>(normalized[idx]);
            if (b == ' ' || b == '\n' || b == '\r' || b == '\t') {
                break;
            }
            ++idx;
        }

        std::vector<uint8_t> bytes;
        bytes.reserve(static_cast<size_t>(pending_spaces) + (idx - start));
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

std::string BPETokenizer::decode(const std::vector<int>& token_ids) {
    if (!loaded) {
        return {};
    }

    buildByteEncoder();

    std::string output;
    output.reserve(token_ids.size() * 4);

    for (int id : token_ids) {
        if (id < 0 || static_cast<size_t>(id) >= id_to_token.size()) {
            continue;
        }
        const std::string& token = id_to_token[static_cast<size_t>(id)];
        if (token == "<PAD>" || token == "<BOS>" || token == "<EOS>") {
            continue;
        }
        if (token == "<UNK>") {
            output += "<UNK>";
            continue;
        }

        std::vector<uint8_t> bytes = unicodeToBytes(token);
        if (bytes.empty()) {
            output += token;
            continue;
        }
        for (uint8_t b : bytes) {
            output.push_back(static_cast<char>(b));
        }
    }

    return output;
}

