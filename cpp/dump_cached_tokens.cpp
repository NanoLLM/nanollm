#include "model.h"
#include "bpe_tokenizer.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace {

std::string config_dir(const std::string& config_path) {
    size_t last_slash = config_path.find_last_of("/\\");
    if (last_slash == std::string::npos) {
        return "";
    }
    return config_path.substr(0, last_slash + 1);
}

bool file_exists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

std::string resolve_tokenizer_path(const std::string& config_path) {
    const std::string dir = config_dir(config_path);
    const std::string tokenizer_json = dir + "tokenizer.json";
    if (file_exists(tokenizer_json)) {
        return tokenizer_json;
    }
    return dir + "vocab.json";
}

bool load_tokenizer(BPETokenizer& tokenizer, const std::string& config_path, const std::string& explicit_path) {
    if (!explicit_path.empty()) {
        if (tokenizer.loadFromFile(explicit_path)) {
            return true;
        }
        if (tokenizer.loadFromVocabFile(explicit_path)) {
            return true;
        }
    }

    const std::string path = resolve_tokenizer_path(config_path);
    if (path.find("tokenizer.json") != std::string::npos && tokenizer.loadFromFile(path)) {
        return true;
    }
    return tokenizer.loadFromVocabFile(path);
}

std::vector<int> parse_token_list(const std::string& text) {
    std::vector<int> tokens;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find(',', start);
        if (end == std::string::npos) {
            end = text.size();
        }
        std::string part = text.substr(start, end - start);
        start = end + 1;
        if (part.empty()) {
            continue;
        }
        tokens.push_back(std::atoi(part.c_str()));
    }
    return tokens;
}

bool looks_like_token_list(const std::string& prompt) {
    if (prompt.empty()) {
        return false;
    }
    for (char ch : prompt) {
        if (!(std::isdigit(static_cast<unsigned char>(ch)) || ch == ',')) {
            return false;
        }
    }
    return prompt.find(',') != std::string::npos;
}

void printJsonArray(const std::vector<int>& values) {
    std::cout << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        std::cout << values[i];
        if (i + 1 < values.size()) {
            std::cout << ",";
        }
    }
    std::cout << "]";
}

std::vector<int> generate_cached(NanoLLM& model, const std::vector<int>& prompt, int max_new_tokens) {
    const int seq_len = std::min(static_cast<int>(prompt.size()), model.getConfig().max_seq_len);
    const size_t context_start = prompt.size() > static_cast<size_t>(seq_len)
        ? prompt.size() - static_cast<size_t>(seq_len)
        : 0;

    model.resetCache();
    int next_token = -1;
    for (size_t i = context_start; i < prompt.size(); ++i) {
        next_token = model.decodeStep(prompt[i], 1.0f);
    }

    std::vector<int> generated = prompt;
    if (max_new_tokens <= 0 || next_token < 0) {
        return generated;
    }

    generated.push_back(next_token);
    if (next_token == 3) {
        return generated;
    }

    for (int step = 1; step < max_new_tokens; ++step) {
        next_token = model.decodeStep(next_token, 1.0f);
        generated.push_back(next_token);
        if (next_token == 3) {
            break;
        }
    }
    return generated;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <weights_path> <config_path> <prompt> [max_new_tokens] [tokenizer_path]" << std::endl;
        return 1;
    }

    const std::string weights_path = argv[1];
    const std::string config_path = argv[2];
    const std::string prompt = argv[3];
    const int max_new_tokens = (argc > 4) ? std::max(1, std::atoi(argv[4])) : 1;
    const std::string tokenizer_path = (argc > 5) ? argv[5] : "";

    NanoLLM model;
    if (!model.load(weights_path, config_path)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    std::vector<int> prompt_tokens;
    if (looks_like_token_list(prompt)) {
        prompt_tokens = parse_token_list(prompt);
    } else {
        BPETokenizer tokenizer;
        if (!load_tokenizer(tokenizer, config_path, tokenizer_path)) {
            std::cerr << "Failed to load tokenizer near config: " << config_path << std::endl;
            return 1;
        }
        prompt_tokens = tokenizer.encode(prompt);
    }
    if (prompt_tokens.empty()) {
        std::cerr << "Prompt produced no tokens" << std::endl;
        return 1;
    }

    std::vector<int> generated_tokens = generate_cached(model, prompt_tokens, max_new_tokens);
    int next_token = -1;
    if (generated_tokens.size() > prompt_tokens.size()) {
        next_token = generated_tokens[prompt_tokens.size()];
    }

    std::cout << "{";
    std::cout << "\"prompt_tokens\":";
    printJsonArray(prompt_tokens);
    std::cout << ",\"generated_tokens\":";
    printJsonArray(generated_tokens);
    std::cout << ",\"next_token\":" << next_token;
    std::cout << ",\"cached\":true";
    std::cout << "}" << std::endl;

    return 0;
}
