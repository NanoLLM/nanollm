#include "model.h"
#include "bpe_tokenizer.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
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

    BPETokenizer tokenizer;
    if (!load_tokenizer(tokenizer, config_path, tokenizer_path)) {
        std::cerr << "Failed to load tokenizer near config: " << config_path << std::endl;
        return 1;
    }

    std::vector<int> prompt_tokens = tokenizer.encode(prompt);
    if (prompt_tokens.empty()) {
        std::cerr << "Prompt produced no tokens" << std::endl;
        return 1;
    }

    std::vector<int> generated_tokens = model.generate(prompt_tokens, max_new_tokens, 1.0f);
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
    std::cout << "}" << std::endl;

    return 0;
}
