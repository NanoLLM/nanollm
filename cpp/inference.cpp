#include "model.h"
#include "bpe_tokenizer.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

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

bool load_tokenizer(BPETokenizer& tokenizer, const std::string& config_path) {
    const std::string dir = config_dir(config_path);
    const std::string tokenizer_json = dir + "tokenizer.json";
    if (file_exists(tokenizer_json) && tokenizer.loadFromFile(tokenizer_json)) {
        return true;
    }
    return tokenizer.loadFromVocabFile(dir + "vocab.json");
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <weights_path> <config_path> [prompt] [max_tokens]" << std::endl;
        return 1;
    }
    
    std::string weights_path = argv[1];
    std::string config_path = argv[2];
    std::string prompt = (argc > 3) ? argv[3] : "Hello";
    int max_tokens = (argc > 4) ? std::stoi(argv[4]) : 50;
    
    // Create model
    NanoLLM model;
    
    // Load model
    std::cout << "Loading model..." << std::endl;
    if (!model.load(weights_path, config_path)) {
        std::cerr << "Failed to load model!" << std::endl;
        return 1;
    }
    
    ModelConfig config = model.getConfig();
    std::cout << "Model loaded successfully!" << std::endl;
    std::cout << "  Vocab size: " << config.vocab_size << std::endl;
    std::cout << "  Model dim: " << config.d_model << std::endl;
    std::cout << "  Layers: " << config.n_layers << std::endl;
    std::cout << "  Memory usage: " << (model.getMemoryUsage() / 1024) << " KB" << std::endl;
    
    // Load tokenizer if available
    BPETokenizer tokenizer;
    std::vector<int> prompt_tokens;
    
    if (load_tokenizer(tokenizer, config_path)) {
        std::cout << "Using BPE tokenizer (vocab_size=" << tokenizer.getVocabSize() << ")" << std::endl;
        prompt_tokens = tokenizer.encode(prompt);
    } else {
        // Fallback to byte-level encoding
        std::cout << "BPE tokenizer not found, using byte-level encoding" << std::endl;
        for (char c : prompt) {
            prompt_tokens.push_back(static_cast<unsigned char>(c) % config.vocab_size);
        }
    }
    
    std::cout << "\nGenerating with prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "Max tokens: " << max_tokens << std::endl;
    std::cout << "\nGenerated text: ";
    
    // Generate
    std::vector<int> generated = model.generate(prompt_tokens, max_tokens, 1.0f);
    
    // Decode and print
    if (tokenizer.isLoaded()) {
        // Use BPE decoder
        std::string decoded = tokenizer.decode(generated);
        std::cout << decoded << std::endl;
    } else {
        // Fallback to byte-level
        for (int token : generated) {
            if (token >= 32 && token < 127) {  // Printable ASCII
                std::cout << static_cast<char>(token);
            } else {
                std::cout << "[" << token << "]";
            }
        }
        std::cout << std::endl;
    }

    // Inference timing (desktop runtime; prefill = full prompt forward,
    // decode = per-token forwards over the growing history).
    if (model.prefill_ms > 0.0 || model.tokens_decoded > 0) {
        double prefill_tps = (model.prefill_ms > 0.0)
            ? (static_cast<double>(prompt_tokens.size()) * 1000.0 / model.prefill_ms) : 0.0;
        double decode_tps = (model.decode_ms > 0.0)
            ? (static_cast<double>(model.tokens_decoded) * 1000.0 / model.decode_ms) : 0.0;
        std::cout << "\nInference timing (desktop):" << std::endl;
        std::cout << "  Prefill: " << prompt_tokens.size() << " prompt tokens in "
                  << model.prefill_ms << " ms (" << prefill_tps << " tok/s)" << std::endl;
        std::cout << "  Decode: " << model.tokens_decoded << " tokens in "
                  << model.decode_ms << " ms (" << decode_tps << " tok/s)" << std::endl;
        std::cout << "  Mean decode latency: "
                  << (model.tokens_decoded > 0 ? model.decode_ms / model.tokens_decoded : 0.0)
                  << " ms/token" << std::endl;
    }

    return 0;
}

