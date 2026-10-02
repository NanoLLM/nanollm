#include "model.h"
#include "bpe_tokenizer.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <filesystem>

// Simple test framework
int tests_run = 0;
int tests_passed = 0;

#define TEST(name) \
    do { \
        std::cout << "TEST: " << name << " ... "; \
        tests_run++; \
        try {

#define END_TEST \
            std::cout << "PASS" << std::endl; \
            tests_passed++; \
        } catch (const std::exception& e) { \
            std::cout << "FAIL: " << e.what() << std::endl; \
        } \
    } while(0)

void test_model_loading() {
    NanoLLM model;
    
    // Try to load model (may fail if files don't exist, that's OK for unit test)
    bool loaded = model.load("../weights/model.bin", "../weights/model_config.json");
    
    if (loaded) {
        ModelConfig config = model.getConfig();
        assert(config.vocab_size > 0);
        assert(config.d_model > 0);
        assert(config.n_layers > 0);
        
        std::cout << "Model loaded: " << config.vocab_size << " vocab, " 
                  << config.d_model << " dim, " << config.n_layers << " layers" << std::endl;
    } else {
        std::cout << "Model files not found (skipping)" << std::endl;
    }
}

void test_generation() {
    NanoLLM model;
    
    bool loaded = model.load("../weights/model.bin", "../weights/model_config.json");
    if (!loaded) {
        std::cout << "Model files not found (skipping)" << std::endl;
        return;
    }
    
    // Test generation
    std::vector<int> prompt = {72, 101, 108, 108, 111};  // "Hello"
    std::vector<int> generated = model.generate(prompt, 10, 1.0f);
    
    assert(generated.size() > prompt.size());
    assert(generated.size() == prompt.size() + 10);
    
    std::cout << "Generated " << (generated.size() - prompt.size()) << " tokens" << std::endl;
}

void test_memory_usage() {
    NanoLLM model;
    
    bool loaded = model.load("../weights/model.bin", "../weights/model_config.json");
    if (!loaded) {
        std::cout << "Model files not found (skipping)" << std::endl;
        return;
    }
    
    size_t memory = model.getMemoryUsage();
    std::cout << "Memory usage: " << (memory / 1024) << " KB" << std::endl;
    
    // Should be reasonable (less than 500KB for small models)
    assert(memory < 500 * 1024);
}

void test_tokenizer_preserves_capitalization() {
    const std::vector<std::string> candidates = {
        "case_tokenizer/tokenizer.json",
        "cpp/build/case_tokenizer/tokenizer.json",
    };

    std::string tokenizer_path;
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            tokenizer_path = candidate;
            break;
        }
    }

    if (tokenizer_path.empty()) {
        std::cout << "Case tokenizer fixture not found (skipping)" << std::endl;
        return;
    }

    BPETokenizer tokenizer;
    assert(tokenizer.loadFromFile(tokenizer_path));

    const std::string text = "Hello NASA";
    std::vector<int> tokens = tokenizer.encode(text);
    std::string decoded = tokenizer.decode(tokens);
    assert(decoded == text);

    std::cout << "Tokenizer preserved: " << decoded << std::endl;
}

int main() {
    std::cout << "NanoLLM C++ Inference Tests" << std::endl;
    std::cout << "============================" << std::endl;
    std::cout << std::endl;
    
    TEST("Model Loading") {
        test_model_loading();
    } END_TEST;
    
    TEST("Text Generation") {
        test_generation();
    } END_TEST;
    
    TEST("Memory Usage") {
        test_memory_usage();
    } END_TEST;

    TEST("Tokenizer Capitalization") {
        test_tokenizer_preserves_capitalization();
    } END_TEST;
    
    std::cout << std::endl;
    std::cout << "============================" << std::endl;
    std::cout << "Tests run: " << tests_run << std::endl;
    std::cout << "Tests passed: " << tests_passed << std::endl;
    std::cout << "Tests failed: " << (tests_run - tests_passed) << std::endl;
    
    return (tests_run == tests_passed) ? 0 : 1;
}

