/*
 * NanoLLM M5Stack Cardputer Chat Interface
 * 
 * Interactive chat interface for the NanoLLM language model.
 * 
 * Hardware: M5Stack Cardputer
 * - ESP32-S3
 * - 1.14" TFT Display (240x135)
 * - QWERTY Keyboard
 */

#include <M5Cardputer.h>
#include <SPIFFS.h>
#include <esp32-hal-psram.h>
#include "model_esp32.h"
#include "bpe_tokenizer_esp32.h"
#include "chat_interface.h"
#include "serial_verify.h"
#include "system_utils.h"
#include "optimize_ops.h"

// Model instance (MoE decoder)
NanoLLM model;

// Tokenizer instance
BPETokenizer tokenizer;

// Chat interface
ChatInterface chat;

struct StreamState {
    std::vector<int> tokens;
    String text;
    unsigned long last_ui_ms = 0;
    int tokens_since_render = 0;
};

String fallbackTokenText(int token) {
    if (token >= 32 && token < 127) {
        return String(static_cast<char>(token));
    }
    return "[" + String(token) + "]";
}

void streamGeneratedToken(int token, int step, void* user_data) {
    StreamState* state = static_cast<StreamState*>(user_data);
    if (!state) {
        return;
    }

    state->tokens.push_back(token);
    if (tokenizer.isLoaded()) {
        state->text = tokenizer.decode(state->tokens);
    } else {
        state->text += fallbackTokenText(token);
    }

    const unsigned long now = millis();
    state->tokens_since_render++;
    const bool render_now = (step == 0) || (state->tokens_since_render >= 4) ||
                            (now - state->last_ui_ms >= 200);
    if (render_now) {
        chat.updateLastMessage("assistant", state->text);
        state->last_ui_ms = now;
        state->tokens_since_render = 0;
    }

    Serial.printf("@STREAM|%d|%d|%s\n", step, token, state->text.c_str());
    M5Cardputer.update();
    nano_feed_watchdog();
}

void setup() {
    // Initialize M5Cardputer
    auto cfg = M5.config();
    M5Cardputer.begin(cfg, true);

    Serial.begin(115200);
    delay(10);
    Serial.println("[system] NanoLLM Cardputer boot");
    gelu_lut_init();  // Phase 1: GELU lookup-table init for faster inference
    nano_log_reset_reason();
    
    M5Cardputer.Display.setRotation(1);
    M5Cardputer.Display.setTextSize(1);
    M5Cardputer.Display.setTextColor(WHITE, BLACK);
    M5Cardputer.Display.fillScreen(BLACK);

#ifndef NANOLLM_NO_PSRAM
    if (!psramFound()) {
        M5Cardputer.Display.println("PSRAM not detected!");
        M5Cardputer.Display.println("Rebuild with PIO_ENV=m5stack_cardputer_nopsram");
        delay(4000);
        return;
    }
#else
    if (psramFound()) {
        M5Cardputer.Display.println("PSRAM detected (no-PSRAM build)");
        delay(500);
    }
#endif
    
    // Initialize SPIFFS
    M5Cardputer.Display.setCursor(0, 0);
    M5Cardputer.Display.println("Initializing SPIFFS...");
    
    if (!SPIFFS.begin(true)) {
        M5Cardputer.Display.println("SPIFFS Mount Failed!");
        delay(2000);
        return;
    }
    
    M5Cardputer.Display.println("SPIFFS OK");
    delay(500);

    // Load tokenizer before model buffers — buildByteEncoder needs contiguous heap.
    bool tokenizer_loaded = false;
    M5Cardputer.Display.fillScreen(BLACK);
    M5Cardputer.Display.setCursor(0, 0);
#ifdef NANOLLM_USE_EMBEDDED_VOCAB
    M5Cardputer.Display.println("Loading embedded vocab...");
    tokenizer_loaded = tokenizer.loadFromEmbedded();
    if (!tokenizer_loaded) {
        M5Cardputer.Display.println("Embedded vocab failed!");
        M5Cardputer.Display.println("Falling back to SPIFFS...");
        delay(500);
    }
#endif
    if (!tokenizer_loaded) {
        if (SPIFFS.exists("/vocab.json")) {
            M5Cardputer.Display.println("Loading tokenizer from SPIFFS...");
            tokenizer_loaded = tokenizer.loadFromFile("/vocab.json");
            if (tokenizer_loaded) {
                M5Cardputer.Display.printf("Tokenizer: %d tokens\n", tokenizer.getVocabSize());
            }
        }
    } else {
        M5Cardputer.Display.printf("Embedded vocab: %d tokens\n", tokenizer.getVocabSize());
    }
    
    // Load model
    M5Cardputer.Display.println("");
    M5Cardputer.Display.println("Loading model...");
    bool model_loaded = false;

#ifdef NANOLLM_USE_EMBEDDED_WEIGHTS
    // Try loading from embedded firmware first
    M5Cardputer.Display.println("Loading from firmware...");
    model_loaded = model.loadFromEmbedded();

    if (!model_loaded) {
        M5Cardputer.Display.println("Embedded load failed!");
        M5Cardputer.Display.println("Falling back to SPIFFS...");
        delay(1000);
    }
#endif

    // Fall back to SPIFFS if embedded loading failed or not enabled
    if (!model_loaded) {
        if (SPIFFS.exists("/model.bin") && SPIFFS.exists("/model_config.json")) {
            M5Cardputer.Display.println("Loading from SPIFFS...");
            model_loaded = model.load("/model.bin", "/model_config.json");
        } else {
            M5Cardputer.Display.println("Model files not found!");
            M5Cardputer.Display.println("");
            M5Cardputer.Display.println("Please either:");
            M5Cardputer.Display.println("1. Embed weights in firmware");
            M5Cardputer.Display.println("2. Upload to SPIFFS:");
            M5Cardputer.Display.println("   - model.bin");
            M5Cardputer.Display.println("   - model_config.json");
            delay(5000);
            return;
        }
    }

    if (!model_loaded) {
        M5Cardputer.Display.println("Model load failed!");
        M5Cardputer.Display.println("Likely out of memory.");
        M5Cardputer.Display.println("Use PIO_ENV=m5stack_cardputer");
        M5Cardputer.Display.println("(PSRAM build) for MoE models.");
        Serial.println("Model load failed - halting to avoid reboot loop.");
        while (true) {
            delay(1000);
        }
    }

    ModelConfig config = model.getConfig();

    // Initialize chat interface
    chat.begin();

    // Show model info briefly
    chat.showStatus("Model ready! Vocab: " + String(config.vocab_size) +
                    (tokenizer_loaded ? " (BPE)" : ""));
    delay(1500);
    chat.hideStatus();

    Serial.printf("[system] tokenizer_loaded=%d vocab=%d free_heap=%u\n",
                  tokenizer_loaded, tokenizer.getVocabSize(), ESP.getFreeHeap());
    serialVerifyAnnounceReady();
}

String generateResponse(const String& prompt) {
    // Match the exact single-turn template used by Python chat fine-tuning.
    String chat_prompt = "User: ";
    chat_prompt += prompt;
    chat_prompt += "\nAssistant:";

    // Convert prompt to tokens (BPE if available, else byte-level)
    std::vector<int> prompt_tokens;
    if (tokenizer.isLoaded()) {
        prompt_tokens = tokenizer.encode(chat_prompt);
    } else {
        // Fallback to byte-level
        for (int i = 0; i < chat_prompt.length(); i++) {
            prompt_tokens.push_back(static_cast<unsigned char>(chat_prompt.charAt(i)) % 256);
        }
    }
    
    StreamState stream_state;
    std::vector<int> generated = model.generate(prompt_tokens, 30, 1.0f, streamGeneratedToken, &stream_state);

    if (stream_state.text.length() > 0) {
        return stream_state.text;
    }

    std::vector<int> new_tokens;
    if (generated.size() > prompt_tokens.size()) {
        new_tokens.assign(generated.begin() + prompt_tokens.size(), generated.end());
    }
    if (tokenizer.isLoaded()) {
        return tokenizer.decode(new_tokens);
    }

    String result;
    for (int token : new_tokens) {
        result += fallbackTokenText(token);
    }
    return result;
}

void loop() {
    serialVerifyPoll(model, tokenizer);

    M5Cardputer.update();
    
    // Update chat interface
    chat.loop();
    
    // Check if user wants to send a message
    if (chat.isSendRequested()) {
        String user_input = chat.getInputBuffer();
        
        if (user_input.length() > 0) {
            // Add user message to chat
            chat.addMessage("user", user_input);
            chat.clearInput();
            chat.clearSendFlag();
            
            // Add assistant placeholder so the result can stream into it
            chat.addMessage("assistant", "");

            // Show generating status
            chat.showStatus("Generating...");

            // Generate response (feeds WDT inside model.generate)
            String response = generateResponse(user_input);

            // Hide status
            chat.hideStatus();
            
            // Ensure the final decoded text is visible even if generation stopped early
            chat.updateLastMessage("assistant", response);

            Serial.printf("[system] free_heap=%u after generate\n", ESP.getFreeHeap());
        } else {
            chat.clearSendFlag();
        }
    }

    nano_feed_watchdog();
}
