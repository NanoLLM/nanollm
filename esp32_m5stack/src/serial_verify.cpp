#include "serial_verify.h"
#include "system_utils.h"
#include <ArduinoJson.h>

namespace {

String g_line_buffer;

void writeJsonArray(JsonArray arr, const std::vector<int>& values) {
    for (size_t i = 0; i < values.size(); ++i) {
        arr.add(values[i]);
    }
}

std::vector<int> encodePrompt(BPETokenizer& tokenizer, const String& text) {
    if (tokenizer.isLoaded()) {
        return tokenizer.encode(text);
    }

    std::vector<int> tokens;
    tokens.reserve(text.length());
    for (int i = 0; i < text.length(); ++i) {
        tokens.push_back(static_cast<unsigned char>(text.charAt(i)) % 256);
    }
    return tokens;
}

String fallbackTokenText(int token) {
    if (token >= 32 && token < 127) {
        return String(static_cast<char>(token));
    }
    return "[" + String(token) + "]";
}

struct SerialStreamState {
    BPETokenizer* tokenizer;
    std::vector<int> tokens;
    String text;
};

void streamSerialToken(int token, int step, void* user_data) {
    SerialStreamState* state = static_cast<SerialStreamState*>(user_data);
    if (!state) {
        return;
    }

    state->tokens.push_back(token);
    if (state->tokenizer && state->tokenizer->isLoaded()) {
        state->text = state->tokenizer->decode(state->tokens);
    } else {
        state->text += fallbackTokenText(token);
    }

    Serial.printf("@STREAM|%d|%d|%s\n", step, token, state->text.c_str());
}

void sendGenerateResponse(NanoLLM& model, BPETokenizer& tokenizer,
                          const String& prompt, int max_new_tokens, float temperature) {
    std::vector<int> prompt_tokens = encodePrompt(tokenizer, prompt);
    if (prompt_tokens.empty()) {
        Serial.println("@ERROR|prompt produced no tokens");
        return;
    }

    SerialStreamState stream_state{&tokenizer, {}, ""};
    std::vector<int> generated = model.generate(prompt_tokens, max_new_tokens, temperature,
                                                streamSerialToken, &stream_state);
    int next_token = -1;
    if (generated.size() > prompt_tokens.size()) {
        next_token = generated[prompt_tokens.size()];
    }

    StaticJsonDocument<4096> doc;
    JsonArray prompt_arr = doc.createNestedArray("prompt_tokens");
    writeJsonArray(prompt_arr, prompt_tokens);
    JsonArray generated_arr = doc.createNestedArray("generated_tokens");
    writeJsonArray(generated_arr, generated);
    doc["next_token"] = next_token;
    doc["prompt"] = prompt;
    doc["max_new_tokens"] = max_new_tokens;
    doc["temperature"] = temperature;
    doc["tokenizer_loaded"] = tokenizer.isLoaded();

    String payload;
    serializeJson(doc, payload);
    Serial.print("@GENERATE|");
    Serial.println(payload);
}

void handleCommand(NanoLLM& model, BPETokenizer& tokenizer, const String& line) {
    if (line.length() == 0) {
        return;
    }

    if (line == "PING") {
        Serial.println("@PONG");
        return;
    }

    if (line == "INFO") {
        ModelConfig cfg = model.getConfig();
        StaticJsonDocument<512> doc;
        doc["vocab_size"] = cfg.vocab_size;
        doc["d_model"] = cfg.d_model;
        doc["n_layers"] = cfg.n_layers;
        doc["n_heads"] = cfg.n_heads;
        doc["d_ff"] = cfg.d_ff;
        doc["max_seq_len"] = cfg.max_seq_len;
        doc["use_moe"] = cfg.use_moe;
        doc["moe_n_experts"] = cfg.moe_n_experts;
        doc["moe_top_k"] = cfg.moe_top_k;
        doc["moe_shared_d_ff"] = cfg.moe_shared_d_ff;
        doc["tokenizer_loaded"] = tokenizer.isLoaded();
        doc["tokenizer_vocab"] = static_cast<int>(tokenizer.getVocabSize());
        doc["free_heap"] = ESP.getFreeHeap();

        String payload;
        serializeJson(doc, payload);
        Serial.print("@INFO|");
        Serial.println(payload);
        return;
    }

    if (line.startsWith("ENCODE|")) {
        String prompt = line.substring(strlen("ENCODE|"));
        std::vector<int> prompt_tokens = encodePrompt(tokenizer, prompt);

        // Large prompts (latency sweeps) exceed a 2KB static doc if we embed
        // the full prompt text; keep metadata small and omit prompt body.
        DynamicJsonDocument doc(4096 + prompt_tokens.size() * 8);
        doc["prompt_chars"] = prompt.length();
        doc["prompt_preview"] = prompt.substring(0, 64);
        JsonArray arr = doc.createNestedArray("prompt_tokens");
        writeJsonArray(arr, prompt_tokens);
        doc["tokenizer_loaded"] = tokenizer.isLoaded();

        String payload;
        serializeJson(doc, payload);
        Serial.print("@ENCODE|");
        Serial.println(payload);
        return;
    }

    if (line.startsWith("GENERATE|")) {
        String args = line.substring(strlen("GENERATE|"));
        int sep1 = args.indexOf('|');
        if (sep1 < 0) {
            Serial.println("@ERROR|usage: GENERATE|<prompt>|<max_new_tokens>");
            return;
        }

        String prompt = args.substring(0, sep1);
        String rest = args.substring(sep1 + 1);
        int max_new_tokens = rest.toInt();
        if (max_new_tokens <= 0) {
            max_new_tokens = 1;
        }

        sendGenerateResponse(model, tokenizer, prompt, max_new_tokens, 1.0f);
        nano_feed_watchdog();
        return;
    }

    Serial.print("@ERROR|unknown command: ");
    Serial.println(line);
}

} // namespace

void serialVerifyAnnounceReady() {
    Serial.println("@READY");
    Serial.println("@HELP|PING, INFO, ENCODE|<text>, GENERATE|<prompt>|<max_new_tokens>");
}

void serialVerifyPoll(NanoLLM& model, BPETokenizer& tokenizer) {
    while (Serial.available() > 0) {
        char ch = static_cast<char>(Serial.read());
        if (ch == '\r') {
            continue;
        }
        if (ch == '\n') {
            handleCommand(model, tokenizer, g_line_buffer);
            g_line_buffer = "";
            continue;
        }
        // Long GENERATE prompts for latency sweeps need >512 chars (S≈100 ≈ 500B).
        if (g_line_buffer.length() < 8192) {
            g_line_buffer += ch;
        }
    }
}
