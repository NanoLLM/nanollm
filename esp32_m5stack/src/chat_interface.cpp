#include "chat_interface.h"
#include <algorithm>

namespace {
static constexpr uint16_t kScrollBarColor = 0x7BEF;  // 50% gray in RGB565
}

ChatInterface::ChatInterface() 
    : send_requested(false), scroll_offset(0), status_visible(false) {
}

ChatInterface::~ChatInterface() {
}

void ChatInterface::begin() {
    M5Cardputer.Display.setRotation(1);
    M5Cardputer.Display.setTextSize(FONT_SIZE);
    M5Cardputer.Display.setTextColor(WHITE, BLACK);
    M5Cardputer.Display.fillScreen(BLACK);
    
    // Initialize keyboard
    M5Cardputer.Keyboard.begin();
    
    // Add welcome message
    addMessage("assistant", "Hello! I'm NanoLLM. Type a message and press Enter to chat.");
    
    render();
}

void ChatInterface::loop() {
    handleKeyboard();
    
    // Auto-scroll to bottom if new message added
    if (send_requested) {
        scrollToBottom();
    }
}

void ChatInterface::addMessage(const String& role, const String& text) {
    ChatMessage msg;
    msg.role = role;
    msg.text = text;
    messages.push_back(msg);

    if (Serial) {
        Serial.printf("[chat][%s] %s\n", role.c_str(), text.c_str());
    }
    
    // Limit conversation history to prevent memory issues
    if (messages.size() > 50) {
        messages.erase(messages.begin(), messages.begin() + 10);
    }
    
    scrollToBottom();
    render();
}

void ChatInterface::updateLastMessage(const String& role, const String& text) {
    if (messages.empty() || messages.back().role != role) {
        addMessage(role, text);
        return;
    }

    messages.back().text = text;
    scrollToBottom();
    render();
}

void ChatInterface::clearInput() {
    input_buffer = "";
    render();
}

void ChatInterface::clearConversation() {
    messages.clear();
    scroll_offset = 0;
    addMessage("assistant", "Conversation cleared. How can I help you?");
}

void ChatInterface::showStatus(const String& status) {
    status_message = status;
    status_visible = true;
    render();
}

void ChatInterface::hideStatus() {
    status_visible = false;
    render();
}

void ChatInterface::render() {
    M5Cardputer.Display.fillScreen(BLACK);
    renderMessages();
    renderInputArea();
    if (status_visible) {
        renderStatus();
    }
    renderScrollIndicator();
}

void ChatInterface::renderMessages() {
    int y = 0;
    int visible_start = 0;
    
    // Calculate which messages to show based on scroll
    int total_lines = getTotalLines();
    int max_visible = MAX_VISIBLE_LINES - (status_visible ? 2 : 0);
    
    if (total_lines > max_visible) {
        visible_start = std::max(0, total_lines - max_visible - scroll_offset);
    }
    
    int current_line = 0;
    for (size_t i = 0; i < messages.size(); i++) {
        const ChatMessage& msg = messages[i];
        
        // Calculate lines for this message
        String display_text = (msg.role == "user" ? "You: " : "AI: ") + msg.text;
        int msg_lines = (getTextWidth(display_text) / (SCREEN_WIDTH - 10)) + 1;
        
        if (current_line + msg_lines > visible_start) {
            // Show this message
            uint16_t color = (msg.role == "user") ? CYAN : GREEN;
            int x = (msg.role == "user") ? 5 : 5;
            
            drawTextWrapped(x, y, display_text, color);
            
            // Calculate y for next message
            y += msg_lines * LINE_HEIGHT;
            if (y > SCREEN_HEIGHT - INPUT_AREA_HEIGHT - (status_visible ? STATUS_HEIGHT : 0)) {
                break;
            }
        }
        
        current_line += msg_lines;
    }
}

void ChatInterface::renderInputArea() {
    int input_y = SCREEN_HEIGHT - INPUT_AREA_HEIGHT;
    
    // Draw input box border
    M5Cardputer.Display.drawRect(0, input_y, SCREEN_WIDTH, INPUT_AREA_HEIGHT, WHITE);
    
    // Draw input text
    M5Cardputer.Display.setCursor(2, input_y + 4);
    M5Cardputer.Display.setTextColor(WHITE, BLACK);
    
    String display_input = input_buffer;
    if (display_input.length() > 0) {
        // Truncate if too long
        int max_chars = (SCREEN_WIDTH - 4) / 6;  // Approximate chars per line
        if (display_input.length() > max_chars) {
            display_input = "..." + display_input.substring(display_input.length() - max_chars + 3);
        }
        M5Cardputer.Display.print(display_input);
    }
    
    // Draw cursor
    int cursor_x = 2 + getTextWidth(display_input);
    M5Cardputer.Display.drawLine(cursor_x, input_y + 2, cursor_x, input_y + INPUT_AREA_HEIGHT - 2, WHITE);
}

void ChatInterface::renderStatus() {
    int status_y = SCREEN_HEIGHT - INPUT_AREA_HEIGHT - STATUS_HEIGHT;
    
    // Draw status bar
    M5Cardputer.Display.fillRect(0, status_y, SCREEN_WIDTH, STATUS_HEIGHT, DARKGREY);
    M5Cardputer.Display.setCursor(2, status_y + 1);
    M5Cardputer.Display.setTextColor(WHITE, DARKGREY);
    M5Cardputer.Display.print(status_message);
}

void ChatInterface::renderScrollIndicator() {
    int total_lines = getTotalLines();
    int max_visible = MAX_VISIBLE_LINES - (status_visible ? 2 : 0);
    
    if (total_lines > max_visible) {
        // Show scroll indicator on right side
        int indicator_height = (max_visible * SCREEN_HEIGHT) / total_lines;
        int indicator_y = (scroll_offset * SCREEN_HEIGHT) / total_lines;
        
        M5Cardputer.Display.fillRect(SCREEN_WIDTH - 3, indicator_y, 3, indicator_height, kScrollBarColor);
    }
}

void ChatInterface::handleKeyboard() {
    if (M5Cardputer.Keyboard.isChange()) {
        M5Cardputer.Keyboard.updateKeysState();
        if (M5Cardputer.Keyboard.isPressed()) {
            auto& status = M5Cardputer.Keyboard.keysState();
            
            // Handle special keys
            if (status.enter) {
                if (input_buffer.length() > 0) {
                    send_requested = true;
                }
                return;
            }
            
            if (status.del) {
                if (input_buffer.length() > 0) {
                    input_buffer.remove(input_buffer.length() - 1);
                    render();
                }
                return;
            }
            
            if (status.space) {
                input_buffer += " ";
                render();
                return;
            }
            
            // Handle navigation keys via HID codes
            for (auto hid_key : status.hid_keys) {
                if (hid_key == 0x52) {  // Up arrow
                    scrollUp();
                    return;
                }
                if (hid_key == 0x51) {  // Down arrow
                    scrollDown();
                    return;
                }
            }
            
            // Handle regular characters
            for (auto i : status.word) {
                char c = i;
                if (c >= 32 && c < 127) {  // Printable ASCII
                    input_buffer += c;
                    render();
                }
            }
        }
    }
}

void ChatInterface::handleSpecialKeys(char key) {
    switch (key) {
        case '\n':
        case '\r':
            if (input_buffer.length() > 0) {
                send_requested = true;
            }
            break;
        case '\b':
        case 127:  // Backspace
            if (input_buffer.length() > 0) {
                input_buffer.remove(input_buffer.length() - 1);
                render();
            }
            break;
        case 27:  // ESC - clear input
            clearInput();
            break;
    }
}

String ChatInterface::wrapText(const String& text, int max_width) {
    // Simple wrapping - just return as is, rendering will handle it
    return text;
}

void ChatInterface::drawTextWrapped(int x, int y, const String& text, uint16_t color) {
    M5Cardputer.Display.setCursor(x, y);
    M5Cardputer.Display.setTextColor(color, BLACK);
    
    int max_width = SCREEN_WIDTH - x - 10;
    int char_width = 6;  // Approximate
    int chars_per_line = max_width / char_width;
    
    int current_x = x;
    int current_y = y;
    String line = "";
    
    for (int i = 0; i < text.length(); i++) {
        char c = text.charAt(i);
        
        if (c == '\n' || line.length() >= chars_per_line) {
            // Print current line
            M5Cardputer.Display.setCursor(current_x, current_y);
            M5Cardputer.Display.print(line);
            
            // Move to next line
            current_y += LINE_HEIGHT;
            line = "";
            
            if (c == '\n') {
                continue;
            }
        }
        
        line += c;
    }
    
    // Print remaining line
    if (line.length() > 0) {
        M5Cardputer.Display.setCursor(current_x, current_y);
        M5Cardputer.Display.print(line);
    }
}

int ChatInterface::getTextWidth(const String& text) {
    return text.length() * 6;  // Approximate character width
}

int ChatInterface::getTextHeight(const String& text) {
    int width = getTextWidth(text);
    int max_width = SCREEN_WIDTH - 10;
    return ((width / max_width) + 1) * LINE_HEIGHT;
}

void ChatInterface::scrollUp() {
    int total_lines = getTotalLines();
    int max_visible = MAX_VISIBLE_LINES - (status_visible ? 2 : 0);
    
    if (total_lines > max_visible) {
        scroll_offset = std::min(scroll_offset + 1, total_lines - max_visible);
        render();
    }
}

void ChatInterface::scrollDown() {
    if (scroll_offset > 0) {
        scroll_offset--;
        render();
    }
}

void ChatInterface::scrollToBottom() {
    scroll_offset = 0;
}

int ChatInterface::getTotalLines() {
    int total = 0;
    for (const auto& msg : messages) {
        String display_text = (msg.role == "user" ? "You: " : "AI: ") + msg.text;
        total += (getTextWidth(display_text) / (SCREEN_WIDTH - 10)) + 1;
    }
    return total;
}

