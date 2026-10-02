#ifndef CHAT_INTERFACE_H
#define CHAT_INTERFACE_H

#include <M5Cardputer.h>
#include <vector>

// Chat message structure
struct ChatMessage {
    String role;  // "user" or "assistant"
    String text;
};

class ChatInterface {
public:
    ChatInterface();
    ~ChatInterface();
    
    // Initialize the interface
    void begin();
    
    // Main loop - handles keyboard input and display
    void loop();
    
    // Add a message to the chat
    void addMessage(const String& role, const String& text);

    // Replace the most recent message for streaming updates
    void updateLastMessage(const String& role, const String& text);
    
    // Get the current input buffer
    String getInputBuffer() const { return input_buffer; }
    
    // Clear input buffer
    void clearInput();
    
    // Check if user wants to send (Enter pressed)
    bool isSendRequested() const { return send_requested; }
    
    // Clear send flag
    void clearSendFlag() { send_requested = false; }
    
    // Get conversation history for model
    std::vector<ChatMessage> getHistory() const { return messages; }
    
    // Clear conversation
    void clearConversation();
    
    // Show status message
    void showStatus(const String& status);
    
    // Hide status
    void hideStatus();

private:
    // Display settings
    static const int SCREEN_WIDTH = 240;
    static const int SCREEN_HEIGHT = 135;
    static const int FONT_SIZE = 1;
    static const int LINE_HEIGHT = 8;
    static const int MAX_VISIBLE_LINES = 15;
    static const int INPUT_AREA_HEIGHT = 16;
    static const int STATUS_HEIGHT = 10;
    
    // Chat state
    std::vector<ChatMessage> messages;
    String input_buffer;
    bool send_requested;
    int scroll_offset;
    String status_message;
    bool status_visible;
    
    // Display functions
    void render();
    void renderMessages();
    void renderInputArea();
    void renderStatus();
    void renderScrollIndicator();
    
    // Input handling
    void handleKeyboard();
    void handleSpecialKeys(char key);
    
    // Text utilities
    String wrapText(const String& text, int max_width);
    void drawTextWrapped(int x, int y, const String& text, uint16_t color);
    int getTextWidth(const String& text);
    int getTextHeight(const String& text);
    
    // Scrolling
    void scrollUp();
    void scrollDown();
    void scrollToBottom();
    int getTotalLines();
};

#endif // CHAT_INTERFACE_H

