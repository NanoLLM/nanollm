# Chat Interface for M5Stack Cardputer

An intuitive chat-like interface for interacting with the NanoLLM language model on the M5Stack Cardputer.

## Features

- **Interactive Chat**: Type messages and get AI responses
- **Conversation History**: See previous messages in the chat
- **Keyboard Input**: Full QWERTY keyboard support
- **Visual Feedback**: Color-coded messages (Cyan for user, Green for AI)
- **Status Indicators**: Shows "Generating..." while model is working
- **Scroll Support**: Arrow keys to scroll through long conversations
- **Input Area**: Dedicated input box at bottom of screen

## Usage

### Starting a Chat

1. Power on the Cardputer
2. Wait for model to load (shows "Model ready!" status)
3. Welcome message appears: "Hello! I'm NanoLLM. Type a message and press Enter to chat."

### Typing Messages

- **Type normally**: Use the keyboard to type your message
- **Send**: Press **Enter** to send your message
- **Backspace**: Press **Del** to delete characters
- **Clear input**: Press **ESC** to clear current input

### Scrolling

- **Up Arrow**: Scroll up through conversation history
- **Down Arrow**: Scroll down (back to recent messages)

### Screen Layout

```
┌─────────────────────────────┐
│                             │
│  AI: Hello! I'm NanoLLM...  │  ← Green text
│                             │
│  You: Hello                 │  ← Cyan text
│  AI: Hi there!              │
│                             │
│  You: How are you?          │
│  AI: I'm doing well...      │
│                             │
│  [Status: Generating...]    │  ← Status bar (when active)
│  ┌───────────────────────┐ │
│  │ Type your message...   │ │  ← Input area
│  └───────────────────────┘ │
└─────────────────────────────┘
```

## Controls

| Key | Action |
|-----|--------|
| **Enter** | Send message |
| **Del** | Delete character |
| **ESC** | Clear input |
| **Up Arrow** | Scroll up |
| **Down Arrow** | Scroll down |
| **Space** | Space character |
| **All other keys** | Type normally |

## Display

- **Screen Size**: 240x135 pixels
- **Font**: Small (1x) for maximum text
- **Colors**:
  - User messages: Cyan
  - AI messages: Green
  - Input area: White border
  - Status bar: Dark grey background

## Memory Management

- Conversation history limited to 50 messages
- Oldest messages automatically removed when limit reached
- Each message stored as role + text string

## Customization

### Change Welcome Message

Edit `chat_interface.cpp`:

```cpp
addMessage("assistant", "Your custom welcome message here");
```

### Adjust Generation Length

Edit `main.cpp`:

```cpp
std::vector<int> generated = model.generate(prompt_tokens, 50, 1.0f);
//                                                      ^^
//                                              Change this number
```

### Change Colors

Edit `chat_interface.cpp` in `renderMessages()`:

```cpp
uint16_t color = (msg.role == "user") ? YELLOW : MAGENTA;
```

## Troubleshooting

### Keyboard not responding
- Check that `M5Cardputer.Keyboard.begin()` is called in `begin()`
- Verify keyboard is properly connected

### Messages not displaying
- Check screen initialization
- Verify text wrapping is working
- Check memory usage (may need to reduce history limit)

### Input area not visible
- Check `INPUT_AREA_HEIGHT` constant
- Verify screen height calculations

### Scrolling not working
- Check arrow key handling in `handleKeyboard()`
- Verify `scroll_offset` is being updated

## Technical Details

### ChatInterface Class

- **Header**: `chat_interface.h`
- **Implementation**: `chat_interface.cpp`
- **Main integration**: `main.cpp`

### Key Methods

- `begin()`: Initialize interface
- `loop()`: Handle keyboard input
- `addMessage()`: Add message to chat
- `render()`: Redraw entire screen
- `handleKeyboard()`: Process keyboard events

### Message Storage

Messages stored in `std::vector<ChatMessage>`:
```cpp
struct ChatMessage {
    String role;  // "user" or "assistant"
    String text;
};
```

## Example Conversation

```
AI: Hello! I'm NanoLLM. Type a message and press Enter to chat.

You: What is AI?
AI: AI stands for artificial intelligence...

You: Tell me a joke
AI: Why did the AI go to school? To improve its neural network!

You: Thanks!
AI: You're welcome! Happy to help.
```

Enjoy chatting with NanoLLM on your Cardputer! 🚀

