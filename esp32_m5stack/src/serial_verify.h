#ifndef NANOLLM_SERIAL_VERIFY_H
#define NANOLLM_SERIAL_VERIFY_H

#include <Arduino.h>
#include "model_esp32.h"
#include "bpe_tokenizer_esp32.h"

// Line-oriented serial protocol for cross-checking against desktop PyTorch.
// Commands (newline-terminated):
//   PING
//   INFO
//   ENCODE|<text>
//   GENERATE|<prompt>|<max_new_tokens>
//
// Responses (newline-terminated):
//   @READY
//   @PONG
//   @INFO|<json>
//   @ENCODE|<json>
//   @GENERATE|<json>
//   @ERROR|<message>

void serialVerifyAnnounceReady();
void serialVerifyPoll(NanoLLM& model, BPETokenizer& tokenizer);

#endif
