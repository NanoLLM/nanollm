#pragma once

#include <Arduino.h>
#include <esp_system.h>
#include <esp_task_wdt.h>

/** Reset the task WDT and yield so loopTask stays alive during long inference. */
inline void nano_feed_watchdog() {
    yield();
    esp_task_wdt_reset();
}

inline const char* nano_reset_reason_string(esp_reset_reason_t reason) {
    switch (reason) {
        case ESP_RST_POWERON: return "power-on";
        case ESP_RST_EXT: return "external pin";
        case ESP_RST_SW: return "software";
        case ESP_RST_PANIC: return "exception/panic";
        case ESP_RST_INT_WDT: return "interrupt WDT";
        case ESP_RST_TASK_WDT: return "task WDT (loop blocked too long)";
        case ESP_RST_WDT: return "other WDT";
        case ESP_RST_DEEPSLEEP: return "deep sleep wake";
        case ESP_RST_BROWNOUT: return "brownout (power)";
        case ESP_RST_SDIO: return "sdio";
        default: return "unknown";
    }
}

inline void nano_log_reset_reason() {
    esp_reset_reason_t reason = esp_reset_reason();
    Serial.printf("[system] reset_reason=%s (%d)\n", nano_reset_reason_string(reason), static_cast<int>(reason));
}
