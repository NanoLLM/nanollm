cmake_minimum_required(VERSION 3.10)
project(NanoLLM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags for optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# Common sources shared by executables
set(COMMON_SOURCES
    model.cpp
    bpe_tokenizer.cpp
)

# Executable
add_executable(inference ${COMMON_SOURCES} inference.cpp)

# Test executable
add_executable(test_inference ${COMMON_SOURCES} test_inference.cpp)
target_compile_definitions(test_inference PRIVATE TEST_MODE=1)

# Utility executable for emitting next-token predictions (used by tests)
add_executable(dump_next_token ${COMMON_SOURCES} dump_next_token.cpp)
add_executable(dump_cached_tokens ${COMMON_SOURCES} dump_cached_tokens.cpp)

# For ESP32, you would use:
# - ESP-IDF build system
# - Or PlatformIO
# - Or Arduino framework
# This CMakeLists.txt is for desktop/testing

# Example ESP32 configuration would require:
# - Different memory management
# - Flash storage for weights
# - Different I/O (SPIFFS/LittleFS for file system)
# - Reduced precision operations

