# Testing Guide

This document describes the testing infrastructure for NanoLLM.

## End-to-End Test

The `test_end_to_end.py` script verifies the complete pipeline from training to deployment.

### Running the Test

```bash
python test_end_to_end.py
```

### What It Tests

1. **Training** - Trains a minimal model (2 epochs, tiny dimensions)
2. **Weight Export** - Exports PyTorch weights to C++ binary format
3. **Python Inference** - Tests inference with the trained model
4. **C++ Build** - Compiles the C++ inference code
5. **C++ Inference** - Runs inference using the C++ implementation

### Test Output

The test will:
- Create temporary directories (`test_checkpoints/`, `test_weights/`)
- Train a small model
- Export weights
- Test both Python and C++ inference
- Clean up temporary files
- Print a summary of all tests

### Expected Results

All tests should pass. The test uses very small model dimensions (d_model=32) to run quickly.

### Troubleshooting

**Training fails:**
- Check PyTorch installation
- Verify CUDA is available (if using GPU)
- Check available disk space

**C++ build fails:**
- Ensure CMake is installed
- Check C++17 compiler is available
- Verify all dependencies are present

**C++ inference fails:**
- Check that weights were exported correctly
- Verify model files exist in test_weights/
- Check file paths are correct

## C++ Unit Tests

The C++ test suite (`cpp/test_inference.cpp`) provides unit-level testing.

### Building Tests

```bash
cd cpp
mkdir build && cd build
cmake ..
make test_inference
```

### Running Tests

```bash
./test_inference
```

### Test Coverage

- Model loading from binary files
- Text generation
- Memory usage validation

Note: Tests will skip if model files are not found (this is expected if you haven't trained a model yet).

## Manual Testing

### Python Inference

```bash
python example_inference.py \
    --checkpoint checkpoints/model_best.pt \
    --prompt "Hello" \
    --max_tokens 50
```

### C++ Inference

```bash
cd cpp/build
./inference ../../weights/model.bin ../../weights/model_config.json "Hello" 50
```

### ESP32 Testing

1. Build and upload the ESP32 project
2. Upload model files to SPIFFS
3. Power on the device
4. Verify text generation and display

## Continuous Integration

For CI/CD pipelines, the end-to-end test can be run as part of the build process:

```yaml
# Example GitHub Actions
- name: Run tests
  run: python test_end_to_end.py
```

## Performance Benchmarks

The test suite doesn't include performance benchmarks, but you can add them:

- Inference latency (tokens/second)
- Memory usage during inference
- Model size verification
- Accuracy metrics (if you have a test dataset)

## Adding New Tests

To add new tests:

1. **Python tests**: Add functions to `test_end_to_end.py`
2. **C++ tests**: Add test cases to `cpp/test_inference.cpp`
3. **Integration tests**: Create new test scripts in the project root

## Test Data

The end-to-end test uses a small synthetic dataset. For more comprehensive testing:

1. Use a larger, real dataset
2. Add validation metrics
3. Test edge cases (empty prompts, long sequences, etc.)
4. Test error handling

