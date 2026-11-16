# Neural Network Framework in C++

This is a personal project focused on learning and practicing what I've seen in class and
on the internet. My goal is to create a light, highly optimized framework that is able to
rival PyTorch, so I avoid using it without losing practicity (Plus, I am more passionate
at low-level stuff). Built with modern C++ (C++26) and Eigen for linear algebra.

## Features

- Dense (Fully connected) layers behind an abstract `Layer` interface, ready for
  convolutional / attention layers.
- Activations: ReLU, Sigmoid, Tanh, Identity - each bundled with its derivative and a
  matching initialization scheme (He / Xavier).
- Losses: Mean Squared Error (Cross-Entropy is next).
- Optimizers: SGD and Adam (Kingma & Ba, 2015) with per-layer state and bias correction.
- Shape validation with clear error messages instead of silent corruption.
- Reproducible runs: every layer accepts an optional seed.
- Test suite with gradient checking (finite differences) wired to CTest.

## Project layout

```
include/            Public headers (the library API)
  Activation.hpp      Activation + derivative + init scheme
  Layer.hpp           Abstract layer interface
  DenseLayer.hpp      Fully connected layer
  Loss.hpp            MSE and its gradient
  Network.hpp         Layer stack + optimizer orchestration
  Optimizer.hpp       SGD and Adam
src/                Library implementation
examples/           Runnable demos (XOR)
tests/              Unit tests (activations, gradient check, training)
external/eigen/     Eigen (git submodule)
```

## Requirements

- A compiler with C++26 support: GCC 14+ or Clang 18+.
- CMake 3.25 or higher.
- Eigen - bundled as a git submodule, nothing else to install.

## Build & run

```bash
git clone <repo-url>
cd cpp-neural-network
git submodule update --init --recursive

# Configure + build (Release, -O3, recommended for training & benchmarks)
cmake --preset release
cmake --build --preset release

# Run the test suite
ctest --preset release

# Run the XOR demo
./build/release/examples/xor_demo
```

For development there is also a `debug` preset with AddressSanitizer and
UBSan enabled:

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

## Roadmap

1. ~~Prepare working environment, tools and knowledge.~~
2. ~~Create the basics: activations, loss, forward pass.~~
3. ~~Work on the network: layer, network, optimizer, backpropagation.~~
4. ~~Training examples and tests (XOR demo + gradient checking).~~
5. Expand the framework: cross-entropy + softmax, more optimizers (momentum, RMSProp),
   mini-batch shuffling, serialization.

  `And in the near future...`

6. New layer types: `Conv2D` (im2col + GEMM), pooling, embeddings, attention (a first
   transformer block for NLP).
7. Performance: OpenMP / BLAS back-end, then CUDA.
8. Benchmark against PyTorch on equal hardware.

## License

MIT - see [LICENSE](LICENSE).
