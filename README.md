# Tesseract

Tesseract is a tensor library written in Zig. Its defining feature is that it can verify the validity of tensor operations (broadcasting, matrix multiplciation) at compile time, and generated optimized code using ahead-of-time knowledge of tensor shapes. Zig was chosen specifically for this purpose due to its compile time type functions, which Tesseract makes heavy use of to reason about the shapes and strides of tensors at compile time to provide memory safety and compile time checks. 

Tesseract is somewhere between an n-dimensional array library like NumPy, ArrayFire, PyTorch, Tinygrad, etc. and an array processing domain-specific language (DSL) like Halide, TVM, Tiramisu, etc. which has an optimizing compiler. Tesseract code does not get executed, but is instead lowered to a compute graph, then a loop or tile based IR, and finally source code or a compiler specific IR. 

Generated code comes in the form of a library (.so) follows the C ABI in order to be called from any language, enhancing the ability to incorporate Tesseract code into existing codebases that use other languages via FFI / C interfaces. The end goal of Tesseract is to run differentiable tensor operations on a variety of processors/accelerators and to power a high performance deep learning framework.

## Core Principles

### Verification During Compilation

- Trace all operations to build the computation graph
- Dynamic computation graph, no manual graph calls needed
- Verify that all shapes and dtypes for inputs/outputs are valid
- Invalid operations will fail to compile and provide nice error messages

### Aggressive Optimization

- Avoid heap allocation entirely through predetermined array sizes
- Use bump arena allocators when an allocator is needed (e.g. GPU)
- Minimize usage of slower memory, automatically inline ops
- Optimize loops for maximum throughput

### Minimal Dependencies, Fast Compilation
- Minimize dependencies by using only Zig compiler including self-hosted
- Access other compilers via C API or build commands
- Small codebase for better maintainability
- Use codegen or compiler APIs to generate device specific code
- Keep compile times low so that external JIT compilation is fast

## Internals

### How it works

- Developer has access to a set of operations (`functions.zig` and `tensor.zig`) for tensor manipulation and computation
- Developer defines tensors and operations in comptime blocks or functions
- For each new tensor, a runtime callback is created to add the operation and tensor to the graph
- The runtime callback is recursively called to build the entire graph
- Backwards graph is built in a similar way with a backwards callback
- The graph is progressively lowered and eventually compiled to executable code

## Roadmap

### MVP Feature Checklist

- Compile time shapes/strides
    - [x] Broadcasting, reduction
    - [x] Reshaping, re-striding
    - [x] Matrix multiplication
    - [ ] Convolution
    - [ ] Symbolic shapes and strides
- Computation graph 
    - [x] Building up the compute graph
    - [x] Generating code to evaluate the compute graph
    - [ ] Automatic differentiation, backwards compute graph
- Optimization passes
    - [x] Fusing of map, zip, reduce ops where possible and optimal
    - [ ] Polyhedral analysis and optimization of nested loops
    - [ ] Automatic parallelization with OpenMP for CPU and GPU groups, blocks, warps
    - [ ] Applying machine learning to learn a heuristic for optimization

### Future Goals

- Support for accelerator frameworks like CUDA, HIP/ROCm, Metal, WebGPU, etc.
    - Use codegen or LLVM targets to generate device specific code
    - Target as many platform specific instructions/functions as possible (e.g. fused muladd / WMMA)
- Support for other deep learning compiler IRs like StableHLO, MLIR, Triton-IR
    - Take advantage of optimizations implemented by teams developing XLA, etc.
    - Be able to run Tesseract on closed-source hardware (e.g. TPU)
- Python/C interface
    - Use JIT compilation to take advantage of type guards in this library
    - Load and unload data from NumPy / C array to support existing codebases and pipelines
- Nerual network library
    - Module architecture
    - Drop-in replacement for PyTorch (the nn namespace)
    - Support existing neural network formats like ONNX, PyTorch
    - Convert files to Tesseract tensor definitions
- Distributed computing
    - Multiple GPUs
    - Multiple computers with multiple GPUs
