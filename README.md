# Tesseract

Tesseract is a tensor library / embedded domain specific language (eDSL). Its defining feature is that it can verify the validity of tensor operations (broadcasting, matrix multiplciation) at compile time, and generated optimized code using ahead-of-time knowledge of tensor shapes. Zig was chosen specifically for this purpose due to its compile time type functions, which Tesseract makes heavy use of to reason about the shapes and strides of tensors at compile time to provide memory safety and compile time checks. 

Tesseract is somewhere between an n-dimensional array library like NumPy, ArrayFire, PyTorch, Tinygrad, etc. and an array processing domain-specific language (DSL) like Halide, TVM, Tiramisu, etc. which has an optimizing compiler. Tesseract code does not get executed, but is instead lowered to a compute graph, then a loop or tile based IR, and finally source code or a compiler specific IR. 

Generated code comes in the form of a .so that follows the C ABI in order to be called from any language, enhancing the ability to incorporate Tesseract code into existing codebases via FFI / C interfaces. The end goal of Tesseract is to run differentiable tensor operations with full utilization of a variety processors/accelerators features and power a high performance deep learning framework for training and inference.

## Core Principles

### Verification During Compilation

- Use type system to keep track of dtype and shape of tensors
- Verify that dtype and shape for inputs are valid
- Invalid operations will fail to compile and provide helpful error messages

### Optimized Code Generation

- Create pseudo define-by-run computation graph
- Apply graph optimizations such as operator fusion
- Apply memory access pattern optimization such as loop tiling, skewing, unrolling
- Apply hardware intrinsic usage such as vectorization, MMA intrinsics
- No heap memory, everything should be statically allocated
- Generate code (compiled to .so or .dll) or IR (e.g. LLVM IR)

### Minimal Dependencies, Small Footprint

- Minimize compile dependencies by only using the Zig compiler (as the language develops, use it more)
- Minimize executable dependencies (including standard library)
- Avoid usage of handwritten kernel libraries

## Internals

### How it works

- Use a Torch-like API to create Tensors (actually LazyTensors) and define operations on tensors at compile time
- Computation graph is implicitly created which can be used to create the backwards graph for training
- Computation graph is converted to a schedule graph where Halide-like API is used to define optimizations (which will be autotunable)
- Optimized schedule graph is lowered and compiled to a .so or .dll file
- To run, the caller passes buffers (e.g. NumPy) containing input data, model params, and pre-allocated output buffer to t

## Roadmap

### MVP Feature Checklist

- Shape-based types
    - [x] Broadcasting, reduction
    - [x] Reshaping, re-striding
    - [x] Matrix multiplication
    - [ ] Convolution
    - [ ] Symbolic shapes and strides
- Computation graph 
    - [x] Building up the compute graph
    - [ ] Operator fusion
    - [ ] Generating code to evaluate the compute graph
    - [ ] Automatic differentiation, backwards compute graph
- Optimization passes
    - [x] Fusing of map, zip, reduce ops where possible and optimal
    - [ ] Polyhedral analysis and optimization of nested loops
    - [ ] Automatic parallelization with OpenMP for CPU and GPU groups, blocks, warps
    - [ ] Applying machine learning to learn a heuristic for optimization

### Future Goals

- Support for accelerator frameworks like CUDA, HIP/ROCm, Metal, WebGPU, etc.
    - Use codegen or LLVM targets to generate device specific code
    - Target as many platform specific instructions/functions as possible (e.g. SIMD, FMA, MMA)
- Nerual network library
    - Module architecture
    - Drop-in replacement for PyTorch (the nn namespace)
    - Support existing neural network formats like ONNX, PyTorch
    - Convert files to Tesseract tensor definitions
- Python/C interface
    - Use JIT compilation to take advantage of type guards in this library
    - Load and unload data from NumPy / C array to support existing codebases and pipelines
- Support for other deep learning compiler IRs like StableHLO, MLIR, Triton-IR
    - Be able to run Tesseract on closed-source hardware (e.g. TPU)
- Distributed computing
    - Multiple GPUs
    - Multiple computers with multiple GPUs
