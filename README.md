# Tesseract

Tesseract is a tensor library written in Zig. Its defining feature is that it can verify the validity of tensor operations (broadcasting, matrix multiplciation) at compile time, and generated optimized code using ahead-of-time knowledge of tensor shapes. Zig was chosen specifically for this purpose due to its compile time type functions, which Tesseract makes heavy use of to reason about the shapes and strides of tensors at compile time to provide memory safety and compile time checks. 

Tesseract is somewhere between an n-dimensional array library like NumPy, ArrayFire, PyTorch, Tinygrad, etc. and an array processing domain-specific language (DSL) like Halide, TVM, Tiramisu, etc. which has an optimizing compiler. Tesseract code does not get executed, but is instead lowered to 2 intermediate representations, and a final source / IR file that gets compiled. 

Generated code comes in the form of a library (.so) follows the C ABI in order to be called from any language, enhancing the ability to incorporate Tesseract code into existing codebases that use other languages via FFI / C interfaces. The end goal of Tesseract is to run differentiable tensor operations on a variety of processors/accelerators and to power a high performance deep learning framework.

## Core Principles

### Verification During Compilation

- Trace all operations to build the computation graph
- Verify that all shapes and dtypes for inputs/outputs are valid
- Invalid operations will fail to compile (and provide nice error messages)

### Aggressive Optimization

- Avoid heap allocation entirely (array sizes are predetermined)
- Minimize usage of slower memory, automatically inline
- Optimize loops for parallelism and locality

### Minimal Dependencies, Fast Compilation
- Minimize dependencies by using only Zig compiler (including self-hosted)
- Access other compilers via C API or build commands
- Small codebase for better maintainability
- Use codegen or compiler APIs to generate device specific code
- Keep compile times low so that external JIT compilation is fast

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
    - Target as many platform specific instructions/functions as possible (e.g. fused mul add)
- Support for other deep learning compiler IRs like StableHLO, MLIR, Triton-IR
    - Take advantage of optimizations implemented by teams developing XLA, etc.
    - Be able to run Tesseract on closed-source hardware (e.g. TPU)
- Python/C interface
    - Use JIT compilation to take advantage of type guards in this library
    - Load and unload data from NumPy / C arrays
    - Support existing deep learning codebases and pipelines like OpenCV
- Nerual network library
    - Module architecture
    - Drop-in replacement for PyTorch (the nn namespace)
    - Support existing neural network formats like ONNX, PyTorch
    - Convert files to Tesseract tensor definitions
- Distributed computing
    - Multiple GPUs
    - Multiple computers with multiple GPUs
