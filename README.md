# Tesseract

A tensor library written in Zig that can verify the validity of tensor operations (broadcasting, matrix multiplciation) at compile time, and generated optimized code using ahead-of-time knowledge of tensor shapes. 

Tesseract is somewhere between an n-dimensional array library like NumPy, ArrayFire, PyTorch, Tinygrad, etc. and an array processing domain-specific language (DSL) like Halide, TVM, Tiramisu, etc. which has an optimizing compiler. Generated code follows the C ABI in order to be called from any language, enhancing the ability to incorporate Tesseract code into existing codebases that use other languages. 

The goal is to run differentiable tensor operations on a variety of processors/accelerators and to power a high performance deep learning framework.

## Core Principles

### Verification During Compilation

- Trace all operations to build the computation graph
- Verify that all shapes and dtypes for inputs/outputs are valid
- Invalid operations will fail to compile (and provide nice error messages)

### Aggressive Optimization

- Avoid heap allocation as much as possible using predetermined memory requirements
- Minimize usage of slower memory (e.g. global memory on GPUs), inlining of operations
- Optimize for parallelism and/or locality

### Minimal Dependencies, Fast Compilation
- Minimize dependencies by using only Zig compiler + optional external compilers for specific hardware
- Small codebase for better maintainability
- Use codegen or compiler APIs to generate device specific code
- Keep compile times low

## Structure

Tesseract consists of a few prevailing structures:

### Ops 
- Enums for unary (map), binary (zip), reduce ops, and others
- Defines the core set of operations that any hardware abstraction must implement
- To make it easy to write a hardware abstraction, the number of ops is kept small (RISC)
- Each op maps to a function that can be called with tensors as input, and produce an output tensor

### Graph 
- Global variable tracking the computation graph
- Will be used as the initial point for lowering until codegen is performed

### Tensor
- Generic type defined by the element type, shape, and strides
- Contains functions to perform mathematical operations

## Demo

A demo of the library can be found in `demo.zig`.

## Roadmap

### Feature Checklist

- [x] Compile time verification of shapes and strides
    - [x] Contiguousness of strides is checked
    - [x] Broadcasting between arrays
    - [x] Reduction across an array dimension
    - [x] Reshaping operations
- [x] Compute graph
    - [x] Building up the compute graph
    - [x] Generating code to evaluate the compute graph
- [ ] Optimization passes
    - [x] Fusing of map, zip, reduce ops where possible and optimal
    - [ ] Polyhedral analysis and optimization (loop transforms)
    - [ ] Applying machine learning to learn a heuristic for optimization
- [ ] Backpropagation and calculatation of gradients
    - [ ] Automatic creation of backwards graph and optimized codegen
    - [ ] Support for higher order derivatives

### Future Goals
 
- Python interface
    - Use JIT compilation to take advantage of type guards in this library
    - Load and unload data from NumPy / C arrays
    - Support existing deep learning codebases and pipelines 
- Support existing neural network formats like ONNX, PyTorch
    - Make it easy to import models and weights 
- Support for accelerator frameworks like CUDA, Metal, WebGPU, etc.
    - Use codegen or LLVM targets to generate device specific code
- Support for other deep learning compiler IRs like StableHLO, Triton-IR
    - Take advantage of optimizations implemented by teams developing XLA, etc.
