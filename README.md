# Tesseract

A tensor library written in Zig that features compile time verification of all tensor operations and soon, compute graph optimization. The goal is to be able to run differentiable tensor operations on a variety of processors/accelerators and to power a deep learning framework. 

## Core Principles

### Do as much as possible at compile time

- Take advantage of Zig's compile time evaluation to trace tensor operations
- Verify that all shapes and dtypes for inputs/outputs are valid using Zig's type system and generics
- Errors involving broadcasting between two tensors will no longer exist at runtime

### Laziness

- Tensors are lazy, no data is allocated or operated on until requested
- This builds on the side effects of compile time evaluation, as data does not exist at compile time
- The operations still "run" (in compile time) to produce output tensor metadata
- A function can be called on the output tensor to evaluate the compute graph

### Efficiency
- Avoid heap allocation as much as possible, use compile time evaluation to predetermine memory requirements
- The compute graph will be static and storage for batches of training data can be allocated once. 
- Fuse operations to reduce memory bandwidth requirements during inference

### Codegen

- Required in order to do compute graph rewriting when fusing operations 
- Take advantage of Zig's incremental compilation to generate a source file or IR code
- Call a more specialized compiler for the desired output

### Acceleration
- Compile models for high performance accelerator hardware
- Some architectures may be supported through Zig's compiler
- Other architectures, IRs for other compilers, etc. can be supported through codegen

### Run anywhere
- No dependencies required, unless targeting a specific accelerator
- Ships with a Zig backend that only uses builtins and standard library
- If Zig can compile for a device, Tesseract can run on it

## Structure

The library consists of a few prevailing structures: ops, backends, storage, and tensors.

### Ops 
- Enums for unary (map), binary (zip), reduce ops, and others
- Defines the core set of operations that any backend must implement
- To make it easy to write a backend, the number of ops is kept small
- Each op maps to a function that can be called with tensors

### Graph 
- Global variable tracking the computation graph
- Will be used as the initial point for lowering until codegen is performed

### Tensor / TensorView
- Generic type defined by the element type, shape, and strides
- Contains functions to perform mathematical operations

## Demo

A demo of the library can be found in `demo.zig`.

```zig
const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor = @import("src/tensor.zig");

// Example of a softmax
fn softmax(x: anytype, comptime dim: u8) @TypeOf(x) {
    const max = x.max(null);
    const x_minus_max = x.sub(max);
    const exp = x_minus_max.exp();
    const sumexp = exp.sum(dim);
    const sm = x_minus_max.div(sumexp);
    return sm;
}

const Graph = @import("src/Graph.zig");
pub fn main() !void {
    // Initialize the global graph
    Graph.init();
    defer Graph.deinit();

    // All tensor code should must be in comptime
    const out = comptime blk: {
        const x = tensor.Tensor(f32, .{ 2, 16 }).full(3);
        break :blk softmax(x, 1);
    };

    // Call trace on the output to build its computation graph
    out.trace();

    // Show the graph
    Graph.show();
}
```

## Roadmap

### First Milestone

- [x] Compile time verification of shapes and strides
    - [x] Contiguousness of strides is checked
    - [x] Check if a broadcast between two arrays of different shapes are valid
    - [x] Reducing a tensor along a dim yields a new tensor type where the reduce dim is now 1
    - [x] Operations for reshaping, permuting, flattening, etc. a tensor
- [x] Compute graph
    - [x] When a function is called, the output tensor receives a closure with references to input tensors
    - [x] Building up the compute graph
    - [x] Generating code to evaluate the compute graph
- [ ] Optimization passes
    - [ ] Fusing of map, zip, reduce ops where possible and optimal
    - [ ] Loop transformations to improve cache performance
- [ ] Backpropagation and calculatation of gradients
    - [ ] Automatic creation of backwards graph and optimized codegen
    - [ ] Support for higher order derivatives

### Future Goals
 
- Python interface
    - Use JIT compilation to take advantage of type guards in this library
    - Load and unload data from NumPy arrays
    - Support existing deep learning codebases and pipelines 
- Support existing neural network formats like ONNX, PyTorch
    - Make it easy to import models and weights 
- Support for accelerator frameworks like CUDA, Metal, WebGPU, etc.
    - Use codegen or LLVM targets to generate device specific code
- Support for other deep learning compiler IRs like StableHLO, Triton-IR
    - Take advantage of optimizations implemented by teams developing XLA, etc.
