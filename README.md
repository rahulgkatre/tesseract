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
- Each op maps to a function defined by each backend to actually perform the computation

### Backends 
- Provide implementations of ops, and higher level functions that are composed of multiple ops
- Use device specific APIs to provide an interface for managing memory (allocation, freeing)
- The implementations of ops will directly manipulate data in the storage

### Storage
- A reserved section of memory that contains the data for a tensor in a 1D format
- Highly coupled to a single backend, each backend must provide a Storage implementation
- Storage does not exist at compile time, as no memory can be allocated during compile time evaluation

### Tensor / TensorView
- Generic type defined by the element type, shape, and strides
- Contains a backend and the backend's associated storage 
- Provides a multidimensional view into the 1 dimensional storage using shape and strides
- Contains functions to perform operations using the backend
- Makes up the computation graph defined by input tensor(s) and the op applied to them

## Demo

A demo of the library can be found in `demo.zig`. In general to use the library your code would look something like this:

```zig
const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("src/tensor.zig").Tensor;
const Backend = @import("src/backend.zig").Backend;

pub fn main() !void {
    // All tensor code must be in comptime
    const Backend = 
    const out = comptime blk: {
        const x1 = fn1();
        const x2 = fn2(x1);
        break :blk x2;
    };

    // Initialize the backend which will allow for allocation of tensor storage
    TestBackend.runtime(.{});
    defer TestBackend.finished();

    // Print the storage to show the data
    const eval_out = out.eval();
    std.debug.print("\n{any}\n", .{eval_out.storage});
}
```

## Roadmap

### First Milestone

- [x] Compile time verification of shapes and strides
    - [x] Contiguousness of strides is checked
    - [x] Check if a broadcast between two arrays of different shapes are valid
    - [x] Reducing a tensor along a dim yields a new tensor type where the reduce dim is now 1
    - [x] Operations for reshaping, permuting, flattening, etc. a tensor
- [x] Building the compute graph
    - [x] When a function is called, the output tensor receives a closure to evaluate itself
- [x] Using the compute graph to perform computations on tensors
    - [x] Implementations of ops in Zig
    - [ ] Generating code to evaluate the compute graph
- [ ] Optimization passes
    - [ ] Automated fusion of operators to reduce memory bandwidth requirements
- [ ] Backpropagation and calculatation of gradients


### Future Goals
 
- Python interface
    - Use JIT compilation to take advantage of type guards in this library
    - Use Python buffer protocol to load and unload data
    - Support existing deep learning codebases and pipelines 
- Support existing neural network formats like ONNX, PyTorch
    - Make it easy to import models and weights 
- Support for accelerator frameworks like CUDA, Metal, WebGPU, etc.
    - Use codegen or LLVM targets to generate device specific code
- Support for other deep learning compiler IRs like StableHLO, Triton-IR
    - Take advantage of optimizations implemented by teams developing XLA, etc.
