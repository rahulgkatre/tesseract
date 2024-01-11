# deepzig

A deep learning library written in Zig that features compile time verification of all tensor operations and advanced acceleration and optimization. 

## Core Principles

### Do as much as possible at compile time

- Take advantage of Zig's compile time evaluation to evaluate tensor operations
- Verify that all shapes and dtypes for inputs/outputs are valid using Zig's type system and generics
- Errors involving broadcasting between two tensors will no longer exist at runtime

### Laziness

- Tensors are lazy, no data is modified until later
- This is a side effect of compile time evaluation, as data does not exist at compile time 
- The operations still "run" to produce output tensor metadata
- A function can be called on the output tensor to evaluate the compute graph

### Efficiency
- Avoid heap allocation as much as possible
- The compute graph will be static and storage for batches of training data can be allocated once. 
- Fuse operations to reduce memory bandwidth requirements

### Acceleration
- Interface with high performance hardware by compiling Zig code or through codegen
- Direct compilation depends heavily on the Zig language's support through LLVM
- Codegen will emit code that will be compiled by an external compiler
- Calls compiler a C API to compile and run on the device

### Run anywhere
- No dependencies required, unless targeting a specific accelerator
- The library ships with a Zig backend for tensor operations that only uses builtins
- If Zig can compile for a device, code written with this library can run on it

### Why Zig?

I tried a few languages (Nim, Cython) but ultimately chose Zig

- **Compile time code execution via comptime** 
    - Enables simplification, verification, and optimization
- **SIMD via @Vector**
    - Simple API for SIMD that is built into the langugage
- **C interop via @cImport** 
    - Zig can natively import C/C++ files and compile with them
    - This is useful for importing neural accelrator C APIs

## Structure

The library consists of a few prevailing structures: ops, backends, storage, and tensors.

### Ops 
- Enums for unary (map), binary (zip), reduce ops, and others
- Defines the core set of operations that any backend must implement
- To make it easy to write a backend, the number of ops is kept small
- Can be mapped to a function to actually perform the computation

### Backends 
- Provide implementations of ops, and higher level functions that are composed of multiple ops
- Use device specific APIs to provide an interface for managing memory (allocation, freeing)
- The implementations of ops will directly manipulate data in the storage

For example, when writing a CUDA backend, the ops might be implemented as CUDA kernels that manipulate data in the CUDA device's memory. 

### Storage
- A reserved section of memory that contains the data for a tensor in a 1D format
- Highly associated with a backend, each backend must provide a Storage implementation
- Storage does not exist at compile time, as no memory can be allocated during compile time evaluation

### Tensor
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
    // To take advantage of comptime features, all tensor code should be in comptime
    const out = comptime blk: {
        const x1 = fn1();
        const x2 = fn2(x1);
        break :blk x2;
    };

    // Use comptime on the graph call to see the compute graph
    // comptime out.graph();

    // Print the tensors created during compile time, which now exist at runtime
    // as they have memory addresses
    out.graph();

    // Initialize the backend which will allow for allocation of tensor storage
    TestBackend.init(.{});
    defer TestBackend.deinit();

    // Print the storage to show the data
    const eval_out = out.eval();
    std.debug.print("\n{any}\n", .{eval_out.storage});
```

## Roadmap

### First Milestone

- [x] Compile time verification of shapes and strides
    - [x] Contiguousness of strides is checked
    - [x] Check if a broadcast between two arrays of different shapes are valid
    - [x] Reducing a tensor along a dim yields a new tensor type where the reduce dim is now 1
    - [ ] Operations for reshaping, permuting, flattening, etc. a tensor
- [x] Building the compute graph at compile time
    - [x] When a function is called, the output tensor receives a closure to evaluate itself
    - [x] The recursive traversal (calling of input evaluate functions) can happen at compile time
    - [ ] Automated fusion of operators to reduce memory bandwidth requirements
- [x] Using the compute graph to perform computations on tensors
    - [x] Implementations of ops for the Zig backend
    - [ ] Implement matrix multiplication (GEMM)
    - [ ] Use SIMD via @Vector and multithreading as needed to accelerate the Zig backend
- [ ] Backpropagation and calculatation of gradients
- [ ] Model definition, traning, and inference and purely using this library
    - [ ] Implementations of gradient descent based optimizers
    - [ ] Convolutional neural network for the MNIST dataset

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
