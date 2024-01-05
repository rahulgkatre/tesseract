# deepzig

The goal of this project is to write a deep learning library in Zig. Name is WIP. 

### Compile time verification of all tensor operations
- If the code compiles, there will be no issues with shaping and broadcasting.
- Zig compiler will generate simplified functions computing index, size, etc. for each tensor shape and stride in the program.

### Tensors are immutable and lazy 
- Any operation on a tensor will return a new Tensor object (immutable) but no data will be modified yet (laziness)
- Employ kernel fusing techniques to optimize operations, so that extra intermediate tensor storage does not need to be allocated
- A function can be called on the output tensor to evaluate the compute graph

###  Direct compilation or codegen to run on accelerators
- Direct compilation depends heavily on the Zig language's support. 
- Codegen will emit C code that will be compiled by an external compiler
- Calls compiler a C API to compile and run on the device.

### Zero dependencies outside of accelerator libraries
- If Zig can run on a machine, it can run deepzig

## Current state / checklist

### Done

- Compile time verification of shapes and strides
    - Permuting a tensor yields a tensor with the same underlying storage, but a different Tensor type (different shape and strides)
    - Contiguousness of strides is checked to prevent calls of .view() as it is not valid for non-contiguous tensors
    - Check if a broadcast between two arrays of different shapes are valid, used for zip/binary functions
    - Reducing a tensor along a dim yields a new Tensor type where the reduce dim is now 1

- Building the compute graph at compile time
    - When a function is called, the output tensor receives a closure to evaluate itself
    - The closure contains pointers to the inputs of the function, and a call to the evaluate function of the inputs
    - The evaluate function will eventually read the underlying data and operate on it to calculate the new data
    - The recursive traversal (calling of input evaluate functions) can happen at compile time

### In progress
- Using the compute graph to perform operations on some data
- Automated kernel fusion
 

### Not started
- Implementing forward and backward functions for operations supported by StableHLO and ONNX
- Running ONNX models (YOLO and Llama2 for now)
- Generating StableHLO programs as a compilation output
- Generating CUDA kernel code, compiling it, and running it on a CUDA device

## Rambling

### Why Zig?

I was playing around with a few languages (Nim, Cython) to determine the best language to write this library in. 

- **Compile time code execution via comptime** 
    - Enables simplification, verification, and optimization.
- **SIMD via @Vector**
    - Simple API for SIMD that is built into the langugage. 
- **C interop via @cImport** 
    - Zig can natively import C/C++ files and compile with them. 
    - This is useful for importing headers for GPUs and calling functions to compile and run emitted code.

### Other notes

- Minibatch training has a fixed batch size.
    - Data can be loaded as a slice, from which arrays of the batch size can be extracted and passed to the model. 
    - Slightly related, I have no idea if deepzig will work for variable length inputs such as for chat models. In my work with NLP we had set a max tokens parameter and used padding and masking to keep tensor shapes the same. 
- Ideally, there should be no heap allocation. 
    - The entire compute graph can be kept in CPU memory while the actual data buffers will be kept on device memory (GPU, etc.). 
    - The compute graph is static and buffers for batches of training data can be pre allocated. 
    - This will make deepzig usable in environments where heap memory is not available (e.g. embedded).
- Python interface
    - Support the buffer protocol in order to load in data from Numpy arrays
    - Mainly for OpenCV and Pandas support, since those are easier to use from Python than C
    - To create models using Zig tensors, codegen + JIT might need to be used to compile a Zig file to take advantage of comptime
- Alternative names
    - tenzor
    - tensorzig