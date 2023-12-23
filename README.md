# deepzig

The goal of this project is to write a deep learning library in Zig. Name is WIP. Here are some features I want, decisions I've made, and goals for what this library should accomplish.  

### Compile time verification of all tensor operations
- If the code compiles, there will be no issues with shaping and broadcasting.
- All bounds checking is uneccessary and every loop can be unrolled.
- Zig compiler will generate simplified functions computing index, size, etc. for each tensor shape and stride in the program.


### Tensors are immutable and lazy in order to optimize 
- Any operation on a tensor will return a new Tensor object (immutable) but no data will be modified yet (laziness)
- Employ kernel fusing techniques to optimize operations, so that extra intermediate tensor storage does not need to be allocated

###  Direct compilation or codegen to run on accelerators
- Direct compilation depends heavily on the Zig language's support. 
- Codegen will emit C code that will be compiled by an external compiler
- Calls compiler a C API to compile and run on the device.

### Zero dependencies outside of accelerator libraries
- If Zig can run on a machine, it can run deepzig


Ultimately, I want deepzig to be a high performance library for training and running deep neural networks, with an API similar to PyTorch, requiring as little Zig knowledge as possible. I also want the codebase to be as small as possible, 

If the code compiles, your model will not have any issues that would interrupt the training or inference process once started. 

## Current state / checklist

### Done

- Compile time verification of shapes and strides
    - Permuting a tensor yields a tensor with the same underlying storage, but a different Tensor type (different shape and strides)
    - Contiguousness of strides is checked to prevent calls of .view() as it is not valid for non-contiguous tensors
    - Check if a broadcast between two arrays of different shapes are valid, used for zip/binary functions
    - Reducing a tensor along a dim yields a new Tensor type where the reduce dim is now 1

### In progress
- Building the compute graph at compile time
    - Assigning each tensor a unique ID 
    - Adding last function, and last function input to the history of a tensor

### Not started
- Implementing forward and backward functions for every operation supported by StableHLO
- Optimizing the compute graph by combining intermediate nodes into "metanodes" that perform multiple operations
- Generating StableHLO programs as a compilation output
- Generating CUDA kernel code, compiling it, and running it on a CUDA device


## Rambling

### Why Zig?

I was playing around with a few languages (Nim, Cython) to determine the best language to write this library in. I knew that 

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
- Uuse StableHLO as the opset that deepzig will implement in order to run popular pretrained models. 
    - In StableHLO, tensor dimensions are part of the type (as seen here https://scottamain.github.io/stablehlo/spec/). 
    - By implmenting the same typing in Zig, deepzig can both run and generate StableHLO programs, allowing users to take advantage of propietary DL compilers that will probably outperform deepzig. 
    - StableHLO is already compatible with ONNX, and has industry support.
- Some of the internal type guarding needs to be improved. 
    - There is a lot of anytype, mainly used by comptime functions to check tensor shapes and receive tensors of any shape
    - I wish Zig had a way to narrow types without having to use type reflection every time. I'm a big fan of TypeScript's generics and being able to pass in a type as wildcard (indicating that any value of that type is accepted) would be a really good feature to have in Zig. 
- Comptime is being abused
    - This could lead to long compile times, especially for more complex models
    - It depends a lot on the efficiency of the compiler. If I make a TransformerDecoder module type, and instantiate it multiple times, technically the Zig compiler should only generate 1 actual type (assuming it is instantiated with the same args every time)
    - Zig compiler is also changing, some comptime tricks may not work in the future (example: https://stackoverflow.com/questions/68555025/global-comptime-var-in-zig) so the amount of comptime tricks needs to be as few as possible. Currently using blocks and type reflection. 


