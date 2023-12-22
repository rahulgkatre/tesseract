# deepzig

The goal of this project is to write a deep learning library in Zig. Here are some features I want, decisions I've made, and goals for what this library should accomplish.  

- **Compile time verification of all tensor operations** 
    - Make shapes a generic parameter of a tensor. 
    - If the code compiles, there will be no issues with shaping and broadcasting.
    - All bounds checking is uneccessary and every loop can be unrolled.
    - Zig compiler will generate simplified functions computing index, size, etc. for each tensor shape in the program. 
- **Tensors are immutable** 
    - Any operation on a tensor will return a new Tensor object. 
    - However, the underlying buffer may be the shared between tensors. 
- **Compute graph optimization** 
    - Employ kernel fusing techniques to optimize operations. 
    - Ties in with laziness in operations.
- **Direct compilation or codegen to as many accelerators as possible** 
    - Direct compilation depends heavily on the Zig language's support. 
    - Codegen will emit C code that will be compiled by an external compiler
    - Calls compiler a C API to compile and run on the device.
- **Zero dependencies outside of accelerator specific libraries** 
    - If Zig can run on a machine, it can run deepzig. 

Why Zig?
- **Compile time code execution via comptime** 
    - Enables simplification, verification, and optimization.
- **SIMD via @Vector**
    - Simple API for SIMD that is built into the langugage. 
- **C interop via @cImport** 
    - Zig can natively import C/C++ files and compile with them. 
    - This is useful for importing headers for GPUs and calling functions to compile and run emitted code.

Other notes
- Minibatch training has a fixed batch size.
    - Data can be loaded as a slice, from which arrays of the batch size can be extracted and passed to the model. 
    - Slightly related, I have no idea if deepzig will work for variable length inputs such as for chat models. In my work with NLP we had set a max tokens parameter and used padding and masking to keep tensor shapes the same. 
- Ideally, there should be no heap allocation. 
    - The entire compute graph can be kept in CPU memory while the actual data buffers will be kept on device memory (GPU, etc.). 
    - The compute graph is static and buffers for batches of training data can be pre allocated. 
- Uuse StableHLO as the "ISA" that deepzig will implement in order to run popular pretrained models. 
    - In StableHLO, tensor dimensions are part of the type (as seen here https://scottamain.github.io/stablehlo/spec/). 
    - By implmenting the same typing in Zig, deepzig can both run and generate StableHLO programs, allowing users to take advantage of propietary DL compilers that will probably outperform deepzig. 
- Some of the internal type guarding needs to be improved. 
    - There is a lot of anytype, mainly used by comptime functions to check tensor shapes and receive tensors of any shape
    - I wish Zig had a way to narrow types without having to use type reflection every time. I'm a big fan of TypeScript's generics and being able to pass in a type as wildcard (indicating that any value of that type is accepted) would be a really good feature to have in Zig. 


