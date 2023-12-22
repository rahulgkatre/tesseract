# deepzig

The goal of this project is to write a deep learning library in Zig. Some features / goals: 

- Compile time verification of all tensor operationss. If the code compiles, there will be no issues with shaping and broadcasting.
- Compute graph optimization. Employ kernel fusing techniques to optimize operations. Ties in with laziness in operations.
- Direct compilation or codegen to as many accelerators as possible. Direct compilation depends heavily on the Zig language's Stage Table. Codegen will emit C code that will be compiled by an external compiler that uses a C API to compile and run on the device.

Why Zig?
- Compile time code execution (metaprogramming). This feature will enable verification and optimization of the compute graph before it is compiled.
- Portable SIMD. Usage of Zig's SIMD features for tensor computations on the CPU means that CPU operations can be written once and run fast anywhere.
- C interop. Zig can natively import C/C++ files and compile with them. This is useful for importing headers for GPUs and calling functions to compile and run emitted code.

Other notes
- Tensors are immutable. Any operation on a tensor will return a new Tensor object. However, the underlying buffer may be the same. 
- With tensor shapes being known at compile time, all bounds checking is uneccessary and every loop can be unrolled.
- Sinced minibatch/batch training is used and has a fixed batch size, data can be loaded as a slice, from which arrays of the batch size can be extracted and passed to the model. 
- For now, ONNX ops will be targeted but StableHLO can be added in the future.

