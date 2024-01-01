// TODO: Make a backend struct
// Backends provide APIS for performing various computations and management of buffers in memory
// Backends are not limited to CPU backend, they can run on other devices (GPU, TPU, etc.)
// To make this work, use a DType enum to store the type of each tensor element
// Side benefit of this is that it allows for support of datatypes that are not built into Zig
// but are part of other backends (e.g. bfloat16)
// In the case of the Zig CPU backend the buffer will be an anyopaque slice
// Using the DType enum the anyopaque slice can be cast to the correct element type slice
