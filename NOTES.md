# Notes

### Python Interface

To take advantage of the comptime logic in this library, two interfaces are provided. The first interface is primarily for passing data into and out of the model defined using this library. It will implement the Python buffer protocol and use it to move data from Python sources (e.g. NumPy, TensorFlow, PyTorch) into buffers (no matter the device).

The second interface is for defining models. The functions available to use can be generated from the functions in the Zig library, but under the hood the Python interface is actually generating a Zig file, which will be compiled at the end and can be interacted with from Python via a C interface. This way the compile time tensor shape checking can actually take place inside a Python script. 