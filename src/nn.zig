const std = @import("std");
const tensor = @import("tensor.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;
const dtypes = @import("dtypes.zig");

pub const Module = struct {
    fn is(comptime T: type) bool {
        return @hasDecl(T, "forward") and T.IType == Module;
    }

    pub fn IFace(comptime T: type, comptime Impl: type) type {
        return struct {
            pub const IType = Module;
            /// A limitation of Zig is that the return type of a function type cannot be generic on the inputs to the function
            /// So the implementation the dev provides should actually return an AnyTensor, but this will be narrowed back
            /// to the actual tensor type by the real forward function that is called
            /// TODO: Support returning a tuple of AnyTensor
            const ReturnType: type = Impl.ReturnType;
            const forwardImpl: fn (comptime T, comptime anytype) ReturnType = Impl.forwardImpl;
            pub fn forward(comptime self: T, comptime in: anytype) @TypeOf(comptime @This().forwardImpl(self, in).narrow()) {
                return forwardImpl(self, in).narrow();
            }
        };
    }
};

// TODO: Make functions.zig no longer a generic type and make a Functional module generator
// that basically does this same thing but for all functions in functions.zig;
pub const ReLU = struct {
    const Self = @This();
    pub usingnamespace Module.IFace(Self, struct {
        pub const ReturnType = AnyTensor;
        pub fn forwardImpl(comptime _: Self, comptime x: anytype) AnyTensor {
            std.debug.assert(tensor.isTensor(@TypeOf(x)));
            return x.relu().widen();
        }
    });
};

pub fn Linear(comptime in: u64, comptime out: u64, comptime dtype: dtypes.DType) type {
    return struct {
        const Self = @This();

        pub usingnamespace Module.IFace(Self, struct {
            pub const ReturnType = AnyTensor;
            pub fn forwardImpl(comptime self: Self, comptime x: anytype) AnyTensor {
                std.debug.assert(tensor.isTensor(@TypeOf(x)));
                return x.startGroup(std.fmt.comptimePrint("Linear_{d}_{d}", .{ in, out }))
                    .matmul(self.weight)
                    .add(self.bias)
                    .endGroup()
                    .widen();
            }
        });

        const Weight = tensor.tensor(dtype, .{ in, out });
        const Bias = tensor.tensor(dtype, .{out});

        weight: Weight = Weight.param(),
        bias: Bias = Bias.param(),
    };
}

pub fn Sequential(comptime modules: anytype) type {
    return struct {
        const Self = @This();
        pub usingnamespace Module.IFace(Self, struct {
            pub const ReturnType = AnyTensor;
            pub fn forwardImpl(comptime _: Self, comptime x: anytype) AnyTensor {
                std.debug.assert(tensor.isTensor(@TypeOf(x)));
                var result: AnyTensor = x.widen();
                for (modules) |module| {
                    std.debug.assert(Module.is(@TypeOf(module)));
                    result = module.forward(result.narrow().*).widen();
                }
                return result;
            }
        });
    };
}

test "linear" {
    const x = comptime tensor.tensor(.f32, .{ 16, 784 }).input();
    const linear = comptime Linear(784, 256, .f32){};
    const y = comptime linear.forwardImpl(x);
    const writer = std.io.Writer(std.fs.File, std.fs.File.WriteError, std.fs.File.write){ .context = std.io.getStdOut() };
    try @import("utils.zig").dataflowViz(&[_]*const AnyTensor{&y.widen()}, writer, std.testing.allocator);
}
