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
            const ReturnType: type = Impl.ReturnType;
            pub fn forward(comptime self: T, comptime in: anytype) @TypeOf(comptime Impl.forward(self, in)) {
                return Impl.forward(self, in);
            }
        };
    }
};

pub const ReLU = struct {
    const Self = @This();
    pub usingnamespace Module.IFace(Self, struct {
        pub fn forward(comptime _: Self, comptime x: anytype) @TypeOf(x) {
            std.debug.assert(tensor.isTensor(@TypeOf(x)));
            return x.relu();
        }
    });
};

pub fn Linear(comptime in: u64, comptime out: u64, comptime dtype: dtypes.DType) type {
    return struct {
        const Self = @This();

        pub usingnamespace Module.IFace(Self, struct {
            pub fn forward(comptime self: Self, comptime x: anytype) @TypeOf(x).MatMul(Weight) {
                std.debug.assert(tensor.isTensor(@TypeOf(x)));
                return x.startGroup(std.fmt.comptimePrint("Linear_{d}_{d}", .{ in, out }))
                    .matmul(self.weight)
                    .add(self.bias)
                    .endGroup();
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
            fn ReturnType(comptime in: anytype) type {
                var result: AnyTensor = in.widen();
                for (modules) |module| {
                    std.debug.assert(Module.is(@TypeOf(module)));
                    result = module.forward(result.narrow().*).widen();
                }
                return result.Narrow();
            }
            pub fn forward(comptime _: Self, comptime in: anytype) ReturnType(in) {
                var result: AnyTensor = in.widen();
                for (modules) |module| {
                    std.debug.assert(Module.is(@TypeOf(module)));
                    result = module.forward(result.narrow().*).widen();
                }
                return result.narrow().*;
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
