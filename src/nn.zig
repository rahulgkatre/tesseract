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
            pub fn forward(comptime self: T, comptime in: anytype) @TypeOf(comptime Impl.forward(self, in)) {
                return Impl.forward(self, in);
            }
        };
    }
};

pub const LazyModule = struct {
    pub fn IFace(comptime T: type, comptime Impl: type) type {
        return struct {
            pub usingnamespace Module.IFace(T, struct {
                pub fn forward(_: T, comptime in: anytype) @TypeOf(comptime Impl.RealModule(in).forward(Impl.RealModule(in){}, in)) {
                    return Impl.RealModule(in).forward(Impl.RealModule(in){}, in);
                }
            });
        };
    }
};

pub const ReLU = struct {
    const Self = @This();
    pub usingnamespace Module.IFace(Self, struct {
        pub fn forward(_: Self, comptime x: anytype) @TypeOf(x) {
            std.debug.assert(tensor.isTensor(@TypeOf(x)));
            return x.relu();
        }
    });
};

pub fn LazyLinear(out: u64, dtype: dtypes.DType, comptime label: []const u8) type {
    return struct {
        const Self = @This();
        pub usingnamespace LazyModule.IFace(Self, struct {
            pub fn RealModule(comptime in: anytype) type {
                return Linear(tensor.asTensor(in).dimSize(-1), out, dtype, label);
            }
        });
    };
}

test "lazy linear" {
    const x = comptime tensor.Tensor([16][784]f32).input(null);
    const linear = comptime LazyLinear(256, .f32, "lazy_fc"){};
    const y1 = comptime linear.forward(x);

    const x2 = comptime tensor.Tensor([16][1]f32).input("x2");
    const y2 = comptime linear.forward(x2);

    const x3 = comptime tensor.Tensor([16][1]f32).input("x3");
    const y3 = comptime linear.forward(x3);

    const writer = std.io.Writer(std.fs.File, std.fs.File.WriteError, std.fs.File.write){ .context = std.io.getStdOut() };
    try @import("utils.zig").dataflowViz(&[_]*const AnyTensor{ &y1.widen(), &y2.widen(), &y3.widen() }, writer, std.testing.allocator);
}

pub fn Linear(in: u64, out: u64, dtype: dtypes.DType, label: []const u8) type {
    return struct {
        const Self = @This();

        pub usingnamespace Module.IFace(Self, struct {
            pub fn forward(self: Self, x: anytype) tensor.AsTensor(x).MatMul(self.weight) {
                std.debug.assert(tensor.isTensor(@TypeOf(x)));
                return x.startGroup(std.fmt.comptimePrint("Linear_{d}_{d}_{s}", .{ in, out, label }))
                    .matmul(self.weight)
                    .add(self.bias)
                    .endGroup();
            }
        });

        const Weight = tensor.Tensor([in][out]dtypes.ZigType(dtype));
        const Bias = tensor.Tensor([out]dtypes.ZigType(dtype));

        weight: Weight = Weight.param(label ++ "_weight"),
        bias: Bias = Bias.param(label ++ "_bias"),
    };
}

pub fn Sequential(comptime modules: anytype) type {
    return struct {
        const Self = @This();
        pub usingnamespace Module.IFace(Self, struct {
            fn ReturnType(in: anytype) type {
                var result: AnyTensor = in.widen();
                for (modules) |module| {
                    std.debug.assert(Module.is(@TypeOf(module)));
                    result = module.forward(result.narrow().*).widen();
                }
                return result.Narrow();
            }
            pub fn forward(_: Self, in: anytype) ReturnType(in) {
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
    const x = comptime tensor.Tensor([16][784]f32).input(null);
    const linear = comptime Linear(784, 256, .f32, "fc"){};
    const y = comptime linear.forward(x);
    _ = y;
    // const writer = std.io.Writer(std.fs.File, std.fs.File.WriteError, std.fs.File.write){ .context = std.io.getStdOut() };
    // try @import("utils.zig").dataflowViz(&[_]*const AnyTensor{&y.widen()}, writer, std.testing.allocator);
}
