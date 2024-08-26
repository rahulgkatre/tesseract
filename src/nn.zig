const std = @import("std");
const AnyTensor = tensor.AnyTensor;
const dtypes = @import("dtypes.zig");

const F = @import("tensor/functions.zig");
const tensor = @import("tensor/tensor.zig");
const types = @import("tensor/types.zig");

const asTensor = types.asTensor;
const TensorTypeOf = types.TensorTypeOf;

pub const Module = struct {
    fn is(comptime T: type) bool {
        return @hasDecl(T, "forward") and @hasDecl(T, "IType") and T.IType == Module;
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
        pub fn forward(_: Self, comptime x: anytype) TensorTypeOf(x) {
            return asTensor(x).relu();
        }
    });
};

pub fn LazyLinear(out: u64, dtype: dtypes.DType, comptime label: []const u8) type {
    return struct {
        const Self = @This();
        pub usingnamespace LazyModule.IFace(Self, struct {
            pub fn RealModule(comptime in: anytype) type {
                return Linear(asTensor(in).dimSize(-1), out, dtype, label);
            }
        });
    };
}

test LazyLinear {
    comptime {
        const x1 = tensor.Tensor([16][784]f32).input("x1");
        const linear = LazyLinear(256, .f32, "lazy_fc"){};
        _ = linear.forward(x1);
        const x2 = tensor.Tensor([16][1]f32).input("x2");
        _ = linear.forward(x2);
        const x3 = tensor.Tensor([16][1]f32).input("x3");
        _ = linear.forward(x3);
    }
}

pub fn Linear(in: u64, out: u64, dtype: dtypes.DType, label: []const u8) type {
    return struct {
        const Self = @This();

        pub usingnamespace Module.IFace(Self, struct {
            pub fn forward(self: Self, input: anytype) F.MatMul(input, Weight.empty()) {
                return asTensor(input).linear(self.weight, self.bias);
            }
        });

        const Weight = tensor.Tensor([in][out]dtypes.ZigType(dtype));
        const Bias = tensor.Tensor([out]dtypes.ZigType(dtype));

        weight: Weight = Weight.param(label ++ "_weight"),
        bias: Bias = Bias.param(label ++ "_bias"),
    };
}

pub fn Sequential(comptime label: []const u8, comptime modules: anytype) type {
    return struct {
        const Self = @This();
        const name = std.fmt.comptimePrint("Sequential_{s}", .{label});

        pub usingnamespace Module.IFace(Self, struct {
            fn ReturnType(in: anytype) type {
                var result: *const AnyTensor = asTensor(in).toAnyTensor();
                for (modules) |module| {
                    std.debug.assert(Module.is(@TypeOf(module)));
                    result = asTensor(module.forward(result.toTensor())).toAnyTensor();
                }
                return TensorTypeOf(result);
            }
            pub fn forward(_: Self, in: anytype) ReturnType(in) {
                var result: *const AnyTensor = asTensor(in).toAnyTensor();
                for (modules) |module| {
                    std.debug.assert(Module.is(@TypeOf(module)));
                    result = asTensor(module.forward(result.toTensor())).toAnyTensor();
                }
                return asTensor(result);
            }
        });
    };
}

test Linear {
    comptime {
        const x = tensor.Tensor([16][784]f32).input("input");
        const linear = Linear(784, 256, .f32, "fc"){};
        _ = linear.forward(x);
    }
}
