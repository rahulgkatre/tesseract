const std = @import("std");
const tensor = @import("tensor.zig");
const dtypes = @import("dtypes.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;

const isTensor = tensor.isTensor;
const IntTensor = dtypes.IntTensor;
const BoolTensor = dtypes.BoolTensor;
const FloatTensor = dtypes.FloatTensor;

const LN_2 = tensor.asTensor(0.69314718056);
const INV_LN_2 = LN_2.recip();

pub fn Gradient(comptime T: type) type {
    comptime std.debug.assert(isTensor(T));
    const Tensor = tensor.Tensor(T.dtype, T.ndims, T.shape);

    return struct {
        pub fn d_exp2(a: Tensor, grad: anytype) Tensor.BinaryFnResult(grad, .Mul) {
            return a.exp2().mul(LN_2).mul(grad);
        }

        pub fn d_log2(a: Tensor, grad: anytype) Tensor.BinaryFnResult(grad, .Mul) {
            return a.recip().mul(INV_LN_2).mul(grad);
        }

        pub fn d_neg(a: Tensor, grad: anytype) Tensor.BinaryFnResult(grad, .Mul) {
            return a.neg().mul(grad);
        }

        pub fn d_recip(a: Tensor, grad: anytype) Tensor.BinaryFnResult(grad, .Mul) {
            return a.mul(a).recip().neg().mul(grad);
        }

        pub fn d_sqrt(a: Tensor, grad: anytype) Tensor.BinaryFnResult(grad, .Mul) {
            return a.sqrt().recip().mul(0.5).mul(grad);
        }

        pub fn d_id(_: Tensor, grad: anytype) @TypeOf(grad) {
            return grad;
        }

        // pub fn d_where(a: Tensor, b: anytype, c: anytype, grad: anytype) Tensor.Whe
    };
}
