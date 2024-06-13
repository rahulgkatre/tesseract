const std = @import("std");
const tensor = @import("tensor.zig");
const dtypes = @import("dtypes.zig");
const ops = @import("ops.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;

const asTensor = tensor.asTensor;
const TensorTypeOf = tensor.TensorTypeOf;
const TensorTuple = tensor.TensorTuple;
const IntTensor = dtypes.IntTensor;
const BoolTensor = dtypes.BoolTensor;
const FloatTensor = dtypes.FloatTensor;

const F = @import("functions.zig");

pub fn NoBackwardReturnType(grad: anytype) type {
    _ = grad;
    return void;
}

pub fn noBackwardFn(grad: anytype) NoBackwardReturnType(grad) {
    @compileLog("Grad not implemented");
    return;
}

pub fn UnaryBackwardReturnType(a: anytype, grad: anytype) type {
    return TensorTypeOf(a).BinaryFnResultType(grad, .mul);
}

pub fn unaryBackward(comptime op: ops.UnaryOp, a: anytype, grad: anytype) UnaryBackwardReturnType(a, grad) {
    return switch (op) {
        .exp2 => F.exp2(a).mul(F.LN_2.mul(grad)),
        .log2 => F.div(F.INV_LN_2.mul(grad), a),
        .neg => F.neg(a).mul(grad),
        .recip => F.div(grad, F.mul(a, a).neg()),
        .sqrt => F.mul(0.5, grad).div(F.sqrt(a)),
        else => @compileError("not implemented"),
    };
}

pub fn BinaryBackwardReturnType(comptime op: ops.BinaryOp, a: anytype, b: anytype, grad: anytype) type {
    return switch (op) {
        .add => TensorTuple(.{ grad, grad }),
        .mul => TensorTuple(.{ F.mul(b, grad), F.mul(a, grad) }),
        else => @compileError("not implemented"),
    };
}

pub fn binaryBackward(comptime op: ops.BinaryOp, a: anytype, b: anytype, grad: anytype) BinaryBackwardReturnType(op, a, b, grad) {
    return switch (op) {
        .add => .{ grad, grad },
        .mul => .{ F.mul(b, grad), F.mul(a, grad) },
        else => @compileError("not implemented"),
    };
}

test binaryBackward {
    comptime {
        const a = tensor.Tensor([2][1][4]f32).empty();
        const b = tensor.Tensor([2][3][1]f32).empty();
        const c = F.mul(a, b);
        const getGradReturnType: *const fn (anytype) type = @ptrCast(c.meta.get_grad_fn_return_type);
        const gradFn: *const fn (anytype) getGradReturnType(c) = @ptrCast(c.meta.grad_fn);
        @compileLog(gradFn(c));
    }
}
