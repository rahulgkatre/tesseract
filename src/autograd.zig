const std = @import("std");
const tensor = @import("tensor.zig");
const dtypes = @import("dtypes.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;

const asTensor = tensor.asTensor;
const TensorTypeOf = tensor.TensorTypeOf;
const IntTensor = dtypes.IntTensor;
const BoolTensor = dtypes.BoolTensor;
const FloatTensor = dtypes.FloatTensor;

const F = @import("functions.zig");

pub fn dexp2(a: anytype, grad: anytype) TensorTypeOf(a).BinaryFnResultType(grad, .mul) {
    return asTensor(a).exp2().mul(F.LN_2.mul(grad));
}

pub fn dlog2(a: anytype, grad: anytype) TensorTypeOf(a).BinaryFnResultType(grad, .mul) {
    return asTensor(a).recip().mul(F.INV_LN_2.mul(grad));
}

pub fn dneg(a: anytype, grad: anytype) TensorTypeOf(a).BinaryFnResultType(grad, .mul) {
    return asTensor(a).neg().mul(grad);
}

pub fn drecip(a: anytype, grad: anytype) TensorTypeOf(a).BinaryFnResultType(grad, .mul) {
    return F.mul(a, a).recip().neg().mul(grad);
}

pub fn dsqrt(a: anytype, grad: anytype) TensorTypeOf(a).BinaryFnResultType(grad, .mul) {
    return asTensor(a).sqrt().recip().mul(0.5).mul(grad);
}
