const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

const utils = @import("utils.zig");
const dtypes = @import("dtypes.zig");
const IntTensor = dtypes.IntTensor;
const BoolTensor = dtypes.BoolTensor;
const FloatTensor = dtypes.FloatTensor;

const typing = @import("typing.zig");
const TensorType = typing.TensorType;
const TensorTypeOf = typing.TensorTypeOf;
const asTensor = typing.asTensor;
const isTensor = typing.isTensorType;

const LN_2 = asTensor(0.69314718056);
const INV_LN_2 = asTensor(1.4426950408888495760773985077695);

// =============================================================================
// Tensor creating functions that define tensor type based on input
// =============================================================================

/// Create a 1D tensor filled with a sequence from {start} to {stop} exclusive
pub fn range(
    comptime start: comptime_int,
    comptime stop: comptime_int,
) Tensor([stop - start]comptime_int) {
    return Tensor([stop - start]comptime_int).range(start, stop);
}

/// Create a tensor with the same shape as the input but filled with random values
pub fn randLike(comptime input: anytype) TensorTypeOf(input) {
    return TensorTypeOf(input).rand();
}

/// Create a tensor with the same shape as the input but filled with a constant
pub fn fullLike(comptime input: anytype, value: dtypes.ZigType(input.dtype)) TensorTypeOf(input) {
    return TensorTypeOf(input).full(value);
}

// =============================================================================
// Unary functions
// =============================================================================

pub fn exp2(input: anytype) FloatTensor(TensorTypeOf(input)) {
    return asTensor(input).unaryFn(.Exp2);
}
pub fn log2(input: anytype) FloatTensor(TensorTypeOf(input)) {
    return asTensor(input).unaryFn(.Log2);
}
pub fn neg(input: anytype) TensorTypeOf(input) {
    return asTensor(input).unaryFn(.Neg);
}
pub fn recip(input: anytype) FloatTensor(TensorTypeOf(input)) {
    return asTensor(input).unaryFn(.Rcp);
}
pub fn sin(input: anytype) FloatTensor(TensorTypeOf(input)) {
    return asTensor(input).unaryFn(.Sin);
}
pub fn sqrt(input: anytype) FloatTensor(TensorTypeOf(input)) {
    return asTensor(input).unaryFn(.Sqrt);
}

test "unary" {
    const tensor1 = comptime Tensor([2][3][4]i32).full(3);
    const tensor2 = comptime tensor1.neg();
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 4 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expect(tensor2.meta.op_tracker.UnaryOp.op == .Neg);
    try std.testing.expectEqual(tensor2.meta.op_tracker.UnaryOp.in[0].toTensor().*, tensor1);
}

// =============================================================================
// Binary functions
// =============================================================================

pub fn add(input: anytype, other: anytype) TensorTypeOf(input).BinaryFnResultType(other, .Add) {
    return asTensor(input).binaryFn(other, .Add);
}
pub fn mul(input: anytype, other: anytype) TensorTypeOf(input).BinaryFnResultType(other, .Mul) {
    return asTensor(input).binaryFn(other, .Mul);
}
pub fn maximum(input: anytype, other: anytype) TensorTypeOf(input).BinaryFnResultType(other, .Max) {
    return asTensor(input).binaryFn(other, .Max);
}
pub fn mod(input: anytype, other: anytype) TensorTypeOf(input).BinaryFnResultType(other, .Mod) {
    return asTensor(input).binaryFn(other, .Mod);
}
pub fn lessThan(input: anytype, other: anytype) TensorTypeOf(input).BinaryFnResultType(other, .Lt) {
    return asTensor(input).binaryFn(other, .Lt);
}
pub fn equals(input: anytype, other: anytype) TensorTypeOf(input).BinaryFnResultType(other, .Eq) {
    return asTensor(input).binaryFn(other, .Eq);
}
pub fn xor(input: anytype, other: anytype) TensorTypeOf(input).BinaryFnResultType(other, .Xor) {
    return asTensor(input).binaryFn(other, .Xor);
}

test "binary" {
    const tensor1 = comptime Tensor([2][1][4]i32).full(2);
    const tensor2 = comptime Tensor([3][1]i32).full(3);
    const tensor3 = comptime tensor1.add(tensor2);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 4 }, tensor3.shape[0..tensor3.ndims]);
    try std.testing.expect(tensor3.meta.op_tracker.BinaryOp.op == .Add);
    try std.testing.expectEqualDeep(tensor3.meta.op_tracker.BinaryOp.in[0].meta.op_tracker.TypeOp.in[0].toTensor().*, tensor1);
    try std.testing.expectEqualDeep(tensor3.meta.op_tracker.BinaryOp.in[1].meta.op_tracker.TypeOp.in[0].toTensor().*, tensor2);
}

// =============================================================================
// Reduce functions
// =============================================================================

pub fn sum(input: anytype, comptime dims: anytype) TensorTypeOf(input).ReduceFnResultType(dims) {
    return asTensor(input).reduceFn(.Add, dims);
}
pub fn max(input: anytype, comptime dims: anytype) TensorTypeOf(input).ReduceFnResultType(dims) {
    return asTensor(input).reduceFn(.Max, dims);
}

test "reduce" {
    const tensor1 = comptime Tensor([2][3][4]i32).full(5);
    const tensor2 = comptime tensor1.sum(1);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 1, 4 }, tensor2.shape[0..tensor1.ndims]);
    try std.testing.expect(tensor2.meta.op_tracker.ReduceOp.op == .Add);
    try std.testing.expectEqual(tensor2.meta.op_tracker.ReduceOp.in[0].toTensor().*, tensor1);
    try std.testing.expectEqual(tensor2.meta.op_tracker.ReduceOp.args.mask[0..tensor2.ndims].*, ([_]bool{ false, true, false }));
}

test "multiple dim reduce" {
    const tensor1 = comptime Tensor([2][3][4]i32).full(5);
    const tensor2 = comptime tensor1.sum(.{ 0, 1 });
    try std.testing.expectEqualSlices(u64, &[_]u64{ 1, 1, 4 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expect(tensor2.meta.op_tracker.ReduceOp.op == .Add);
    try std.testing.expectEqual(tensor2.meta.op_tracker.ReduceOp.in[0].toTensor().*, tensor1);
    try std.testing.expectEqualDeep(tensor2.meta.op_tracker.ReduceOp.args.mask[0..tensor2.ndims], &[_]bool{ true, true, false });
}

// =============================================================================
// Expand
// =============================================================================

pub fn Expand(input: anytype, comptime new_shape: anytype) type {
    const A = TensorTypeOf(input);
    return TensorType(A.dtype, utils.broadcastShape(A.shape, new_shape));
}
/// Expand a tensor along 1 or more dimensions with size 1 and stride 0
/// The new shape must broadcast with the old shape
pub fn expand(input: anytype, comptime new_shape: anytype) Expand(input, new_shape) {
    const A = TensorTypeOf(input);
    const a = asTensor(input);
    const Out = Expand(input, new_shape);
    if (A == Out) {
        return a;
    }
    const bc_strides: [new_shape.len]u64 = blk: {
        var bc_strides: [new_shape.len]u64 = undefined;
        for (0..new_shape.len) |i| {
            bc_strides[new_shape.len - i - 1] = if (i >= a.ndims) 0 else a.strides[a.ndims - i - 1];
        }
        break :blk bc_strides;
    };

    return a.view(
        new_shape,
        bc_strides,
        a.offset,
    );
}

// =============================================================================
// Flatten
// =============================================================================

pub fn Flatten(input: anytype, comptime dims: utils.DimRange) type {
    const A = TensorTypeOf(input);
    const from = utils.signedToUnsignedDim(A.ndims, dims.from);
    const to = utils.signedToUnsignedDim(A.ndims, dims.to);
    if (from == to) {
        return @This();
    }
    var new_shape: [A.ndims - (to - from)]u64 = undefined;
    new_shape[from] = 1;
    for (0..A.ndims) |d| {
        if (d < from or d > to) {
            new_shape[d] = A.shape[d];
        } else {
            new_shape[from] *= A.shape[d];
        }
    }
    return Reshape(input, new_shape);
}
/// Flatten a range of dims, collapsing them to 1 dimension
pub fn flatten(input: anytype, comptime dim_range: utils.DimRange) Flatten(input, dim_range) {
    return asTensor(input).reshape(Flatten(input, dim_range).shape);
}

// =============================================================================
//
// =============================================================================
/// Get a mask of where padding values exist in the tensor
/// This could be useful for packed padded sequences for NLP applications
pub fn paddingMask(input: anytype, comptime padding: anytype) TensorType(.bool, TensorTypeOf(input).shape) {
    const a = asTensor(input);
    const not_padding = TensorType(.bool, a.shape).full(true);
    return not_padding.pad(padding, .{ .Constant = .{ .value = false } });
}

// =============================================================================
// Permute
// =============================================================================

pub fn Permute(comptime input: anytype, comptime perm: [TensorTypeOf(input).ndims]u8) type {
    const A = TensorTypeOf(input);
    return Reshape(input, utils.arrayPermute(u64, A.ndims, A.shape, perm));
}
/// Permute the dimensions of the  A valid permutation must contain
/// values from 0 to ndims and each value must appear exactly once.
pub fn permute(comptime input: anytype, comptime perm: [TensorTypeOf(input).ndims]u8) Permute(input, perm) {
    const A = TensorTypeOf(input);
    const a = asTensor(input);
    return a.view(
        Permute(input, perm).shape,
        utils.arrayPermute(u64, A.ndims, a.strides[0..A.ndims].*, perm),
        a.offset,
    );
}
test "permute" {
    const tensor1 = comptime Tensor([2][3][4]f32).full(0);
    const tensor2 = comptime tensor1.permute(.{ 0, 2, 1 });
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 4, 3 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 12, 1, 4 }, tensor2.strides[0..tensor2.ndims]);
}

// =============================================================================
// Reshape
// =============================================================================

pub fn Reshape(comptime input: anytype, comptime new_shape: anytype) type {
    const OldType = TensorTypeOf(input);
    const NewType = TensorType(OldType.dtype, new_shape);
    std.debug.assert(OldType.num_entries == NewType.num_entries);
    return NewType;
}
/// Change the shape of the  This changes the type too.
pub fn reshape(comptime input: anytype, comptime new_shape: anytype) Reshape(input, new_shape) {
    const a = asTensor(input);
    return a.contiguous().view(new_shape, Reshape(a, new_shape).contiguous_strides, a.offset);
}
test "reshape" {
    const tensor1 = comptime Tensor([2][3][4]i32).full(0);
    const tensor2 = comptime tensor1.reshape(.{ 12, 2 });
    const tensor3 = comptime tensor2.reshape(.{24});
    try std.testing.expectEqualSlices(u64, &[_]u64{ 12, 2 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 1 }, tensor2.strides[0..tensor2.ndims]);
    try std.testing.expectEqualSlices(u64, &[_]u64{24}, tensor3.shape[0..tensor3.ndims]);
    try std.testing.expectEqualSlices(u64, &[_]u64{1}, tensor3.strides[0..tensor3.ndims]);
}

// =============================================================================
// Squeeze
// =============================================================================

pub fn Squeeze(comptime input: anytype, comptime dim: i16) type {
    const A = TensorTypeOf(input);
    if (A.shape[utils.signedToUnsignedDim(A.ndims, dim)] != 1) {
        @compileError("Cannot squeeze as dimension size is not 1");
    }
    return Reshape(input, utils.arrayDelete(A.ndims, A.shape, utils.signedToUnsignedDim(A.ndims, dim)));
}
/// Remove a dim of size 1 from the shape of the
pub fn squeeze(comptime input: anytype, comptime dim: i16) Squeeze(input, dim) {
    const A = TensorTypeOf(input);
    const a = asTensor(input);
    return a.view(
        Squeeze(input, dim).shape,
        utils.arrayDelete(A.ndims, a.strides[0..A.ndims].*, utils.signedToUnsignedDim(A.ndims, dim)),
        a.offset,
    );
}

// =============================================================================
// Transpose
// =============================================================================

pub fn Transpose(comptime input: anytype, comptime dim1: i16, comptime dim2: i16) type {
    const Type = TensorTypeOf(input);
    const norm1 = utils.signedToUnsignedDim(Type.ndims, dim1);
    const norm2 = utils.signedToUnsignedDim(Type.ndims, dim2);
    var new_shape = Type.shape;
    new_shape[norm1] = Type.shape[norm2];
    new_shape[norm2] = Type.shape[norm1];
    return Reshape(input, new_shape);
}
/// Transpose two dimensions of the  Similar to permute, but only for two dimensions.
pub fn transpose(comptime input: anytype, comptime dim1: i16, comptime dim2: i16) Transpose(input, dim1, dim2) {
    const A = TensorTypeOf(input);
    const a = asTensor(input);
    const norm1 = utils.signedToUnsignedDim(A.ndims, dim1);
    const norm2 = utils.signedToUnsignedDim(A.ndims, dim2);
    if (norm1 != norm2) {
        var new_strides = a.strides[0..a.ndims].*;
        new_strides[norm1] = a.strides[norm2];
        new_strides[norm2] = a.strides[norm1];
        return a.view(
            Transpose(input, norm1, norm2).shape,
            new_strides,
            a.offset,
        );
    } else {
        return a;
    }
}
/// Shorthand for transposing rightmost dimensions of tensor
pub fn T(comptime input: anytype) Transpose(input, -2, -1) {
    return asTensor(input).transpose(-2, -1);
}
test "transpose" {
    const tensor1 = comptime Tensor([2][1][4]i32).full(1);
    const tensor2 = comptime tensor1.T();
    const ndims = tensor1.ndims;
    try std.testing.expectEqualDeep(tensor2, comptime tensor1.transpose(-2, -1));
    try std.testing.expectEqualDeep(tensor1.shape[0..ndims], comptime tensor2.T().shape[0..ndims]);
    try std.testing.expectEqualDeep(tensor1.strides[0..ndims], comptime tensor2.T().strides[0..ndims]);
}

// =============================================================================
// Unsqueeze
// =============================================================================

pub fn Unsqueeze(comptime input: anytype, comptime dim: i16) type {
    const A = TensorTypeOf(input);
    return Reshape(input, utils.arrayInsert(A.ndims, A.shape, utils.signedToUnsignedDim(A.ndims, dim), 1));
}
/// Insert a dim of size 1 into the shape of the tensor
pub fn unsqueeze(comptime input: anytype, comptime dim: i16) Unsqueeze(input, dim) {
    const A = TensorTypeOf(input);
    const a = asTensor(input);
    return a.view(
        Unsqueeze(input, dim).shape,
        utils.arrayInsert(a.ndims, a.strides[0..a.ndims].*, utils.signedToUnsignedDim(A.ndims, dim), 0),
        a.offset,
    );
}

// =============================================================================
// Compound functions
// =============================================================================

pub fn div(input: anytype, other: anytype) TensorTypeOf(input).BinaryFnResultType(other, .Add) {
    return asTensor(input).startGroup("div").mul(asTensor(other).recip()).endGroup();
}
pub fn sub(input: anytype, other: anytype) TensorTypeOf(input).BinaryFnResultType(other, .Add) {
    return asTensor(input).add(asTensor(other).neg());
}
pub fn exp(input: anytype) FloatTensor(TensorTypeOf(input)) {
    return asTensor(input).startGroup("exp").mul(INV_LN_2).exp2().endGroup();
}
pub fn log(input: anytype) FloatTensor(TensorTypeOf(input)) {
    return asTensor(input).startGroup("log").log2().mul(LN_2).endGroup();
}
pub fn sigmoid(input: anytype) FloatTensor(TensorTypeOf(input)) {
    const x = asTensor(input).startGroup("sigmoid");
    const x_pos = x.neg().exp().add(1.0).recip();
    const x_neg = x.exp().div(x.exp().add(1.0));
    const mask = x.lessThan(0.0);
    return mask.where(x_neg, x_pos).endGroup();
}
pub fn relu(input: anytype) TensorTypeOf(input) {
    if (dtypes.isFloat(input.dtype)) {
        return asTensor(input).startGroup("relu").maximum(0.0).endGroup();
    } else if (dtypes.isInt(input.dtype)) {
        return asTensor(input).startGroup("relu").maximum(0).endGroup();
    } else {
        unreachable;
    }
}
pub fn softmax(input: anytype, comptime dim: i16) FloatTensor(TensorTypeOf(input)) {
    const x = asTensor(input).startGroup("softmax");
    const minus_max_exp = x.sub(x.max({})).exp();
    const sumexp = minus_max_exp.sum(dim);
    return minus_max_exp.div(sumexp).endGroup();
}
pub fn mean(input: anytype, comptime dims: anytype) FloatTensor(TensorTypeOf(input).ReduceFnResultType(dims)) {
    return input.div(input.sum(dims));
}
pub fn variance(input: anytype, comptime dims: anytype) FloatTensor(TensorTypeOf(input).ReduceFnResultType(dims)) {
    const x = asTensor(input).startGroup("variance");
    const mu = x.mean(dims);
    const N: f64 = @floatFromInt(@divExact(x.num_entries, mu.num_entries));
    const a_minus_mu = x.sub(mu);
    return (a_minus_mu.mul(a_minus_mu)).sum(dims).div(N);
}

pub fn MatMul(input: anytype, other: anytype) type {
    const A = TensorTypeOf(input);
    const B = TensorTypeOf(other);

    // Matrix multiplication invariant
    // (n x m1) matmul (m2 x p) -> (n x p) iff m1 = m2
    // otherwise matmul is invalid, compile error
    const n = if (A.ndims == 1) 1 else A.shape[A.ndims - 2];
    const m = A.shape[A.ndims - 1];
    const b_m = if (B.ndims == 1) 1 else B.shape[B.ndims - 2];
    const p = B.shape[B.ndims - 1];

    if (m != b_m) {
        @compileError(std.fmt.comptimePrint(
            \\Tensors do not satisfy the matrix multiplication invariant, m must equal other_m
            \\Tensor A: {a}
            \\Tensor B: {a}
            \\n = {d}, m = {d}, other_m = {d}, p = {d}
        , .{ A.shape, B.shape, n, m, b_m, p }));
    }

    const mm_ndims = @max(A.ndims, B.ndims);
    var mm_shape: [mm_ndims]u64 = undefined;
    // Expanding check, look only at batch dimensions (everything before last 2 dimensions)
    const mm_bc_shape: [mm_ndims - 2]u64 = utils.broadcastShape(A.shape[0 .. A.ndims - 2].*, B.shape[0 .. B.ndims - 2].*);
    @memcpy(mm_shape[0 .. mm_ndims - 2], &mm_bc_shape);
    mm_shape[mm_ndims - 2] = n;
    mm_shape[mm_ndims - 1] = p;
    return TensorType(A.dtype, mm_shape);
}
pub fn matmul(input: anytype, other: anytype) MatMul(input, other) {
    const A = TensorTypeOf(input);
    const B = TensorTypeOf(other);
    return asTensor(input).startGroup("matmul")
        .unsqueeze(A.ndims - 1)
        .mul(asTensor(other).transpose(B.ndims - 2, B.ndims - 1).unsqueeze(B.ndims - 2))
        .sum(A.ndims)
        .squeeze(A.ndims)
        .endGroup();
}

pub fn Window1d(input: anytype, window: u64) type {
    const I = TensorTypeOf(input);
    return TensorType(I.dtype, I.shape[0 .. I.ndims - 1] ++ .{ I.shape[I.ndimws - 1] - window + 1, window });
}
pub fn window1d(input: anytype, window: u64) Window1d(input, window) {
    const Result = Window1d(input, window);
    const a = asTensor(input);
    return a.view(Result.shape, a.strides.* ++ .{a.strides[a.ndims - 1]}, a.offset);
}

pub fn SlidingWindowView(input: anytype, window_shape: anytype, window_dims: anytype) type {
    const I = TensorTypeOf(input);
    _ = window_shape;
    _ = window_dims;
    _ = I;
    // var window = switch (@TypeOf(window_shape))
    // var dims = switch (@TypeOf(window_dims))

}

test "binary followed by reduce" {
    const tensor1 = comptime Tensor([2][1][4]i32).full(2);
    const tensor2 = comptime Tensor([2][3][1]i32).full(3);
    const tensor3 = comptime tensor1.add(tensor2).sum(1);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 1, 4 }, tensor3.shape[0..tensor3.ndims]);
    try std.testing.expect(tensor3.meta.op_tracker.ReduceOp.op == .Add);
    // Anonymous intermediate tensor that stores tensor1 + tensor2
    const anon = tensor3.meta.op_tracker.ReduceOp.in[0];
    try std.testing.expectEqualDeep(anon.meta.op_tracker.BinaryOp.in[0].meta.op_tracker.TypeOp.in[0].toTensor().*, tensor1);
    try std.testing.expectEqualDeep(anon.meta.op_tracker.BinaryOp.in[1].meta.op_tracker.TypeOp.in[0].toTensor().*, tensor2);
}

fn fn1() Tensor([2][1][4]i32) {
    const tensor1 = Tensor([2][1][4]i32).full(1);
    const tensor2 = Tensor([2][3][1]i32).full(2);
    const tensor3 = tensor1.add(tensor2).sum(1);
    return tensor3;
}

fn fn2(input: anytype) Tensor([2][3][4]i32) {
    const tensor4 = Tensor([2][1][4]i32).full(4);
    const tensor5 = Tensor([2][3][1]i32).full(5);
    const tensor6 = tensor4.mul(input).sum(1).add(tensor5);
    return tensor6;
}

test "functions" {
    _ = comptime blk: {
        const tensor3 = fn1();
        const tensor6 = fn2(tensor3);
        break :blk tensor6;
    };
}
