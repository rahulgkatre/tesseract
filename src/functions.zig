const std = @import("std");
const tensor = @import("tensor.zig");
const utils = @import("utils.zig");
const dtypes = @import("dtypes.zig");

const Tensor = tensor.Tensor;
const TensorType = tensor.TensorType;
const TensorTypeOf = tensor.TensorTypeOf;
const asTensor = tensor.asTensor;
const isTensor = tensor.isTensor;
const IntTensor = dtypes.IntTensor;
const BoolTensor = dtypes.BoolTensor;
const FloatTensor = dtypes.FloatTensor;

const LN_2 = tensor.asTensor(0.69314718056);
const INV_LN_2 = tensor.asTensor(1.4426950408888495760773985077695);

// =============================================================================
// Unary functions
// =============================================================================

pub fn exp2(a: anytype) FloatTensor(TensorTypeOf(a)) {
    return asTensor(a).unaryFn(.Exp2);
}
pub fn log2(a: anytype) FloatTensor(TensorTypeOf(a)) {
    return asTensor(a).unaryFn(.Log2);
}
pub fn neg(a: anytype) TensorTypeOf(a) {
    return asTensor(a).unaryFn(.Neg);
}
pub fn recip(a: anytype) FloatTensor(TensorTypeOf(a)) {
    return asTensor(a).unaryFn(.Rcp);
}
pub fn sin(a: anytype) FloatTensor(TensorTypeOf(a)) {
    return asTensor(a).unaryFn(.Sin);
}
pub fn sqrt(a: anytype) FloatTensor(TensorTypeOf(a)) {
    return asTensor(a).unaryFn(.Sqrt);
}

// =============================================================================
// Binary functions
// =============================================================================

pub fn add(a: anytype, b: anytype) TensorTypeOf(a).BinaryFnResultType(b, .Add) {
    return asTensor(a).binaryFn(b, .Add);
}
pub fn mul(a: anytype, b: anytype) TensorTypeOf(a).BinaryFnResultType(b, .Mul) {
    return asTensor(a).binaryFn(b, .Mul);
}
pub fn maximum(a: anytype, b: anytype) TensorTypeOf(a).BinaryFnResultType(b, .Max) {
    return asTensor(a).binaryFn(b, .Max);
}
pub fn mod(a: anytype, b: anytype) TensorTypeOf(a).BinaryFnResultType(b, .Mod) {
    return asTensor(a).binaryFn(b, .Mod);
}
pub fn lessThan(a: anytype, b: anytype) TensorTypeOf(a).BinaryFnResultType(b, .Lt) {
    return asTensor(a).binaryFn(b, .Lt);
}
pub fn equals(a: anytype, b: anytype) TensorTypeOf(a).BinaryFnResultType(b, .Eq) {
    return asTensor(a).binaryFn(b, .Eq);
}
pub fn xor(a: anytype, b: anytype) TensorTypeOf(a).BinaryFnResultType(b, .Xor) {
    return asTensor(a).binaryFn(b, .Xor);
}

// =============================================================================
// Reduce functions
// =============================================================================

pub fn sum(a: anytype, comptime dims: anytype) TensorTypeOf(a).ReduceFnResultType(dims) {
    return asTensor(a).reduceFn(.Add, dims);
}
pub fn max(a: anytype, comptime dims: anytype) TensorTypeOf(a).ReduceFnResultType(dims) {
    return asTensor(a).reduceFn(.Max, dims);
}

// =============================================================================
// Expand
// =============================================================================

pub fn Expand(any: anytype, comptime new_shape: anytype) type {
    const A = TensorTypeOf(any);
    return TensorType(A.dtype, utils.broadcastShape(A.shape, new_shape));
}
/// Expand a tensor along 1 or more dimensions with size 1 and stride 0
/// The new shape must broadcast with the old shape
pub fn expand(any: anytype, comptime new_shape: anytype) Expand(any, new_shape) {
    const A = TensorTypeOf(any);
    const a = asTensor(any);
    const Out = TensorType(a.dtype, utils.broadcastShape(A.shape, new_shape));
    if (TensorTypeOf(a) == Out) {
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

pub fn Flatten(any: anytype, comptime dim_range: utils.DimRange) type {
    const A = TensorTypeOf(any);
    const from = A.signedToUnsignedDim(dim_range.from);
    const to = A.signedToUnsignedDim(dim_range.to);
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
    return Reshape(any, new_shape);
}
/// Flatten a range of dims, collapsing them to 1 dimension
pub fn flatten(comptime any: anytype, comptime dim_range: utils.DimRange) Flatten(any, dim_range) {
    return asTensor(any).reshape(Flatten(any, dim_range).shape);
}

// =============================================================================
//
// =============================================================================
/// Get a mask of where padding values exist in the tensor
/// This could be useful for packed padded sequences for NLP applications
pub fn paddingMask(comptime any: anytype, comptime padding: anytype) TensorType(.bool, TensorTypeOf(any).shape) {
    const a = asTensor(any);
    const not_padding = TensorType(.bool, a.shape).full(true);
    return not_padding.pad(padding, .{ .Constant = .{ .value = false } });
}

// =============================================================================
// Permute
// =============================================================================

pub fn Permute(comptime any: anytype, comptime perm: [TensorTypeOf(any).ndims]u8) type {
    const A = TensorTypeOf(any);
    return Reshape(any, utils.arrayPermute(u64, A.ndims, A.shape, perm));
}
/// Permute the dimensions of the tensor. A valid permutation must contain
/// values from 0 to ndims and each value must appear exactly once.
pub fn permute(comptime any: anytype, comptime perm: [TensorTypeOf(any).ndims]u8) Permute(any, perm) {
    const A = TensorTypeOf(any);
    const a = asTensor(any);
    return a.view(
        Permute(any, perm).shape,
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

pub fn Reshape(comptime any: anytype, comptime new_shape: anytype) type {
    const OldType = TensorTypeOf(any);
    const NewType = TensorType(OldType.dtype, new_shape);
    std.debug.assert(OldType.num_entries == NewType.num_entries);
    return NewType;
}
/// Change the shape of the tensor. This changes the type too.
pub fn reshape(comptime any: anytype, comptime new_shape: anytype) Reshape(any, new_shape) {
    const a = asTensor(any);
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

pub fn Squeeze(comptime any: anytype, comptime dim: i16) type {
    const A = TensorTypeOf(any);
    if (A.shape[A.signedToUnsignedDim(dim)] != 1) {
        @compileError("Cannot squeeze as dimension size is not 1");
    }
    return Reshape(any, utils.arrayDelete(A.ndims, A.shape, A.signedToUnsignedDim(dim)));
}
/// Remove a dim of size 1 from the shape of the tensor.
pub fn squeeze(comptime any: anytype, comptime dim: i16) Squeeze(any, dim) {
    const A = TensorTypeOf(any);
    const a = asTensor(any);
    return a.view(
        Squeeze(any, dim).shape,
        utils.arrayDelete(A.ndims, a.strides[0..A.ndims].*, A.signedToUnsignedDim(dim)),
        a.offset,
    );
}

// =============================================================================
// Transpose
// =============================================================================

pub fn Transpose(comptime any: anytype, comptime dim1: i16, comptime dim2: i16) type {
    const Type = TensorTypeOf(any);
    const norm1 = Type.signedToUnsignedDim(dim1);
    const norm2 = Type.signedToUnsignedDim(dim2);
    var new_shape = Type.shape;
    new_shape[norm1] = Type.shape[norm2];
    new_shape[norm2] = Type.shape[norm1];
    return Reshape(any, new_shape);
}
/// Transpose two dimensions of the tensor. Similar to permute, but only for two dimensions.
pub fn transpose(comptime any: anytype, comptime dim1: i16, comptime dim2: i16) Transpose(any, dim1, dim2) {
    const A = TensorTypeOf(any);
    const a = asTensor(any);
    const norm1 = A.signedToUnsignedDim(dim1);
    const norm2 = A.signedToUnsignedDim(dim2);
    if (norm1 != norm2) {
        var new_strides = a.strides[0..a.ndims].*;
        new_strides[norm1] = a.strides[norm2];
        new_strides[norm2] = a.strides[norm1];
        return a.view(
            Transpose(any, norm1, norm2).shape,
            new_strides,
            a.offset,
        );
    } else {
        return a;
    }
}
/// Shorthand for transposing rightmost dimensions of tensor
pub fn T(comptime a: anytype) Transpose(a, -2, -1) {
    return asTensor(a).transpose(-2, -1);
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

pub fn Unsqueeze(comptime any: anytype, comptime dim: i16) type {
    const A = TensorTypeOf(any);
    return Reshape(any, utils.arrayInsert(A.ndims, A.shape, A.signedToUnsignedDim(dim), 1));
}
/// Insert a dim of size 1 into the shape of the tensor.
pub fn unsqueeze(comptime any: anytype, comptime dim: i16) Unsqueeze(any, dim) {
    const A = TensorTypeOf(any);
    const a = asTensor(any);
    return a.view(
        Unsqueeze(any, dim).shape,
        utils.arrayInsert(a.ndims, a.strides[0..a.ndims].*, A.signedToUnsignedDim(dim), 0),
        a.offset,
    );
}

// =============================================================================
// Compound functions
// =============================================================================

pub fn div(a: anytype, b: anytype) TensorTypeOf(a).BinaryFnResultType(b, .Add) {
    return asTensor(a).startGroup("div").mul(tensor.asTensor(b).recip()).endGroup();
}
pub fn sub(a: anytype, b: anytype) TensorTypeOf(a).BinaryFnResultType(b, .Add) {
    return asTensor(a).add(tensor.asTensor(b).neg());
}
pub fn exp(a: anytype) FloatTensor(TensorTypeOf(a)) {
    return asTensor(a).startGroup("exp").mul(INV_LN_2).exp2().endGroup();
}
pub fn log(a: anytype) FloatTensor(TensorTypeOf(a)) {
    return asTensor(a).startGroup("log").log2().mul(LN_2).endGroup();
}
pub fn sigmoid(a: anytype) FloatTensor(TensorTypeOf(a)) {
    const x = asTensor(a).startGroup("sigmoid");
    const x_pos = x.neg().exp().add(1.0).recip();
    const x_neg = x.exp().div(x.exp().add(1.0));
    const mask = x.lessThan(0.0);
    return mask.where(x_neg, x_pos).endGroup();
}
pub fn relu(a: anytype) TensorTypeOf(a) {
    if (dtypes.isFloat(a.dtype)) {
        return asTensor(a).startGroup("relu").maximum(0.0).endGroup();
    } else if (dtypes.isInt(a.dtype)) {
        return asTensor(a).startGroup("relu").maximum(0).endGroup();
    } else {
        unreachable;
    }
}
pub fn softmax(a: anytype, comptime dim: i16) FloatTensor(TensorTypeOf(a)) {
    const x = asTensor(a).startGroup("softmax");
    const minus_max_exp = x.sub(x.max({})).exp();
    const sumexp = minus_max_exp.sum(dim);
    return minus_max_exp.div(sumexp).endGroup();
}
pub fn mean(a: anytype, comptime dims: anytype) FloatTensor(TensorTypeOf(a).ReduceFnResultType(dims)) {
    return a.div(a.sum(dims));
}
pub fn variance(a: anytype, comptime dims: anytype) FloatTensor(TensorTypeOf(a).ReduceFnResultType(dims)) {
    const x = asTensor(a).startGroup("variance");
    const mu = x.mean(dims);
    const N: f64 = @floatFromInt(@divExact(x.num_entries, mu.num_entries));
    const a_minus_mu = x.sub(mu);
    return (a_minus_mu.mul(a_minus_mu)).sum(dims).div(N);
}

pub fn MatMul(a: anytype, b: anytype) type {
    const A = TensorTypeOf(a);
    const B = TensorTypeOf(b);

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
pub fn matmul(a: anytype, b: anytype) MatMul(a, b) {
    const A = TensorTypeOf(a);
    const B = TensorTypeOf(b);
    return asTensor(a).startGroup("matmul")
        .unsqueeze(A.ndims - 1)
        .mul(asTensor(b).transpose(B.ndims - 2, B.ndims - 1).unsqueeze(B.ndims - 2))
        .sum(A.ndims)
        .squeeze(A.ndims)
        .endGroup();
}

// TODO: Might want to practice with Conv1d first
// pub fn Conv2d(comptime Filter: type, _stride: anytype, _) type {
//     const stride: [2]u64 = switch (@typeInfo(@TypeOf(_stride)) {
//         .ComptimeInt, .Int => [2]u64{ _stride, _stride },
//         .Array => blk: {
//             if (_stride.len != 2) {
//                 @compileError("2D convolution stride must be a 2 element tuple");
//             }
//             break :blk _stride;
//         },
//         else => {
//             @compileError("2D convolution stride must be 1 number of a 2 element tuple");
//         },
//     };

//     if (ndims == 4) {
//         const batch_size = shape[0];
//         const in_channels = shape[1];
//         const in_height = shape[2];
//         const in_width = shape[3];

//         const out_height = @divFloor(in_height + 2 * , denominator: T)
//     }

// }

test "unary" {
    const tensor1 = comptime Tensor([2][3][4]i32).full(3);
    const tensor2 = comptime tensor1.neg();
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 4 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expect(tensor2.meta.op_tracker.UnaryOp.op == .Neg);
    try std.testing.expectEqual(tensor2.meta.op_tracker.UnaryOp.in[0].toTensor().*, tensor1);
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
