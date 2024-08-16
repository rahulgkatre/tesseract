const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

const ops = @import("../ops.zig");
const utils = @import("../utils.zig");
const dtypes = @import("../dtypes.zig");

const tensor = @import("tensor.zig");

const tensor_typing = @import("tensor_typing.zig");
const TensorType = tensor_typing.TensorType;
const TensorTypeOf = tensor_typing.TensorTypeOf;
const asTensor = tensor_typing.asTensor;

pub const INV_LN_2 = asTensor(1.4426950408888495760773985077695);
pub const LN_2 = INV_LN_2.recip();

const DimRange = struct {
    from: i16 = 0,
    to: i16 = -1,
};

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

fn UnaryFnType(comptime op: ops.UnaryOp) type {
    return @TypeOf(struct {
        pub fn func(input: anytype) TensorTypeOf(input).UnaryOpResultType(op) {
            return asTensor(input).applyUnaryOp(op);
        }
    }.func);
}

fn unaryFn(comptime op: ops.UnaryOp) UnaryFnType(op) {
    return struct {
        pub fn func(input: anytype) TensorTypeOf(input).UnaryOpResultType(op) {
            return asTensor(input).applyUnaryOp(op);
        }
    }.func;
}

pub const exp2 = unaryFn(.exp2);
pub const log2 = unaryFn(.log2);
pub const neg = unaryFn(.neg);
pub const recip = unaryFn(.recip);
pub const sin = unaryFn(.sin);
pub const sqrt = unaryFn(.sqrt);

test unaryFn {
    _ = exp2;
    _ = log2;
    _ = neg;
    _ = recip;
    _ = sin;
    _ = sqrt;
    const tensor1 = comptime Tensor([2][3][4]i32).full(3);
    const tensor2 = comptime tensor1.neg();
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 4 }, tensor2.layout.shape);
    try std.testing.expect(tensor2.instr.UnaryOp.op == .neg);
    try std.testing.expectEqual(tensor2.instr.UnaryOp.in[0].toTensor().*, tensor1);
}

// =============================================================================
// Binary functions
// =============================================================================

fn BinaryFnType(comptime op: ops.BinaryOp) type {
    return @TypeOf(struct {
        pub fn func(input: anytype, other: anytype) TensorTypeOf(input).BinaryOpResultType(other, op) {
            return asTensor(input).applyBinaryOp(other, op);
        }
    }.func);
}

fn binaryFn(comptime op: ops.BinaryOp) BinaryFnType(op) {
    return struct {
        pub fn func(input: anytype, other: anytype) TensorTypeOf(input).BinaryOpResultType(other, op) {
            return asTensor(input).applyBinaryOp(other, op);
        }
    }.func;
}

pub const add = binaryFn(.add);
pub const mul = binaryFn(.mul);
pub const maximum = binaryFn(.max);
pub const mod = binaryFn(.mod);
pub const lessThan = binaryFn(.lt);
pub const equals = binaryFn(.eq);
pub const xor = binaryFn(.xor);

test binaryFn {
    _ = add;
    _ = mul;
    _ = maximum;
    _ = mod;
    _ = lessThan;
    _ = equals;
    _ = xor;
    const tensor1 = comptime Tensor([2][1][4]i32).full(2);
    const tensor2 = comptime Tensor([3][1]i32).full(3);
    const tensor3 = comptime tensor1.add(tensor2);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 4 }, tensor3.layout.shape);
    try std.testing.expect(tensor3.instr.BinaryOp.op == .add);
    try std.testing.expectEqualDeep(tensor3.instr.BinaryOp.in[0].instr.DataOp.in[0].toTensor().*, tensor1);
    try std.testing.expectEqualDeep(tensor3.instr.BinaryOp.in[1].instr.DataOp.in[0].toTensor().*, tensor2);
}

// =============================================================================
// Reduce functions
// =============================================================================

fn ReduceFnType(comptime op: ops.ReduceOp) type {
    return @TypeOf(struct {
        pub fn func(input: anytype, comptime dims: anytype) TensorTypeOf(input).ReduceOpResultType(dims) {
            return asTensor(input).applyReduceOp(op, dims);
        }
    }.func);
}

fn reduceFn(comptime op: ops.ReduceOp) ReduceFnType(op) {
    return struct {
        pub fn func(input: anytype, comptime dims: anytype) TensorTypeOf(input).ReduceOpResultType(dims) {
            return asTensor(input).applyReduceOp(op, dims);
        }
    }.func;
}

pub const sum = reduceFn(.add);
pub const prod = reduceFn(.mul);
pub const max = reduceFn(.max);

test reduceFn {
    _ = sum;
    _ = prod;
    _ = max;
    const tensor1 = comptime Tensor([2][3][4]i32).full(5);

    const tensor2 = comptime tensor1.sum(1);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 1, 4 }, tensor2.layout.shape);
    try std.testing.expect(tensor2.instr.ReduceOp.op == .add);
    try std.testing.expectEqual(tensor2.instr.ReduceOp.in[0].toTensor().*, tensor1);
    try std.testing.expectEqual(tensor2.instr.ReduceOp.args.mask, &[_]bool{ false, true, false });

    const tensor3 = comptime tensor1.sum(.{ 0, 1 });
    try std.testing.expectEqualSlices(u64, &[_]u64{ 1, 1, 4 }, tensor3.layout.shape);
    try std.testing.expect(tensor3.instr.ReduceOp.op == .add);
    try std.testing.expectEqual(tensor3.instr.ReduceOp.in[0].toTensor().*, tensor1);
    try std.testing.expectEqualDeep(tensor3.instr.ReduceOp.args.mask, &[_]bool{ true, true, false });
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
            bc_strides[new_shape.len - i - 1] = if (i >= a.layout.ndims) 0 else a.layout.strides[a.layout.ndims - i - 1];
        }
        break :blk bc_strides;
    };

    return a.view(
        new_shape,
        bc_strides,
        a.layout.offset,
    );
}

// =============================================================================
// Flatten
// =============================================================================

pub fn Flatten(input: anytype, comptime dims: DimRange) type {
    const A = TensorTypeOf(input);
    const from = A.signedToUnsignedDim(dims.from);
    const to = A.signedToUnsignedDim(dims.to);
    if (from == to) {
        return A;
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
pub fn flatten(input: anytype, comptime dims: DimRange) Flatten(input, dims) {
    return asTensor(input).reshape(Flatten(input, dims).shape);
}

// =============================================================================
//
// =============================================================================
/// Get a mask of where padding values exist in the tensor
pub fn paddingMask(input: anytype, comptime padding: anytype) TensorType(.bool, TensorTypeOf(input).shape) {
    const a = asTensor(input);
    const not_padding = TensorType(.bool, a.shape).full(true);
    return not_padding.pad(padding, .{ .constant = .{ .value = false } });
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
        utils.arrayPermute(u64, A.ndims, a.layout.strides[0..A.ndims].*, perm),
        a.layout.offset,
    );
}
test permute {
    const tensor1 = comptime Tensor([2][3][4]f32).full(0);
    const tensor2 = comptime tensor1.permute(.{ 0, 2, 1 });
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 4, 3 }, tensor2.layout.shape);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 12, 1, 4 }, tensor2.layout.strides);
}

// =============================================================================
// Reshape
// =============================================================================

pub fn Reshape(comptime input: anytype, comptime new_shape: anytype) type {
    const OldType = TensorTypeOf(input);
    const NewType = TensorType(OldType.dtype, new_shape);
    std.debug.assert(OldType.num_elements == NewType.num_elements);
    return NewType;
}
/// Change the shape of the  This changes the type too.
pub fn reshape(comptime input: anytype, comptime new_shape: anytype) Reshape(input, new_shape) {
    const a = asTensor(input);
    return a.contiguous().view(new_shape, Reshape(a, new_shape).contiguous_strides, a.layout.offset);
}
test reshape {
    const tensor1 = comptime Tensor([2][3][4]i32).full(0);
    const tensor2 = comptime tensor1.reshape(.{ 12, 2 });
    const tensor3 = comptime tensor2.reshape(.{24});
    try std.testing.expectEqualSlices(u64, &[_]u64{ 12, 2 }, tensor2.layout.shape);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 1 }, tensor2.layout.strides);
    try std.testing.expectEqualSlices(u64, &[_]u64{24}, tensor3.layout.shape);
    try std.testing.expectEqualSlices(u64, &[_]u64{1}, tensor3.layout.strides);
}

// =============================================================================
// Squeeze
// =============================================================================
pub fn Squeeze(comptime input: anytype, comptime dim: i16) type {
    const A = TensorTypeOf(input);
    if (A.shape[A.signedToUnsignedDim(dim)] != 1) {
        @compileError("Cannot squeeze as dimension size is not 1");
    }
    return Reshape(input, utils.arrayDelete(A.ndims, A.shape, A.signedToUnsignedDim(dim)));
}
/// Remove a dim of size 1 from the shape of the input
pub fn squeeze(comptime input: anytype, comptime dim: i16) Squeeze(input, dim) {
    const A = TensorTypeOf(input);
    const a = asTensor(input);
    return a.view(
        Squeeze(input, dim).shape,
        utils.arrayDelete(A.ndims, a.layout.strides[0..A.ndims].*, A.signedToUnsignedDim(dim)),
        a.layout.offset,
    );
}

// =============================================================================
// Transpose
// =============================================================================
pub fn Transpose(comptime input: anytype, comptime dim1: i16, comptime dim2: i16) type {
    const Type = TensorTypeOf(input);
    const norm1 = utils.signedToUnsignedDimNdims(Type.ndims, dim1);
    const norm2 = utils.signedToUnsignedDimNdims(Type.ndims, dim2);
    var new_shape = Type.shape;
    new_shape[norm1] = Type.shape[norm2];
    new_shape[norm2] = Type.shape[norm1];
    return Reshape(input, new_shape);
}
/// Transpose two dimensions of the  Similar to permute, but only for two dimensions.
pub fn transpose(comptime input: anytype, comptime dim1: i16, comptime dim2: i16) Transpose(input, dim1, dim2) {
    const A = TensorTypeOf(input);
    const a = asTensor(input);
    const norm1 = A.signedToUnsignedDim(dim1);
    const norm2 = A.signedToUnsignedDim(dim2);
    if (norm1 != norm2) {
        var new_strides = a.layout.strides[0..a.layout.ndims].*;
        new_strides[norm1] = a.layout.strides[norm2];
        new_strides[norm2] = a.layout.strides[norm1];
        return a.view(
            Transpose(input, norm1, norm2).shape,
            new_strides,
            a.layout.offset,
        );
    } else {
        return a;
    }
}
/// Shorthand for transposing rightmost dimensions of tensor
pub fn T(comptime input: anytype) Transpose(input, -2, -1) {
    return asTensor(input).transpose(-2, -1);
}
test transpose {
    const tensor1 = comptime Tensor([2][1][4]i32).full(1);
    const tensor2 = comptime tensor1.T();
    try std.testing.expectEqualDeep(tensor2, comptime tensor1.transpose(-2, -1));
    try std.testing.expectEqualDeep(tensor1.layout.shape, comptime tensor2.T().layout.shape);
    try std.testing.expectEqualDeep(tensor1.layout.strides, comptime tensor2.T().layout.strides);
}

// =============================================================================
// Unsqueeze
// =============================================================================
pub fn Unsqueeze(comptime input: anytype, comptime dim: i16) type {
    const A = TensorTypeOf(input);
    return Reshape(input, utils.arrayInsert(A.ndims, A.shape, A.signedToUnsignedDim(dim), 1));
}
/// Insert a dim of size 1 into the shape of the tensor
pub fn unsqueeze(comptime input: anytype, comptime dim: i16) Unsqueeze(input, dim) {
    const A = TensorTypeOf(input);
    const a = asTensor(input);
    return a.view(
        Unsqueeze(input, dim).shape,
        utils.arrayInsert(a.layout.ndims, a.layout.strides[0..a.layout.ndims].*, A.signedToUnsignedDim(dim), 0),
        a.layout.offset,
    );
}

// =============================================================================
// Compound functions
// =============================================================================
pub fn div(input: anytype, other: anytype) tensor.FloatTensor(TensorTypeOf(input).BinaryFnResultType(other, .mul)) {
    return asTensor(input).mul(asTensor(other).recip());
}
pub fn sub(input: anytype, other: anytype) TensorTypeOf(input).BinaryFnResultType(other, .add) {
    return asTensor(input).add(asTensor(other).neg());
}
pub fn exp(input: anytype) tensor.FloatTensor(TensorTypeOf(input)) {
    const x = asTensor(input);
    return x.mul(INV_LN_2).exp2();
}
pub fn log(input: anytype) tensor.FloatTensor(TensorTypeOf(input)) {
    const x = asTensor(input);
    return x.log2().mul(LN_2);
}
pub fn sigmoid(input: anytype) tensor.FloatTensor(TensorTypeOf(input)) {
    const x = asTensor(input);
    const x_pos = x.neg().exp().add(1.0).recip();
    const x_neg = x.exp().div(x.exp().add(1.0));
    const mask = x.lessThan(0.0);
    return mask.where(x_neg, x_pos);
}
pub fn relu(input: anytype) TensorTypeOf(input) {
    if (dtypes.isFloat(input.dtype)) {
        return asTensor(input).maximum(0.0);
    } else if (dtypes.isInt(input.dtype)) {
        return asTensor(input).maximum(0);
    } else {
        unreachable;
    }
}
pub fn softmax(input: anytype, comptime dim: i16) tensor.FloatTensor(TensorTypeOf(input)) {
    const x = asTensor(input);
    const minus_max_exp = x.sub(x.max({})).exp();
    const sumexp = minus_max_exp.sum(dim);
    return minus_max_exp.div(sumexp);
}
pub fn mean(input: anytype, comptime dims: anytype) tensor.FloatTensor(TensorTypeOf(input).ReduceFnResultType(dims)) {
    return input.div(input.sum(dims));
}
pub fn variance(input: anytype, comptime dims: anytype) tensor.FloatTensor(TensorTypeOf(input).ReduceFnResultType(dims)) {
    const x = asTensor(input);
    const mu = x.mean(dims);
    const N: f64 = @floatFromInt(@divExact(x.num_elements, mu.num_elements));
    const a_minus_mu = x.sub(mu);
    return a_minus_mu.mul(a_minus_mu).sum(dims).div(N);
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
            \\
            \\Tensors do not satisfy the matrix multiplication invariant, dim n-1 of A must equal dim n-2 of B
            \\Tensor A: {a}
            \\Tensor B: {a}
            \\A.shape[n-2] = {d}, A.shape[n-1] = {d}, B.shape[n-2] = {d}, B.shape[n-1] = {d}
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
    const a = asTensor(input);
    const b = asTensor(other);
    return a.unsqueeze(a.layout.ndims - 1)
        .mul(b.transpose(a.layout.ndims - 2, b.layout.ndims - 1).unsqueeze(b.layout.ndims - 2))
        .sum(a.layout.ndims)
        .squeeze(a.layout.ndims);
}

pub fn linear(input: anytype, weight: anytype, bias: anytype) MatMul(input, weight) {
    return asTensor(input).matmul(weight).add(bias);
}

pub fn Window1d(input: anytype, window: u64) type {
    const I = TensorTypeOf(input);
    return TensorType(I.dtype, I.shape[0 .. I.ndims - 1] ++ .{ I.shape[I.ndims - 1] - window + 1, window });
}
pub fn window1d(input: anytype, window: u64) Window1d(input, window) {
    const Result = Window1d(input, window);
    const a = asTensor(input);
    return a.view(Result.shape, a.strides.* ++ .{a.strides[a.ndims - 1]}, a.offset);
}
