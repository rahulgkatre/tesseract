const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

const ops = @import("../ops.zig");
const utils = @import("../utils.zig");
const dtypes = @import("../dtypes.zig");

const tensor = @import("tensor.zig");

const typing = @import("tensor_typing.zig");
const TensorType = typing.TensorType;
const AsTensorType = typing.AsTensorType;
const asTensor = typing.asTensor;

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
// pub fn range(
//     comptime start: comptime_int,
//     comptime stop: comptime_int,
// ) Tensor([stop - start]comptime_int) {
//     return Tensor([stop - start]comptime_int).range(start, stop);
// }

/// Create a tensor with the same shape as the input but filled with random values
pub fn randLike(
    input: anytype,
    allocator: std.mem.Allocator,
) *const AsTensorType(@TypeOf(input)) {
    return AsTensorType(@TypeOf(input)).random(allocator);
}

/// Create a tensor with the same shape as the input but filled with a constant
pub fn fullLike(
    input: anytype,
    value: dtypes.ZigType(input.dtype),
    allocator: std.mem.Allocator,
) *const AsTensorType(@TypeOf(input)) {
    return AsTensorType(@TypeOf(input)).full(value, allocator);
}

// =============================================================================
// Unary functions
// =============================================================================

fn UnaryFn(comptime op: ops.UnaryOp) type {
    return @TypeOf(struct {
        pub fn func(x: anytype, allocator: std.mem.Allocator) *const typing.UnaryOpResult(@TypeOf(x), op) {
            const t = asTensor(x, allocator);
            return t.applyUnaryOp(op, allocator);
        }
    }.func);
}

fn unaryFn(comptime op: ops.UnaryOp) UnaryFn(op) {
    return struct {
        pub fn func(x: anytype, allocator: std.mem.Allocator) *const typing.UnaryOpResult(@TypeOf(x), op) {
            const t = asTensor(x, allocator);
            return t.applyUnaryOp(op, allocator);
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
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const tensor1 = Tensor([2][3][4]i32).full(3, allocator);
    const tensor2 = tensor1.neg(allocator);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 4 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expect(tensor2.meta.instr.UnaryOp.op == .neg);
    // std.testing.expect(tensor2.meta.instr.UnaryOp.in[0] == tensor1.toAnyTensor());
}

// =============================================================================
// Binary functions
// =============================================================================

fn BinaryFn(comptime op: ops.BinaryOp) type {
    return @TypeOf(struct {
        pub fn func(a: anytype, b: anytype, allocator: std.mem.Allocator) *const typing.BinaryOpResult(@TypeOf(a), @TypeOf(b), op) {
            const t = asTensor(a, allocator);
            return t.applyBinaryOp(b, op, allocator);
        }
    }.func);
}

fn binaryFn(comptime op: ops.BinaryOp) BinaryFn(op) {
    return struct {
        pub fn func(a: anytype, b: anytype, allocator: std.mem.Allocator) *const typing.BinaryOpResult(@TypeOf(a), @TypeOf(b), op) {
            const t = asTensor(a, allocator);
            return t.applyBinaryOp(b, op, allocator);
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
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const tensor1 = Tensor([2][1][4]i32).full(2, allocator);
    const tensor2 = Tensor([3][1]i32).full(3, allocator);
    const tensor3 = tensor1.add(tensor2, allocator);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 4 }, tensor3.shape[0..tensor3.ndims]);
    try std.testing.expect(tensor3.meta.instr.BinaryOp.op == .add);
    try std.testing.expectEqualDeep(tensor3.meta.instr.BinaryOp.in[0].meta.instr.DataOp.in[0], tensor1.toAnyTensor());
    try std.testing.expectEqualDeep(tensor3.meta.instr.BinaryOp.in[1].meta.instr.DataOp.in[0], tensor2.toAnyTensor());
}

// =============================================================================
// Reduce functions
// =============================================================================

fn ReduceFnType(comptime op: ops.ReduceOp) type {
    return @TypeOf(struct {
        pub fn func(x: anytype, comptime dims: anytype, allocator: std.mem.Allocator) !typing.ReduceOpResult(@TypeOf(x), dims) {
            const t = asTensor(x, allocator);
            return t.applyReduceOp(op, dims, allocator);
        }
    }.func);
}

fn reduceFn(comptime op: ops.ReduceOp) ReduceFnType(op) {
    return struct {
        pub fn func(x: anytype, comptime dims: anytype, allocator: std.mem.Allocator) !typing.ReduceOpResult(@TypeOf(x), dims) {
            const t = asTensor(x, allocator);
            return t.applyReduceOp(op, dims, allocator);
        }
    }.func;
}

pub const sum = reduceFn(.add);
pub const prod = reduceFn(.mul);
pub const max = reduceFn(.max);

// test reduceFn {
//     _ = sum;
//     _ = prod;
//     _ = max;
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();
//     const allocator = arena.allocator();
//     const tensor1 = Tensor([2][3][4]i32).full(allocator, 5);

//     const tensor2 = tensor1.sum(1);
//     std.testing.expectEqualSlices(u64, &[_]u64{ 2, 1, 4 }, tensor2.shape[0..tensor1.ndims]);
//     std.testing.expect(tensor2.meta.instr.ReduceOp.op == .add);
//     std.testing.expectEqual(tensor2.meta.instr.ReduceOp.in[0].toTensor().*, tensor1);
//     std.testing.expectEqual(tensor2.meta.instr.ReduceOp.args.mask[0..tensor2.ndims].*, ([_]bool{ false, true, false }));

//     const tensor3 = comptime tensor1.sum(.{ 0, 1 });
//     std.testing.expectEqualSlices(u64, &[_]u64{ 1, 1, 4 }, tensor3.shape[0..tensor3.ndims]);
//     std.testing.expect(tensor3.meta.instr.ReduceOp.op == .add);
//     std.testing.expectEqual(tensor3.meta.instr.ReduceOp.in[0].toTensor().*, tensor1);
//     std.testing.expectEqualDeep(tensor3.meta.instr.ReduceOp.args.mask[0..tensor3.ndims], &[_]bool{ true, true, false });
// }

// =============================================================================
// Expand
// =============================================================================

pub fn Expand(comptime X: type, comptime new_shape: anytype) type {
    const A = AsTensorType(X);
    return TensorType(A.dtype, utils.broadcastShape(A.shape, new_shape));
}
/// Expand a tensor along 1 or more dimensions with size 1 and stride 0
/// The new shape must broadcast with the old shape
pub fn expand(
    x: anytype,
    comptime new_shape: anytype,
    allocator: std.mem.Allocator,
) *const Expand(@TypeOf(x), new_shape) {
    const A = comptime AsTensorType(@TypeOf(x));
    const a = asTensor(x, allocator);
    const Out = comptime Expand(@TypeOf(x), new_shape);
    if (A == Out) {
        return a;
    }
    const new_strides: [new_shape.len]u64 = blk: {
        var bc_strides: [new_shape.len]u64 = undefined;
        for (0..new_shape.len) |i| {
            bc_strides[new_shape.len - i - 1] = if (i >= a.ndims) 0 else a.strides[a.ndims - i - 1];
        }
        break :blk bc_strides;
    };
    return a.view(
        new_shape,
        new_strides,
        a.offset,
        allocator,
    );
}

// =============================================================================
// Flatten
// =============================================================================

pub fn Flatten(comptime X: type, comptime dims: DimRange) type {
    const A = AsTensorType(X);
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
    return Reshape(X, new_shape);
}
/// Flatten a range of dims, collapsing them to 1 dimension
pub fn flatten(
    x: anytype,
    comptime dims: DimRange,
    allocator: std.mem.Allocator,
) *const Flatten(@TypeOf(x), dims) {
    const t = asTensor(x, allocator);
    return t.reshape(Flatten(@TypeOf(x), dims).shape, allocator);
}

// =============================================================================
//
// =============================================================================
/// Get a mask of where padding values exist in the tensor
pub fn paddingMask(input: anytype, comptime padding: anytype) TensorType(.bool, AsTensorType(@TypeOf(input)).shape) {
    const a = asTensor(input);
    const not_padding = TensorType(.bool, a.shape).full(true);
    return not_padding.pad(padding, .{ .constant = .{ .value = false } });
}

// =============================================================================
// Permute
// =============================================================================
pub fn Permute(comptime X: type, comptime perm: [AsTensorType(X).ndims]u8) type {
    const A = AsTensorType(X);
    return Reshape(X, utils.arrayPermute(u64, A.ndims, A.shape, perm));
}
/// Permute the dimensions of the  A valid permutation must contain
/// values from 0 to ndims and each value must appear exactly once.
pub fn permute(
    x: anytype,
    comptime perm: [AsTensorType(@TypeOf(x)).ndims]u8,
    allocator: std.mem.Allocator,
) *const Permute(@TypeOf(x), perm) {
    const A = AsTensorType(@TypeOf(x));
    const a = asTensor(x, allocator);
    return a.view(
        Permute(@TypeOf(x), perm).shape,
        utils.arrayPermute(u64, A.ndims, a.strides[0..A.ndims].*, perm),
        a.offset,
        allocator,
    );
}
test permute {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const tensor1 = Tensor([2][3][4]f32).full(0, allocator);
    const tensor2 = tensor1.permute(.{ 0, 2, 1 }, allocator);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 4, 3 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 12, 1, 4 }, tensor2.strides[0..tensor2.ndims]);
}

// =============================================================================
// Reshape
// =============================================================================

pub fn Reshape(comptime X: type, comptime new_shape: anytype) type {
    const OldType = AsTensorType(X);
    const NewType = TensorType(OldType.dtype, new_shape);
    std.debug.assert(OldType.num_elements == NewType.num_elements);
    return NewType;
}
/// Change the shape of the  This changes the type too.
pub fn reshape(
    x: anytype,
    comptime new_shape: anytype,
    allocator: std.mem.Allocator,
) *const Reshape(@TypeOf(x), new_shape) {
    const a = asTensor(x, allocator);
    const a_contiguous = a.contiguous(allocator);
    return a_contiguous.view(
        new_shape,
        Reshape(@TypeOf(x), new_shape).contiguous_strides,
        a.offset,
        allocator,
    );
}
test reshape {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const tensor1 = Tensor([2][3][4]i32).full(0, allocator);
    const tensor2 = tensor1.reshape(.{ 12, 2 }, allocator);
    const tensor3 = tensor2.reshape(.{24}, allocator);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 12, 2 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 1 }, tensor2.strides[0..tensor2.ndims]);
    try std.testing.expectEqualSlices(u64, &[_]u64{24}, tensor3.shape[0..tensor3.ndims]);
    try std.testing.expectEqualSlices(u64, &[_]u64{1}, tensor3.strides[0..tensor3.ndims]);
}

// =============================================================================
// Squeeze
// =============================================================================
pub fn Squeeze(comptime X: anytype, comptime dim: i16) type {
    const A = AsTensorType(@TypeOf(X));
    if (A.shape[A.signedToUnsignedDim(dim)] != 1) {
        @compileError("Cannot squeeze as dimension size is not 1");
    }
    return Reshape(A, utils.arrayDelete(A.ndims, A.shape, A.signedToUnsignedDim(dim)));
}
/// Remove a dim of size 1 from the shape of the input
pub fn squeeze(
    x: anytype,
    comptime dim: i16,
    allocator: std.mem.Allocator,
) *const Squeeze(x, dim) {
    const A = AsTensorType(@TypeOf(x));
    const a = asTensor(x, allocator);
    return a.view(
        Squeeze(x, dim).shape,
        utils.arrayDelete(A.ndims, a.strides[0..A.ndims].*, A.signedToUnsignedDim(dim)),
        a.offset,
        allocator,
    );
}

// =============================================================================
// Transpose
// =============================================================================
pub fn Transpose(comptime X: type, comptime dim1: i16, comptime dim2: i16) type {
    const Type = AsTensorType(X);
    const norm1 = utils.signedToUnsignedDimNdims(Type.ndims, dim1);
    const norm2 = utils.signedToUnsignedDimNdims(Type.ndims, dim2);
    var new_shape = Type.shape;
    new_shape[norm1] = Type.shape[norm2];
    new_shape[norm2] = Type.shape[norm1];
    return Reshape(X, new_shape);
}
/// Transpose two dimensions of the  Similar to permute, but only for two dimensions.
pub fn transpose(
    x: anytype,
    comptime dim1: i16,
    comptime dim2: i16,
    allocator: std.mem.Allocator,
) *const Transpose(@TypeOf(x), dim1, dim2) {
    const A = comptime AsTensorType(@TypeOf(x));
    const norm1 = comptime A.signedToUnsignedDim(dim1);
    const norm2 = comptime A.signedToUnsignedDim(dim2);
    const a = asTensor(x, allocator);
    if (norm1 != norm2) {
        var new_strides: [A.ndims]u64 = a.strides[0..A.ndims].*;
        new_strides[norm1] = a.strides[norm2];
        new_strides[norm2] = a.strides[norm1];
        return a.view(
            Transpose(A, norm1, norm2).shape,
            new_strides,
            a.offset,
            allocator,
        );
    } else {
        return a;
    }
}
/// Shorthand for transposing rightmost dimensions of tensor
pub fn T(
    x: anytype,
    allocator: std.mem.Allocator,
) *const Transpose(@TypeOf(x), -2, -1) {
    const t = asTensor(x, allocator);
    return t.transpose(-2, -1, allocator);
}
test transpose {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const tensor1 = Tensor([2][1][4]i32).full(1, allocator);
    const tensor2 = tensor1.T(allocator);
    const ndims = tensor1.ndims;
    try std.testing.expectEqualDeep(tensor1.shape[0..ndims], (tensor2.T(allocator)).shape[0..ndims]);
    try std.testing.expectEqualDeep(tensor1.strides[0..ndims], (tensor2.T(allocator)).strides[0..ndims]);
}

// =============================================================================
// Unsqueeze
// =============================================================================
pub fn Unsqueeze(comptime x: anytype, comptime dim: i16) type {
    const X = AsTensorType(@TypeOf(x));
    return Reshape(X, utils.arrayInsert(X.ndims, X.shape, X.signedToUnsignedDim(dim), 1));
}
/// Insert a dim of size 1 into the shape of the tensor
pub fn unsqueeze(
    x: anytype,
    comptime dim: i16,
    allocator: std.mem.Allocator,
) *const Unsqueeze(x, dim) {
    const A = comptime AsTensorType(@TypeOf(x));
    const a = asTensor(x, allocator);
    return a.view(
        Unsqueeze(x, dim).shape,
        utils.arrayInsert(a.ndims, a.strides[0..a.ndims].*, A.signedToUnsignedDim(dim), 0),
        a.offset,
        allocator,
    );
}

// =============================================================================
// Compound functions
// =============================================================================
pub fn div(
    input: anytype,
    other: anytype,
    allocator: std.mem.Allocator,
) typing.FloatTensor(typing.BinaryOpResult(@TypeOf(input), @TypeOf(other), .add)) {
    const a = asTensor(input, allocator);
    const b = asTensor(other, allocator);
    return a.mul(b.recip(allocator), allocator);
}
pub fn sub(
    input: anytype,
    other: anytype,
    allocator: std.mem.Allocator,
) typing.BinaryOpResult(@TypeOf(input), @TypeOf(other), .add) {
    const a = asTensor(input, allocator);
    const b = asTensor(other, allocator);
    return a.add(b.neg(allocator), allocator);
}
pub fn exp(
    input: anytype,
    allocator: std.mem.Allocator,
) *const typing.FloatTensor(AsTensorType(@TypeOf(input))) {
    return asTensor(input, allocator).mul(INV_LN_2, allocator).exp2(allocator);
}
pub fn log(
    input: anytype,
    allocator: std.mem.Allocator,
) *const typing.FloatTensor(AsTensorType(@TypeOf(input))) {
    return asTensor(input, allocator).log2(allocator).mul(LN_2, allocator);
}
// pub fn sigmoid(
//     input: anytype,
//     allocator: std.mem.Allocator,
// ) typing.FloatTensor(AsTensorType(@TypeOf(input))) {
//     const x = asTensor(input, allocator);
//     const x_pos = x.neg().exp().add(1.0).recip();
//     const x_neg = x.exp().div(x.exp().add(1.0));
//     const mask = x.lessThan(0.0);
//     return mask.where(x_neg, x_pos);
// }
pub fn relu(
    input: anytype,
    allocator: std.mem.Allocator,
) *const AsTensorType(@TypeOf(input)) {
    if (dtypes.isFloat(input.dtype)) {
        return (asTensor(input, allocator)).maximum(0.0, allocator);
    } else if (dtypes.isInt(input.dtype)) {
        return (asTensor(input, allocator)).maximum(0, allocator);
    } else {
        unreachable;
    }
}
// pub fn softmax(input: anytype, comptime dim: i16) tensor.FloatTensor(AsTensorType(@TypeOf(input))) {
//     const x = asTensor(input);
//     const minus_max_exp = x.sub(x.max({})).exp();
//     const sumexp = minus_max_exp.sum(dim);
//     return minus_max_exp.div(sumexp);
// }
pub fn mean(input: anytype, comptime dims: anytype) tensor.FloatTensor(AsTensorType(@TypeOf(input)).ReduceFnResultType(dims)) {
    return input.div(input.sum(dims));
}
pub fn variance(input: anytype, comptime dims: anytype) tensor.FloatTensor(AsTensorType(@TypeOf(input)).ReduceFnResultType(dims)) {
    const x = asTensor(input);
    const mu = x.mean(dims);
    const N: f64 = @floatFromInt(@divExact(x.num_elements, mu.num_elements));
    const a_minus_mu = x.sub(mu);
    return a_minus_mu.mul(a_minus_mu).sum(dims).div(N);
}

pub fn MatMul(comptime I: type, comptime O: type) type {
    const A = AsTensorType(I);
    const B = AsTensorType(O);

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
pub fn matmul(
    input: anytype,
    other: anytype,
    allocator: std.mem.Allocator,
) MatMul(@TypeOf(input), @TypeOf(other)) {
    const a = asTensor(input, allocator);
    const b = asTensor(other, allocator);
    return a.unsqueeze(a.ndims - 1, allocator)
        .mul(b.transpose(a.ndims - 2, b.ndims - 1).unsqueeze(b.ndims - 2))
        .sum(a.ndims)
        .squeeze(a.ndims);
}

pub fn linear(input: anytype, weight: anytype, bias: anytype) MatMul(input, weight) {
    return asTensor(input).matmul(weight).add(bias);
}

pub fn Window1d(input: anytype, window: u64) type {
    const I = AsTensorType(@TypeOf(input));
    return TensorType(I.dtype, I.shape[0 .. I.ndims - 1] ++ .{ I.shape[I.ndims - 1] - window + 1, window });
}
pub fn window1d(input: anytype, window: u64) Window1d(input, window) {
    const Result = Window1d(input, window);
    const a = asTensor(input);
    return a.view(Result.shape, a.strides.* ++ .{a.strides[a.ndims - 1]}, a.offset);
}
