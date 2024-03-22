const std = @import("std");
const Allocator = std.mem.Allocator;
const comptimePrint = std.fmt.comptimePrint;
const utils = @import("utils.zig");
const ops = @import("ops.zig");
const Graph = @import("Graph.zig");
const dtypes = @import("dtypes.zig");

// Infer the contiguous stride pattern from the shape
// This is the default stride pattern unless a stride is manually provided
// using asStrided

fn contiguousStrides(comptime ndims: u8, shape: [ndims]u64) [ndims]u64 {
    var offset: u64 = 1;
    var strides: [ndims]u64 = undefined;
    for (0..ndims - 1) |d| {
        const stride = shape[ndims - d - 1] * offset;
        strides[ndims - d - 2] = stride;
        offset = stride;
    }
    strides[ndims - 1] = 1;
    for (0..ndims) |d| {
        if (shape[d] == 0 or shape[d] == 1) {
            strides[d] = 0;
        }
    }
    return strides;
}

pub fn range(
    comptime dtype: dtypes.DType,
    comptime start: if (dtypes.isInt(dtype)) comptime_int else @compileError("Range tensor must have int dtype"),
    comptime stop: if (dtypes.isInt(dtype)) comptime_int else @compileError("Range tensor must have int dtype"),
) UserTensor(dtype, .{stop - start}) {
    return UserTensor(dtype, .{stop - start}).range(start, stop);
}

fn Scalar(comptime dtype: dtypes.DType) type {
    return UserTensor(dtype, .{1});
}

/// Utility function for defining a scalar (a 1-size tensor)
pub fn scalar(comptime dtype: dtypes.DType, value: dtypes.ZigType(dtype)) Scalar(dtype) {
    return Scalar(dtype).full(value);
}

fn isTensor(comptime a: anytype) bool {
    const T = @TypeOf(a);
    return switch (@typeInfo(T)) {
        .Struct => Tensor(T.dtype, T.ndims, T.shape) == T,
        else => false,
    };
}

pub fn randLike(comptime other: anytype) @TypeOf(other) {
    std.debug.assert(isTensor(other));
    return @TypeOf(other).rand();
}

pub fn fullLike(comptime other: anytype, value: dtypes.ZigType(other.dtype)) @TypeOf(other) {
    std.debug.assert(isTensor(other));
    return @TypeOf(other).full(value);
}

pub fn UserTensor(comptime dtype: dtypes.DType, comptime shape: anytype) type {
    return Tensor(dtype, shape.len, shape);
}

pub fn Tensor(
    // Generic parameters are private so they will be redeclare as pub conts in the result type
    comptime _dtype: dtypes.DType,
    comptime _ndims: u8,
    comptime _shape: [_ndims]u64,
) type {
    return struct {
        // All the functions for operations are implemented separately
        pub usingnamespace @import("functions.zig");
        const Self = @This();

        // Type level constants for comptime shape logic (e.g. @TypeOf(a).ndims)
        pub const dtype: dtypes.DType = _dtype;
        pub const ndims: u8 = _ndims;
        pub const shape: [ndims]u64 = _shape;

        /// Used for determining the dtype of a scalar when applying zip or ternary ops
        /// with immediate values rather than tensor values
        fn scalarDType(comptime val: anytype) dtypes.DType {
            return switch (@typeInfo(@TypeOf(val))) {
                .ComptimeInt => if (dtypes.isInt(dtype)) dtype else .i32,
                .ComptimeFloat => if (dtypes.isFloat(dtype)) dtype else .f32,
                .Int, .Float, .Bool => @field(dtypes.DType, @typeName(@TypeOf(val))),
                else => @compileError(@typeName(@TypeOf(val)) ++ " is not a valid tensor element type"),
            };
        }

        /// Used for wrapping immediate values in single size tensors
        fn asTensor(comptime val: anytype) if (!isTensor(val)) Scalar(scalarDType(val)) else @TypeOf(val) {
            return if (!isTensor(val)) Scalar(scalarDType(val)).full(val) else return val;
        }

        ndims: u8 = ndims,
        dtype: dtypes.DType = dtype,
        shape: [ndims]u64 = shape,
        strides: [ndims]u64,
        offset: u64,
        last_op: Graph.AnyOp,

        pub fn initContiguous(comptime last_op: Graph.AnyOp) Self {
            return .{ .last_op = last_op, .strides = contiguousStrides(ndims, shape), .offset = 0 };
        }

        pub fn any(self: *const Self) Graph.AnyTensor {
            @setEvalBranchQuota(std.math.maxInt(u32));
            return .{
                .ptr = self,
                .dtype = dtype,
                .last_op = self.last_op,
                .ndims = ndims,
                .shape = &shape,
                .offset = self.offset,
                .strides = &self.strides,
                .size = self.size(),
                .contiguous = self.isContiguous(),
            };
        }

        pub fn trace(self: *const Self) void {
            self.any().trace();
        }

        pub fn isContiguous(self: Self) bool {
            var prev: u64 = std.math.maxInt(u64);
            for (self.strides) |s| {
                if (s > prev and s > 0) {
                    return false;
                }
                if (s > 0) {
                    prev = s;
                }
            }
            return true;
        }

        pub fn size(self: Self) u64 {
            // The storage size is 1 + last index calculated by the strides and shape
            // shape[d] - 1 is the last index in dimension d
            // Also incorporate the storage offset
            const strides = self.strides;
            var _size: u64 = self.offset + 1;
            for (0..ndims) |d| {
                _size += (shape[d] - 1) * strides[d];
            }
            // The result is the size of the storage needed to visit all indices of the tensor
            return _size;
        }

        /// Used to mark a tensor as an input to a graph,
        /// codegen will make this an argument of the function
        pub fn input() Self {
            return initContiguous(.{ .InitOp = .{ .op = .Input, .args = .{ .Input = {} } } });
        }

        /// Fill a tensor with a value
        pub fn full(comptime value: dtypes.ZigType(dtype)) Self {
            return initContiguous(.{ .InitOp = .{ .op = .Full, .args = .{ .Full = std.fmt.comptimePrint("{any}", .{value}) } } });
        }

        /// Internal function to fill with range, this is not publicly exposed
        /// as shape of range tensor must be constrained
        fn range(comptime start: comptime_int, comptime stop: comptime_int) Self {
            if (ndims != 1) {
                @compileError("Can only use range() on a tensor with exactly 1 dimension");
            }
            return initContiguous(.{ .InitOp = .Range }, {}, .{
                .Range = .{
                    .start = std.fmt.comptimePrint("{d}", .{start}),
                    .stop = std.fmt.comptimePrint("{d}", .{stop}),
                },
            });
        }

        pub fn rand() Self {
            std.debug.assert(dtypes.isFloat(_dtype));
            return initContiguous(.{ .InitOp = .{ .op = .Rand, .args = .{ .Rand = {} } } });
        }

        /// A copy is only needed to make a non-contiguous tensor contiguous again.
        /// Each tensor is immutable and operations already produce new tensors
        /// but intermediate tensors can be eliminated through optimization.
        pub fn copy(a: Self) Self {
            return UserTensor(dtype, shape).initContiguous(.{ .MapOp = .{ .op = .Id, .a = &a.any() } });
        }

        fn Permute(comptime perm: [ndims]u8) type {
            return View(utils.arrayPermute(u64, ndims, shape, perm));
        }
        /// Permute the dimensions of the tensor. A valid permutation must contain
        /// values from 0 to ndims and each value must appear exactly once.
        pub fn permute(a: Self, comptime perm: [ndims]u8) Permute(perm) {
            const Out = Permute(perm);
            return a.asStrided(
                Out.shape,
                utils.arrayPermute(u64, ndims, a.strides, perm),
                a.offset,
            );
        }

        pub fn Transpose(comptime dim1: i16, comptime dim2: i16) type {
            const norm1 = normalizedDim(dim1);
            const norm2 = normalizedDim(dim2);
            var new_shape = shape;
            new_shape[norm1] = shape[norm2];
            new_shape[norm2] = shape[norm1];
            return View(new_shape);
        }
        /// Transpose two dimensions of the tensor. Similar to permute, but only for two dimensions.
        pub fn transpose(a: Self, comptime dim1: i16, comptime dim2: i16) Transpose(dim1, dim2) {
            const norm1 = normalizedDim(dim1);
            const norm2 = normalizedDim(dim2);
            if (norm1 != norm2) {
                const Out = Transpose(norm1, norm2);
                var new_strides = a.strides;
                new_strides[norm1] = a.strides[norm2];
                new_strides[norm2] = a.strides[norm1];
                return a.asStrided(
                    Out.shape,
                    new_strides,
                    a.offset,
                );
            } else {
                return a;
            }
        }

        pub fn View(comptime new_shape: anytype) type {
            return UserTensor(dtype, new_shape);
        }
        /// View the tensor as a different shape.
        pub fn view(a: Self, comptime new_shape: anytype) View(new_shape) {
            if (!isContiguous(a)) {
                @compileError("Cannot change view of a non-contiguous tensor. Copy the tensor first to make it contiguous.");
            }
            return View(new_shape).initContiguous(.{ .TypeOp = .{ .op = .AsStrided, .a = &a.any() } });
        }

        pub fn flatten(comptime a: Self) View(.{a.size() - a.offset}) {
            return a.view(.{a.size() - a.offset});
        }

        fn Unsqueeze(comptime dim: i16) type {
            return View(utils.arrayInsert(ndims, shape, normalizedDim(dim), 1));
        }
        /// Insert a dim of size 1 into the shape of the tensor.
        pub fn unsqueeze(a: Self, comptime dim: i16) Unsqueeze(dim) {
            return a.asStrided(
                Unsqueeze(dim).shape,
                utils.arrayInsert(ndims, a.strides, normalizedDim(dim), 0),
                a.offset,
            );
        }

        fn Squeeze(comptime dim: i16) type {
            if (shape[normalizedDim(dim)] != 1) {
                @compileError(
                    \\[TESSERACT COMPILE ERROR]
                    \\Cannot squeeze as dimension size is not 1 or stride for dimension is not 0
                );
            }
            return View(utils.arrayDelete(ndims, shape, normalizedDim(dim)));
        }
        /// Remove a dim of size 1 from the shape of the tensor.
        pub fn squeeze(a: Self, comptime dim: i16) Squeeze(dim) {
            return a.asStrided(
                Squeeze(dim).shape,
                utils.arrayDelete(ndims, a.strides, normalizedDim(dim)),
                a.offset,
            );
        }

        /// Changes the shape and stride of the tensor to change how the underlying memory is accessed.
        /// Powerful enough to be used to implement any reshaping or windowing operation on a tensor.
        /// There are guiderails to prevent out of bounds access into underlying memory.
        pub fn asStrided(comptime a: Self, comptime new_shape: anytype, comptime new_strides: [new_shape.len]u64, offset: u64) View(new_shape) {
            var out = View(new_shape){
                .last_op = .{ .TypeOp = .{ .op = .AsStrided, .a = &a.any() } },
                .strides = new_strides,
                .offset = offset,
            };
            if (out.size() > a.size()) {
                @compileError(
                    \\[TESSERACT COMPILE ERROR]
                    \\New stride pattern will go out of bounds of the current tensor's underlying memory
                );
            }

            return out;
        }

        ///Cast an array of a datatype to another datatype
        pub fn asType(comptime a: Self, comptime new_dtype: dtypes.DType) UserTensor(new_dtype, shape) {
            return UserTensor(new_dtype, shape).initContiguous(.{ .TypeOp = .{ .op = .AsType, .a = &a.any() } });
        }

        ///Apply an elementwise map operation
        pub fn map(comptime a: Self, comptime op: ops.MapOp) Self {
            return initContiguous(.{ .MapOp = .{ .op = op, .a = &a.any() } });
        }

        pub fn Broadcast(comptime new_shape: anytype) type {
            if (std.mem.eql(u64, &shape, &new_shape)) {
                return Self;
            }
            const bc_ndims = @max(ndims, new_shape.len);
            var bc_shape: [bc_ndims]u64 = undefined;
            for (0..bc_ndims) |i| {
                const dim1 = if (i >= ndims) 1 else shape[ndims - i - 1];
                const dim2 = if (i >= new_shape.len) 1 else new_shape[new_shape.len - i - 1]; // orelse dim1;
                if (dim1 != 1 and dim2 != 1 and dim1 != dim2) {
                    @compileError(comptimePrint(
                        "Cannot broadcast tensors of shapes {any} and {any}",
                        .{ shape, new_shape },
                    ));
                }
                bc_shape[bc_ndims - i - 1] = if (dim1 == dim2 or dim2 == 1) dim1 else dim2;
            }
            return View(bc_shape);
        }
        pub fn expand(comptime a: Self, comptime new_shape: anytype) Broadcast(new_shape) {
            const Out = Broadcast(new_shape);
            if (Self == Out) {
                return a;
            }
            return Out.initContiguous(.{ .TypeOp = .{ .op = .AsStrided, .a = &a.any() } });
        }

        pub fn zipOpResultDType(comptime op: ops.ZipOp, a: Self, b: anytype) dtypes.DType {
            return switch (op) {
                .Equals, .LessThan => .bool,
                else => dtypes.resultDType(a.dtype, asTensor(b).dtype),
            };
        }

        pub fn normalizedDim(dim: i16) u8 {
            const normalized = if (dim < 0) ndims - dim else dim;
            if (normalized < 0 or normalized > ndims) {
                @compileError(comptimePrint("Dimension index {d} is out of bounds {d}", .{ normalized, ndims }));
            }
            return @intCast(normalized);
        }

        /// Apply an elementwise zip (binary) operation on two arrays, with broadcasting
        pub fn zip(comptime a: Self, comptime op: ops.ZipOp, comptime b: anytype) UserTensor(zipOpResultDType(op, a, asTensor(b)), Broadcast(asTensor(b).shape).shape) {
            const b_tensor = comptime asTensor(b);
            const bc_shape = Broadcast(b_tensor.shape).shape;
            const new_dtype = comptime zipOpResultDType(op, a, b);

            // Expand a and b to match the output shape
            const a_expand = comptime a.expand(bc_shape);
            const b_expand = comptime b_tensor.expand(bc_shape);
            const Out = UserTensor(new_dtype, bc_shape);
            return Out.initContiguous(.{ .ZipOp = .{ .op = op, .a = &a_expand.any(), .b = &b_expand.any() } });
        }

        pub fn Reduce(comptime reduce_dims: anytype) type {
            switch (@typeInfo(@TypeOf(reduce_dims))) {
                .ComptimeInt, .Int => {
                    const dim = normalizedDim(reduce_dims);
                    if (dim < 0 or dim >= ndims) {
                        @compileError("Dimension index for single dimension reduce is out of bounds");
                    }
                    var reduced_shape: [ndims]u64 = undefined;
                    @memcpy(&reduced_shape, &shape);
                    reduced_shape[dim] = 1;
                    return View(reduced_shape);
                },
                .Null, .Void => {
                    return View([_]u64{1} ** ndims);
                },
                else => {
                    const dims = reduce_dims;
                    if (dims.len > ndims) {
                        @compileError("Length of dimension index array for multi dimension reduce is out of bounds");
                    }
                    var reduce_dim_mask: [ndims]bool = [_]bool{false} ** ndims;
                    var reduced_shape: [ndims]u64 = undefined;
                    @memcpy(&reduced_shape, &shape);
                    for (0..dims.len) |d| {
                        const norm = normalizedDim(d);
                        if (reduce_dim_mask[norm]) {
                            @compileError("Cannot reuse dimension index for multi dimensional reduce");
                        }
                        reduce_dim_mask[d] = true;
                        reduced_shape[d] = 1;
                    }
                    return View(reduced_shape);
                },
            }
        }
        /// Perform a reduction across 1 or more (or all) dimensions of a tensor.
        /// Dimensions to reduce can be passed as a int for 1 dim, tuple for multiple dims, or null/void for all dims
        pub fn reduce(comptime a: Self, comptime op: ops.ReduceOp, comptime reduce_dims: anytype) Reduce(reduce_dims) {
            const Out = Reduce(reduce_dims);
            const reduction_dim_mask: [ndims]bool = switch (@typeInfo(@TypeOf(reduce_dims))) {
                .ComptimeInt, .Int => blk: {
                    var tmp_mask: [ndims]bool = [_]bool{false} ** ndims;
                    const dim = reduce_dims;
                    tmp_mask[normalizedDim(dim)] = true;
                    break :blk tmp_mask;
                },
                .Null, .Void => [_]bool{true} ** ndims,
                else => blk: {
                    var tmp_mask: [ndims]bool = [_]bool{false} ** ndims;
                    for (reduce_dims) |dim| {
                        tmp_mask[normalizedDim(dim)] = true;
                    }
                    break :blk tmp_mask;
                },
            };
            return Out.initContiguous(.{ .ReduceOp = .{ .op = op, .a = &a.any(), .dims = &reduction_dim_mask } });
        }

        pub fn MatMul(comptime other: anytype) type {
            // Matrix multiplication invariant
            // (n a m1) matmul (m2 a p) -> (n a p) iff m1 = m2
            // otherwise matmul is invalid, compile error
            const Other = @TypeOf(other);
            const n = if (ndims == 1) 1 else shape[ndims - 2];
            const m = shape[ndims - 1];
            const other_m = if (Other.ndims == 1) 1 else Other.shape[Other.ndims - 2];
            const p = Other.shape[Other.ndims - 1];

            if (m == other_m) {
                const mm_ndims = @max(ndims, Other.ndims);
                var mm_shape: [mm_ndims]u64 = undefined;
                // Broadcasting check, look only at batch dimensions (everything before last 2 dimensions)
                for (0..mm_ndims - 2) |i| {
                    const dim1 = if (i >= ndims - 2) 1 else shape[ndims - i - 3];
                    const dim2 = if (i >= Other.ndims - 2) 1 else Other.shape[Other.ndims - i - 3];
                    if (dim1 == dim2 or dim1 == 1 or dim2 == 1) {
                        mm_shape[mm_ndims - i - 3] = if (dim1 == dim2 or dim2 == 1) dim1 else dim2;
                    } else {
                        @compileError(comptimePrint(
                            \\[TESSERACT COMPILE ERROR]
                            \\Cannot perform matrix multiplication on these two tensors
                            \\tensor1: {any}
                            \\tensor2: {any}
                        , .{ shape, Other.shape }));
                    }
                }
                mm_shape[mm_ndims - 2] = n;
                mm_shape[mm_ndims - 1] = p;
                return View(mm_shape);
            }
            @compileError(comptimePrint(
                \\[TESSERACT COMPILE ERROR]
                \\Cannot perform matrix multiplication on these two tensors
                \\tensor1: {any}
                \\tensor2: {any}
            , .{ shape, Other.shape }));
        }
        pub fn matmul(comptime a: Self, comptime b: anytype) MatMul(b) {
            return a
                .unsqueeze(a.ndims - 1)
                .mul(b.transpose(b.ndims - 2, b.ndims - 1).copy().unsqueeze(b.ndims - 2))
                .sum(a.ndims)
                .squeeze(a.ndims);
        }

        pub fn Where(comptime true_value: anytype, comptime false_value: anytype) type {
            const true_tensor = asTensor(true_value);
            const false_tensor = asTensor(false_value);
            const T = @TypeOf(true_tensor);
            const F = @TypeOf(false_tensor);
            std.debug.assert(T.dtype == F.dtype);
            std.debug.assert(dtypes.isBool(Self.dtype));
            const TF = T.Broadcast(F.shape);
            return UserTensor(TF.dtype, Broadcast(TF.shape).shape);
        }
        /// Conditional elementwise operator
        /// out[i] = if (mask[i]) true_value[i] else false_value[i]
        /// Supports broadcasting between all 3 tensors, but true value and false value are broadcasted together first and must also have the same dtype
        pub fn where(comptime mask: Self, comptime true_value: anytype, comptime false_value: anytype) Where(true_value, false_value) {
            const Out = Where(true_value, false_value);
            const mask_expand = mask.expand(Out.shape);
            const true_expand = asTensor(true_value).expand(Out.shape);
            const false_expand = asTensor(false_value).expand(Out.shape);
            return Out.initContiguous(.{ .TernaryOp = .{
                .op = .Where,
                .a = &mask_expand.any(),
                .b = &true_expand.any(),
                .c = &false_expand.any(),
            } });
        }

        // TODO: Need to implement padding to get conv2d to work
        // Might want to practice with Conv1d first
        // pub fn Conv2d(comptime Filter: type, _stride: anytype, _) type {
        //     const stride: [2]u64 = switch (@typeInfo(@TypeOf(_stride))) {
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
    };
}

test "same tensors assignable" {
    // This test catches regressions caused by comptime slices with the same values not being
    // equal to teach other, which would cause this test to not compile
    // Note that the fill value is different: this should have no effect
    comptime {
        const tensor1 = UserTensor(.i32, .{ 2, 3, 4 }).full(0);
        var tensor2 = UserTensor(.i32, .{ 2, 3, 4 }).full(1);
        tensor2 = tensor1;
    }
}

test "permute" {
    comptime {
        const tensor1 = UserTensor(.i32, .{ 2, 3, 4 }).full(0);
        const tensor2 = tensor1.permute(.{ 0, 2, 1 });
        try std.testing.expectEqual([_]u64{ 2, 4, 3 }, tensor2.shape);
        try std.testing.expectEqual([_]u64{ 12, 1, 4 }, tensor2.strides);
    }
}

test "view" {
    comptime {
        const tensor1 = UserTensor(.i32, .{ 2, 3, 4 }).full(0);
        const tensor2 = tensor1.view(.{ 12, 2 });
        const tensor3 = tensor2.view(.{24});
        try std.testing.expectEqual([_]u64{ 12, 2 }, tensor2.shape);
        try std.testing.expectEqual([_]u64{ 2, 1 }, tensor2.strides);
        try std.testing.expectEqual([_]u64{24}, tensor3.shape);
        try std.testing.expectEqual([_]u64{1}, tensor3.strides);
    }
}

test "as strided" {
    // Based on example from https://pytorch.org/docs/stable/generated/torch.as_strided.html
    comptime {
        const tensor1 = UserTensor(.i32, .{ 3, 3 }).full(0);
        const tensor2 = tensor1.asStrided(.{ 2, 2 }, .{ 1, 2 }, 0);

        try std.testing.expectEqual([_]u64{ 2, 2 }, tensor2.shape);
        try std.testing.expectEqual(false, tensor2.isContiguous());

        const test_indices = [_][2]u64{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 } };
        const expected_flat_indices1 = [_]u64{ 0, 2, 1, 3 };
        for (expected_flat_indices1, test_indices) |expected_flat_i, test_i| {
            try std.testing.expectEqual(expected_flat_i, utils.ravelMultiIndex(tensor2.ndims, tensor2.strides, tensor2.offset, test_i));
        }

        const tensor3 = tensor1.asStrided(.{ 2, 2 }, .{ 1, 2 }, 1);
        try std.testing.expectEqual([_]u64{ 2, 2 }, tensor2.shape);
        try std.testing.expectEqual(false, tensor2.isContiguous());

        const expected_flat_indices2 = [_]u64{ 1, 3, 2, 4 };
        for (expected_flat_indices2, test_indices) |expected_flat_i, test_i| {
            try std.testing.expectEqual(expected_flat_i, utils.ravelMultiIndex(tensor3.ndims, tensor3.strides, tensor3.offset, test_i));
        }
    }
}

test "map" {
    const tensor1 = comptime UserTensor(.i32, .{ 2, 3, 4 }).full(3);
    const tensor2 = comptime tensor1.neg();
    try std.testing.expectEqual([_]u64{ 2, 3, 4 }, tensor2.shape);
    Graph.init();
    defer Graph.deinit();
    tensor2.trace();
    try std.testing.expect(tensor2.last_op.MapOp.op == .Neg);
    try std.testing.expectEqual(tensor2.last_op.MapOp.a.*, tensor1.any());
}

test "zip" {
    const tensor1 = comptime UserTensor(.i32, .{ 2, 1, 4 }).full(2);
    const tensor2 = comptime UserTensor(.i32, .{ 3, 1 }).full(3);
    const tensor3 = comptime tensor1.add(tensor2);
    try std.testing.expectEqual([_]u64{ 2, 3, 4 }, tensor3.shape);
    Graph.init();
    defer Graph.deinit();
    tensor3.trace();
    try std.testing.expect(tensor3.last_op.ZipOp.op == .Add);
    try std.testing.expectEqual(tensor3.last_op.ZipOp.a.last_op.TypeOp.a.*, tensor1.any());
    try std.testing.expectEqual(tensor3.last_op.ZipOp.b.last_op.TypeOp.a.*, tensor2.any());
}

test "reduce" {
    const tensor1 = comptime UserTensor(.i32, .{ 2, 3, 4 }).full(5);
    const tensor2 = comptime tensor1.sum(1);
    try std.testing.expectEqual([_]u64{ 2, 1, 4 }, tensor2.shape);
    Graph.init();
    defer Graph.deinit();
    tensor2.trace();
    try std.testing.expect(tensor2.last_op.ReduceOp.op == .Sum);
    try std.testing.expectEqual(tensor2.last_op.ReduceOp.a.*, tensor1.any());
    try std.testing.expectEqual(tensor2.last_op.ReduceOp.dims[0..tensor2.ndims].*, ([_]bool{ false, true, false }));
}

test "multiple dim reduce" {
    const tensor1 = comptime UserTensor(.i32, .{ 2, 3, 4 }).full(5);
    const tensor2 = comptime tensor1.sum(.{ 0, 1 });
    try std.testing.expectEqual([_]u64{ 1, 1, 4 }, tensor2.shape);
    Graph.init();
    defer Graph.deinit();
    tensor2.trace();
    try std.testing.expect(tensor2.last_op.ReduceOp.op == .Sum);
    try std.testing.expectEqual(tensor2.last_op.ReduceOp.a.*, tensor1.any());
    try std.testing.expectEqual(tensor2.last_op.ReduceOp.dims[0..tensor2.ndims].*, [_]bool{ true, true, false });
}

test "zip reduce" {
    const tensor1 = comptime UserTensor(.i32, .{ 2, 1, 4 }).full(2);
    const tensor2 = comptime UserTensor(.i32, .{ 2, 3, 1 }).full(3);
    const tensor3 = comptime tensor1.add(tensor2).sum(1);
    try std.testing.expectEqual([_]u64{ 2, 1, 4 }, tensor3.shape);
    Graph.init();
    defer Graph.deinit();
    tensor3.trace();
    try std.testing.expect(tensor3.last_op.ReduceOp.op == .Sum);
    // Anonymous intermediate tensor that stores tensor1 + tensor2
    const anon = tensor3.last_op.ReduceOp.a;
    try std.testing.expectEqual(anon.last_op.ZipOp.a.last_op.TypeOp.a.*, tensor1.any());
    try std.testing.expectEqual(anon.last_op.ZipOp.b.last_op.TypeOp.a.*, tensor2.any());
}

test "as_type" {
    const tensor1 = comptime UserTensor(.bool, .{3}).full(true);
    try std.testing.expect(tensor1.dtype == .bool);
    const tensor2 = comptime tensor1.asType(.i32);
    try std.testing.expect(tensor2.dtype == .i32);
    const tensor3 = comptime tensor2.asType(.i8);
    try std.testing.expect(tensor3.dtype == .i8);
    const tensor4 = comptime tensor3.asType(.f16);
    try std.testing.expect(tensor4.dtype == .f16);
    const tensor5 = comptime tensor4.asType(.f32);
    try std.testing.expect(tensor5.dtype == .f32);
}

fn fn1() UserTensor(.i32, .{ 2, 1, 4 }) {
    const tensor1 = UserTensor(.i32, .{ 2, 1, 4 }).full(1);
    const tensor2 = UserTensor(.i32, .{ 2, 3, 1 }).full(2);
    const tensor3 = tensor1.add(tensor2).sum(1);
    return tensor3;
}

fn fn2(input: anytype) UserTensor(.i32, .{ 2, 3, 4 }) {
    return comptime blk: {
        const tensor4 = UserTensor(.i32, .{ 2, 1, 4 }).full(4);
        const tensor5 = UserTensor(.i32, .{ 2, 3, 1 }).full(5);
        const tensor6 = tensor4.mul(input).sum(1).add(tensor5);
        break :blk tensor6;
    };
}

test "tensors from functions" {
    const out = comptime blk: {
        const tensor3 = fn1();
        const tensor6 = fn2(tensor3);
        break :blk tensor6;
    };

    Graph.init();
    defer Graph.deinit();
    out.trace();
    // Graph.viz();
}
