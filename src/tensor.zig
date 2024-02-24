const std = @import("std");
const Allocator = std.mem.Allocator;
const comptimePrint = std.fmt.comptimePrint;
const utils = @import("utils.zig");
const ops = @import("ops.zig");
const Graph = @import("Graph.zig");
const dtypes = @import("dtypes.zig");

pub fn constant(comptime dtype: dtypes.DType, comptime value: anytype) InferredStrides(dtype, .{1}) {
    return InferredStrides(dtype, .{1}).full(value);
}

pub fn range(
    comptime dtype: dtypes.DType,
    comptime start: if (dtypes.isInt(dtype)) comptime_int else @compileError("Range tensor must have int dtype"),
    comptime stop: if (dtypes.isInt(dtype)) comptime_int else @compileError("Range tensor must have int dtype"),
) InferredStrides(dtype, .{stop - start}) {
    return InferredStrides(dtype, .{stop - start}).rand(start, stop);
}

pub fn InferredStrides(comptime dtype: dtypes.DType, comptime shape: anytype) type {
    const ndims = shape.len;
    var offset: usize = 1;
    var strides: [ndims + 1]usize = undefined;
    for (0..ndims - 1) |d| {
        const stride = shape[ndims - d - 1] * offset;
        strides[ndims - d - 2] = stride;
        offset = stride;
    }
    strides[ndims - 1] = 1;
    strides[ndims] = 0;
    for (0..ndims) |d| {
        if (shape[d] == 0 or shape[d] == 1) {
            strides[d] = 0;
        }
    }
    return Tensor(dtype, shape.len, shape, strides);
}

fn Tensor(
    // These generic parameters are private so they will be redeclare as pub conts in the result type
    comptime tensor_dtype: dtypes.DType,
    comptime tensor_ndims: u8,
    comptime tensor_shape: [tensor_ndims]usize,
    comptime tensor_strides: [tensor_ndims + 1]usize,
) type {
    return struct {
        // All the functions for operations are implemented separately
        pub usingnamespace @import("functions.zig");
        const Self = @This();

        // Type level constants for comptime logic (e.g. @TypeOf(x).ndims)
        pub const dtype: dtypes.DType = tensor_dtype;
        pub const ndims: u8 = tensor_ndims;
        pub const shape: [ndims]usize = tensor_shape;
        pub const strides: [ndims + 1]usize = tensor_strides;
        pub const size = get_size: {
            // The storage size is 1 + last index calculated by the strides and shape
            // shape[d] - 1 is the last index in dimension d
            // Also incorporate the storage offset
            var _size: usize = strides[ndims] + 1;
            for (0..ndims) |d| {
                _size += (shape[d] - 1) * strides[d];
            }
            // The result is the size of the storage needed to visit all indices of the tensor
            break :get_size _size;
        };
        pub const is_contiguous: bool = is_contiguous: {
            var prev: usize = (1 << @typeInfo(usize).Int.bits) - 1;
            for (strides[0..ndims]) |s| {
                if (s > prev and s > 0) {
                    break :is_contiguous false;
                }
                if (s > 0) {
                    prev = s;
                }
            }
            break :is_contiguous true;
        };

        ndims: u8 = ndims,
        dtype: dtypes.DType = dtype,
        shape: [ndims]usize = shape,
        size: usize = size,
        strides: [ndims + 1]usize = strides,
        is_contiguous: bool = is_contiguous,

        traceFn: *const fn (self: *const Self) void,

        pub fn init(comptime traceFn: *const fn (self: *const Self) void) Self {
            return .{ .traceFn = traceFn };
        }

        pub fn input() Self {
            const traceFn = struct {
                fn trace(self: *const Self) void {
                    Graph.vertex(self, .{
                        .InitOp = .{ .op = .Input, .value = .{ .Input = {} } },
                    }, Self) catch unreachable;
                }
            }.trace;
            return init(traceFn);
        }

        /// Fill a tensor with a value
        pub fn full(comptime value: anytype) Self {
            const traceFn = struct {
                fn trace(self: *const Self) void {
                    Graph.vertex(self, .{
                        .InitOp = .{ .op = .Full, .value = .{ .Full = std.fmt.comptimePrint("{any}", .{value}) } },
                    }, Self) catch unreachable;
                }
            }.trace;
            return init(traceFn);
        }
        pub fn fullLike(_: *const Self, comptime value: anytype) Self {
            return Self.full(value);
        }

        /// Internal function to fill with range, this is not publicly exposed
        /// as shape of range tensor must be constrained
        fn range(comptime start: comptime_int, comptime stop: comptime_int) Self {
            if (ndims > 1) {
                @compileError("Cannot use range() on a tensor with > 1 dimensions");
            }
            const traceFn = struct {
                fn trace(self: *const Self) void {
                    Graph.vertex(self, .{ .InitOp = .{ .op = .Range, .value = .{ .Range = .{
                        .start = std.fmt.comptimePrint("{d}", .{start}),
                        .stop = std.fmt.comptimePrint("{d}", .{stop}),
                    } } } }, Self) catch unreachable;
                }
            }.trace;
            return init(traceFn);
        }

        pub fn rand() Self {
            const traceFn = struct {
                fn trace(self: *const Self) void {
                    Graph.vertex(self, .{ .InitOp = .{ .op = .Rand, .value = .{ .Rand = dtype } } }, Self) catch unreachable;
                }
            }.trace;
            return init(traceFn);
        }
        pub fn randLike(_: *const Self) Self {
            return Self.rand();
        }

        /// A copy is only needed to make a non-contiguous tensor contiguous again.
        /// Each tensor is immutable and operations already produce new tensors
        /// but intermediate tensors can be eliminated through optimization.
        pub fn copy(x: *const Self) InferredStrides(dtype, shape) {
            const Out = InferredStrides(dtype, shape);
            const traceFn = struct {
                fn trace(out: *const Out) void {
                    Graph.trace(x);
                    Graph.vertex(out, .{ .MapOp = .{
                        .op = .Copy,
                        .x = Graph.vertexOf(x),
                    } }, Out) catch unreachable;
                }
            }.trace;
            return Out.init(traceFn);
        }

        fn Permute(comptime perm: [ndims]u8) type {
            var strides_perm: [ndims + 1]u8 = undefined;
            @memcpy(strides_perm[0..ndims], &perm);
            strides_perm[ndims] = ndims;
            return AsStrided(
                utils.arrayPermute(ndims, shape, perm),
                utils.arrayPermute(ndims + 1, strides, strides_perm),
            );
        }
        /// Permute the dimensions of the tensor. A valid permutation must contain
        /// values from 0 to ndims and each value must appear exactly once.
        pub fn permute(x: *const Self, comptime perm: [ndims]u8) Permute(perm) {
            const Out = Permute(perm);
            return x.asStrided(Out.shape, Out.strides);
        }

        pub fn Transpose(comptime dim1: u8, comptime dim2: u8) type {
            if (dim1 == dim2) {
                return Self;
            } else {
                var new_shape = shape;
                new_shape[dim1] = shape[dim2];
                new_shape[dim2] = shape[dim1];
                var new_strides = strides;
                new_strides[dim1] = strides[dim2];
                new_strides[dim2] = strides[dim1];
                return AsStrided(new_shape, new_strides);
            }
        }
        /// Transpose two dimensions of the tensor. Similar to permute, but only for two dimensions.
        pub fn transpose(x: *const Self, comptime dim1: u8, comptime dim2: u8) Transpose(dim1, dim2) {
            if (dim1 != dim2) {
                const Out = Transpose(dim1, dim2);
                return x.asStrided(Out.shape, Out.strides);
            } else {
                return x.*;
            }
        }
        /// View the tensor as a different shape.
        pub fn view(x: *const Self, comptime new_shape: anytype) InferredStrides(dtype, new_shape) {
            const Out = InferredStrides(dtype, new_shape);
            const traceFn = struct {
                fn trace(out: *const Out) void {
                    Graph.trace(x);
                    Graph.vertex(out, .{ .TypeOp = .{
                        .op = .View,
                        .x = Graph.vertexOf(x),
                    } }, Out) catch unreachable;
                }
            }.trace;
            return Out.init(traceFn);
        }

        fn Unsqueeze(comptime dim: u8) type {
            if (dim > ndims) {
                @compileError("dim to unsqueeze at is out of range");
            }
            return AsStrided(
                utils.arrayInsert(ndims, shape, dim, 1),
                utils.arrayInsert(ndims + 1, strides, dim, 0),
            );
        }
        /// Remove a dim of size 1 from the shape of the tensor.
        pub fn unsqueeze(x: *const Self, comptime dim: u8) Unsqueeze(dim) {
            const Out = Unsqueeze(dim);
            return x.asStrided(Out.shape, Out.strides);
        }

        fn Squeeze(comptime dim: u8) type {
            if (dim >= ndims) {
                @compileError("dim to squeeze at is out of range");
            }
            if (shape[dim] != 1 or strides[dim] != 0) {
                @compileError(
                    \\[TESSERACT COMPILE ERROR]
                    \\Cannot squeeze as dimension size is not 1 or stride for dimension is not 0
                );
            }
            return AsStrided(
                utils.arrayDelete(ndims, shape, dim),
                utils.arrayDelete(ndims + 1, strides, dim),
            );
        }
        /// Insert a dim of size 1 into the shape of the tensor.
        pub fn squeeze(x: *const Self, comptime dim: u8) Squeeze(dim) {
            const Out = Squeeze(dim);
            return x.asStrided(Out.shape, Out.strides);
        }

        fn AsStrided(comptime new_shape: anytype, comptime new_strides: anytype) type {
            if (new_shape.len + 1 != new_strides.len) {
                @compileError(
                    \\[TESSERACT COMPILE ERROR]
                    \\Provided shape ndims not compatible with provided strides ndims
                    \\You may be missing the storage offset (strides[ndims])
                );
            }
            const Out = Tensor(dtype, new_shape.len, new_shape, new_strides);
            if (Out.size > Self.size) {
                @compileError(
                    \\[TESSERACT COMPILE ERROR]
                    \\Provided strides will go out of bounds of the current tensor's underlying memory
                );
            }
            return Out;
        }
        /// Changes the shape and stride of the tensor to change how the underlying memory is accessed.
        /// Powerful enough to be used to implement any reshaping or windowing operation on a tensor.
        /// There are guiderails to prevent out of bounds access into underlying memory.
        pub fn asStrided(comptime x: *const Self, comptime new_shape: anytype, comptime new_strides: anytype) AsStrided(new_shape, new_strides) {
            const Out = AsStrided(new_shape, new_strides);
            const traceFn = struct {
                fn trace(out: *const Out) void {
                    Graph.trace(x);
                    Graph.vertex(out, .{ .TypeOp = .{
                        .op = .AsStrided,
                        .x = Graph.vertexOf(x),
                    } }, Out) catch unreachable;
                }
            }.trace;
            return Out.init(traceFn);
        }

        ///Cast an array of a datatype to another datatype
        pub fn asType(comptime x: *const Self, comptime new_dtype: dtypes.DType) Tensor(new_dtype, ndims, shape, strides) {
            const Out: type = Tensor(new_dtype, ndims, shape, strides);
            const traceFn = struct {
                fn trace(out: *const Out) void {
                    Graph.trace(x);
                    Graph.vertex(out, .{ .TypeOp = .{
                        .op = .AsType,
                        .x = Graph.vertexOf(x),
                    } }, Out) catch unreachable;
                }
            }.trace;
            return Out.init(traceFn);
        }

        ///Apply an elementwise map operation
        pub fn map(comptime x: *const Self, comptime op: ops.MapOp) Self {
            const Out: type = @TypeOf(x.*);
            const traceFn = struct {
                fn trace(out: *const Out) void {
                    Graph.trace(x);
                    Graph.vertex(out, .{ .MapOp = .{
                        .op = op,
                        .x = Graph.vertexOf(x),
                    } }, Out) catch unreachable;
                }
            }.trace;
            return Out.init(traceFn);
        }

        pub fn Broadcast(comptime new_shape: anytype, comptime new_dtype: dtypes.DType) type {
            if (std.mem.eql(usize, &shape, &new_shape)) {
                return Self;
            }
            const bc_ndims = @max(ndims, new_shape.len);
            var bc_shape: [bc_ndims]usize = undefined;
            for (0..bc_ndims) |i| {
                const dim1 = if (i >= ndims) 1 else shape[ndims - i - 1];
                const dim2 = if (i >= new_shape.len) 1 else new_shape[new_shape.len - i - 1];
                if (dim1 != 1 and dim2 != 1 and dim1 != dim2) {
                    @compileError(comptimePrint(
                        "Cannot broadcast tensors of shapes {any} and {any}",
                        .{ shape, new_shape },
                    ));
                }
                bc_shape[bc_ndims - i - 1] = if (dim1 == dim2 or dim2 == 1) dim1 else dim2;
            }
            return InferredStrides(new_dtype, bc_shape);
        }
        pub fn expand(comptime x: *const Self, comptime new_shape: anytype) Broadcast(new_shape, dtype) {
            const Out: type = Broadcast(new_shape, dtype);
            if (Self == Out) {
                return x.*;
            }
            const traceFn = struct {
                fn trace(out: *const Out) void {
                    Graph.trace(x);
                    Graph.vertex(out, .{
                        .TypeOp = .{
                            .op = .Broadcast,
                            .x = Graph.vertexOf(x),
                        },
                    }, Out) catch unreachable;
                }
            }.trace;
            return Out.init(traceFn);
        }

        /// Apply an elementwise zip (binary) operation on two arrays, with broadcasting
        pub fn zip(comptime a: *const Self, comptime op: ops.ZipOp, comptime b: anytype) Broadcast(
            b.shape,
            switch (op) {
                .Equals, .LessThan => .bool,
                else => dtype,
            },
        ) {
            const Out: type = Broadcast(
                b.shape,
                switch (op) {
                    .Equals, .LessThan => .bool,
                    else => dtype,
                },
            );

            const a_expand = comptime a.expand(Out.shape);
            const b_expand = comptime b.expand(Out.shape);
            const traceFn = struct {
                fn trace(out: *const Out) void {
                    Graph.trace(&a_expand);
                    Graph.trace(&b_expand);
                    Graph.vertex(out, .{
                        .ZipOp = .{
                            .op = op,
                            .a = Graph.vertexOf(&a_expand),
                            .b = Graph.vertexOf(&b_expand),
                        },
                    }, Out) catch unreachable;
                }
            }.trace;
            return Out.init(traceFn);
        }

        pub fn Reduce(comptime reduce_dims: anytype) type {
            switch (@typeInfo(@TypeOf(reduce_dims))) {
                .ComptimeInt, .Int => {
                    const dim = reduce_dims;
                    if (dim < 0 or dim >= ndims) {
                        @compileError("Dimension index for single dimension reduce is out of bounds");
                    }
                    var reduced_shape: [ndims]usize = undefined;
                    @memcpy(&reduced_shape, &shape);
                    reduced_shape[dim] = 1;
                    return InferredStrides(dtype, reduced_shape);
                },
                .Null, .Void => {
                    return InferredStrides(dtype, [_]usize{1} ** ndims);
                },
                else => {
                    const dims = reduce_dims;
                    if (dims.len > ndims) {
                        @compileError("Length of dimension index array for multi dimension reduce is out of bounds");
                    }
                    var reduced: [ndims]bool = [_]bool{false} ** ndims;
                    var reduced_shape: [ndims]usize = undefined;
                    @memcpy(&reduced_shape, &shape);
                    for (0..dims.len) |d| {
                        if (d < 0 or d >= ndims) {
                            @compileError("Dimension index for multi dimension reduce is out of bounds");
                        }
                        if (reduced[d]) {
                            @compileError("Cannot reuse dimension index for multi dimensional reduce");
                        }
                        reduced[d] = true;
                        reduced_shape[d] = 1;
                    }
                    return InferredStrides(dtype, reduced_shape);
                },
            }
        }
        /// Perform a reduction across 1 or more (or all) dimensions of a tensor.
        /// Dimensions to reduce can be passed as a int for 1 dim, tuple for multiple dims, or null/void for all dims
        pub fn reduce(comptime x: *const Self, comptime op: ops.ReduceOp, comptime reduce_dims: anytype) Reduce(reduce_dims) {
            const Out: type = Reduce(reduce_dims);
            const reduction_dim_mask: [ndims]bool = switch (@typeInfo(@TypeOf(reduce_dims))) {
                .ComptimeInt, .Int => blk: {
                    var tmp_mask: [ndims]bool = [_]bool{false} ** ndims;
                    tmp_mask[reduce_dims] = true;
                    break :blk tmp_mask;
                },
                .Null, .Void => [_]bool{true} ** ndims,
                else => blk: {
                    var tmp_mask: [ndims]bool = [_]bool{false} ** ndims;
                    for (reduce_dims) |dim| {
                        tmp_mask[dim] = true;
                    }
                    break :blk tmp_mask;
                },
            };
            const traceFn = struct {
                fn trace(out: *const Out) void {
                    Graph.trace(x);
                    Graph.vertex(out, .{ .ReduceOp = .{
                        .op = op,
                        .x = Graph.vertexOf(x),
                        .dims = reduction_dim_mask[0..],
                    } }, Out) catch unreachable;
                }
            }.trace;
            return Out.init(traceFn);
        }

        pub fn MatMul(comptime Other: type) type {
            // Matrix multiplication invariant
            // (n x m1) matmul (m2 x p) -> (n x p) if m1 = m2
            // If m1 != m2 then matmul is invalid
            const n = if (ndims == 1) 1 else shape[ndims - 2];
            const m = shape[ndims - 1];
            const other_m = if (Other.ndims == 1) 1 else Other.shape[Other.ndims - 2];
            const p = Other.shape[Other.ndims - 1];

            if (m == other_m) {
                const mm_ndims = @max(ndims, Other.ndims);
                var mm_shape: [mm_ndims]usize = undefined;
                // Broadcasting check, look only at batch dimensions (everything before last 2 dimensions)
                for (0..mm_ndims - 2) |i| {
                    const dim1 = if (i >= ndims - 2) 1 else shape[ndims - i - 3];
                    const dim2 = if (i >= Other.ndims - 2) 1 else Other.shape[Other.ndims - i - 3];
                    if (dim1 == dim2 or dim1 == 1 or dim2 == 1) {
                        mm_shape[mm_ndims - i - 3] = if (dim1 == dim2 or dim2 == 1) dim1 else dim2;
                        @compileError(comptimePrint(
                            \\[TESSERACT COMPILE ERROR]
                            \\Cannot perform matrix multiplication on these two tensors
                            \\tensor1: {any}"
                            \\tensor2: {any}
                        , .{ shape, Other.shape }));
                    }
                }
                mm_shape[mm_ndims - 2] = n;
                mm_shape[mm_ndims - 1] = p;
                return InferredStrides(dtype, mm_shape);
            }
            @compileError(comptimePrint(
                \\[TESSERACT COMPILE ERROR]
                \\Cannot perform matrix multiplication on these two tensors
                \\tensor1: {any}"
                \\tensor2: {any}
            , .{ shape, Other.shape }));
        }

        // pub fn Conv2d(comptime Filter: type, _stride: anytype) type {
        //     const stride: [2]usize = switch (@typeInfo(@TypeOf(_stride))) {
        //         .ComptimeInt, .Int => [2]usize{ _stride, _stride },
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
        // }
    };
}

test "same tensors assignable" {
    // This test catches regressions caused by comptime slices with the same values not being
    // equal to teach other, which would cause this test to not compile
    // Note that the fill value is different: this should have no effect
    comptime {
        const tensor1 = InferredStrides(.i32, .{ 2, 3, 4 }).full(0);
        var tensor2 = InferredStrides(.i32, .{ 2, 3, 4 }).full(1);
        tensor2 = tensor1;
    }
}

test "permute" {
    comptime {
        const tensor1 = InferredStrides(.i32, .{ 2, 3, 4 }).full(0);
        const tensor2 = tensor1.permute(.{ 0, 2, 1 });
        try std.testing.expectEqual([_]usize{ 2, 4, 3 }, tensor2.shape);
        try std.testing.expectEqual([_]usize{ 12, 1, 4, 0 }, tensor2.strides);
    }
}

test "view" {
    comptime {
        const tensor1 = InferredStrides(.i32, .{ 2, 3, 4 }).full(0);
        const tensor2 = tensor1.view(.{ 12, 2 });
        const tensor3 = tensor2.view(.{24});
        try std.testing.expectEqual([_]usize{ 12, 2 }, tensor2.shape);
        try std.testing.expectEqual([_]usize{ 2, 1, 0 }, tensor2.strides);
        try std.testing.expectEqual([_]usize{24}, tensor3.shape);
        try std.testing.expectEqual([_]usize{ 1, 0 }, tensor3.strides);
    }
}

test "as strided" {
    // Based on example from https://pytorch.org/docs/stable/generated/torch.as_strided.html
    comptime {
        const tensor1 = InferredStrides(.i32, .{ 3, 3 }).full(0);
        const tensor2 = tensor1.asStrided(.{ 2, 2 }, .{ 1, 2, 0 });

        try std.testing.expectEqual([_]usize{ 2, 2 }, tensor2.shape);
        try std.testing.expectEqual(false, tensor2.is_contiguous);

        const test_indices = [_][2]usize{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 } };
        const expected_flat_indices1 = [_]usize{ 0, 2, 1, 3 };
        for (expected_flat_indices1, test_indices) |expected_flat_i, test_i| {
            try std.testing.expectEqual(expected_flat_i, utils.ravelMultiIndex(tensor2.ndims, tensor2.strides, test_i));
        }

        const tensor3 = tensor1.asStrided(.{ 2, 2 }, .{ 1, 2, 1 });
        try std.testing.expectEqual([_]usize{ 2, 2 }, tensor2.shape);
        try std.testing.expectEqual(false, tensor2.is_contiguous);

        const expected_flat_indices2 = [_]usize{ 1, 3, 2, 4 };
        for (expected_flat_indices2, test_indices) |expected_flat_i, test_i| {
            try std.testing.expectEqual(expected_flat_i, utils.ravelMultiIndex(tensor3.ndims, tensor3.strides, test_i));
        }
    }
}

test "map" {
    const tensor1 = comptime InferredStrides(.i32, .{ 2, 3, 4 }).full(3);
    const tensor2 = comptime tensor1.neg();
    try std.testing.expectEqual([_]usize{ 2, 3, 4 }, tensor2.shape);
    Graph.init(std.testing.allocator);
    defer Graph.deinit();
    Graph.trace(&tensor2);
    try std.testing.expect(Graph.vertexOf(&tensor2).edge.MapOp.op == .Neg);
    try std.testing.expect(Graph.vertexOf(&tensor2).edge.MapOp.x == Graph.vertexOf(&tensor1));
}

test "zip" {
    const tensor1 = comptime InferredStrides(.i32, .{ 2, 1, 4 }).full(2);
    const tensor2 = comptime InferredStrides(.i32, .{ 3, 1 }).full(3);
    const tensor3 = comptime tensor1.add(tensor2);
    try std.testing.expectEqual([_]usize{ 2, 3, 4 }, tensor3.shape);
    Graph.init(std.testing.allocator);
    defer Graph.deinit();
    Graph.trace(&tensor3);
    try std.testing.expect(Graph.vertexOf(&tensor3).edge.ZipOp.op == .Add);
    try std.testing.expect(Graph.vertexOf(&tensor3).edge.ZipOp.a.edge.TypeOp.x == Graph.vertexOf(&tensor1));
    try std.testing.expect(Graph.vertexOf(&tensor3).edge.ZipOp.b.edge.TypeOp.x == Graph.vertexOf(&tensor2));
}

test "reduce" {
    const tensor1 = comptime InferredStrides(.i32, .{ 2, 3, 4 }).full(5);
    const tensor2 = comptime tensor1.sum(1);
    try std.testing.expectEqual([_]usize{ 2, 1, 4 }, tensor2.shape);
    Graph.init(std.testing.allocator);
    defer Graph.deinit();
    Graph.trace(&tensor2);
    try std.testing.expect(Graph.vertexOf(&tensor2).edge.ReduceOp.op == .Sum);
    try std.testing.expect(Graph.vertexOf(&tensor2).edge.ReduceOp.x == Graph.vertexOf(&tensor1));
    try std.testing.expectEqual(Graph.vertexOf(&tensor2).edge.ReduceOp.dims[0..tensor2.ndims].*, ([_]bool{ false, true, false }));
}

test "multiple dim reduce" {
    const tensor1 = comptime InferredStrides(.i32, .{ 2, 3, 4 }).full(5);
    const tensor2 = comptime tensor1.sum(.{ 0, 1 });
    try std.testing.expectEqual([_]usize{ 1, 1, 4 }, tensor2.shape);
    Graph.init(std.testing.allocator);
    defer Graph.deinit();
    Graph.trace(&tensor2);
    try std.testing.expect(Graph.vertexOf(&tensor2).edge.ReduceOp.op == .Sum);
    try std.testing.expect(Graph.vertexOf(&tensor2).edge.ReduceOp.x == Graph.vertexOf(&tensor1));
    try std.testing.expectEqual(Graph.vertexOf(&tensor2).edge.ReduceOp.dims[0..tensor2.ndims].*, [_]bool{ true, true, false });
}

test "zip reduce" {
    const tensor1 = comptime InferredStrides(.i32, .{ 2, 1, 4 }).full(2);
    const tensor2 = comptime InferredStrides(.i32, .{ 2, 3, 1 }).full(3);
    const tensor3 = comptime tensor1.add(tensor2).sum(1);
    try std.testing.expectEqual([_]usize{ 2, 1, 4 }, tensor3.shape);
    Graph.init(std.testing.allocator);
    defer Graph.deinit();
    Graph.trace(&tensor3);
    try std.testing.expect(Graph.vertexOf(&tensor3).edge.ReduceOp.op == .Sum);
    // Anonymous intermediate tensor that stores tensor1 + tensor2
    const anon = Graph.vertexOf(&tensor3).edge.ReduceOp.x;
    try std.testing.expect(anon.edge.ZipOp.a.edge.TypeOp.x == Graph.vertexOf(&tensor1));
    try std.testing.expect(anon.edge.ZipOp.b.edge.TypeOp.x == Graph.vertexOf(&tensor2));
}

test "as_type" {
    const tensor1 = comptime InferredStrides(.bool, .{3}).full(true);
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

fn fn1() InferredStrides(.i32, .{ 2, 1, 4 }) {
    const tensor1 = InferredStrides(.i32, .{ 2, 1, 4 }).full(1);
    const tensor2 = InferredStrides(.i32, .{ 2, 3, 1 }).full(2);
    const tensor3 = tensor1.add(tensor2).sum(1);
    return tensor3;
}

fn fn2(input: anytype) InferredStrides(.i32, .{ 2, 3, 4 }) {
    return comptime blk: {
        const tensor4 = InferredStrides(.i32, .{ 2, 1, 4 }).full(4);
        const tensor5 = InferredStrides(.i32, .{ 2, 3, 1 }).full(5);
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

    Graph.init(std.testing.allocator);
    defer Graph.deinit();
    Graph.trace(&out);
    // Graph.viz();
}
