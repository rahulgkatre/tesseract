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

pub fn range(comptime dtype: dtypes.DType, comptime start: dtype, comptime stop: dtype) InferredStrides(dtype, .{stop - start}) {
    @setEvalBranchQuota(@as(u32, 2 * stop));
    const data: [stop - start]dtype = std.simd.iota(dtype, stop - start) + @as(@Vector(stop - start, dtype), @splat(start));
    return InferredStrides(dtype, .{stop - start}).from(data[0..]);
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

        // Load the tensor's data from an array pointer
        // Not a slice because this guarantees that the size requirement is met and verified in comptime
        // pub fn from(data: *const [size]anytype) Self {
        //     _ = data;
        //     const traceFn = struct {
        //         fn trace(self: *const Self) void {
        //             Graph.Node.new(self, .{ .InitOp = .{ .op = .From } }, Self);
        //         }
        //     }.trace;
        //     return init(traceFn);
        // }

        // Fill a tensor with a value
        pub fn full(comptime value: anytype) Self {
            _ = value;
            const traceFn = struct {
                fn trace(self: *const Self) void {
                    Graph.Vertex.new(self, .{ .InitOp = .{ .op = .Full } }, Self);
                }
            }.trace;
            return init(traceFn);
        }

        // Fill a tensor with a value
        pub fn rand(comptime value: dtype) Self {
            _ = value;
            const traceFn = struct {
                fn trace(self: *const Self) void {
                    Graph.Vertex.new(self, .{ .InitOp = .{ .op = .Rand } }, Self);
                }
            }.trace;
            return init(traceFn);
        }

        pub fn Permute(comptime perm: [ndims]u8) type {
            var strides_perm: [ndims + 1]u8 = undefined;
            @memcpy(strides_perm[0..ndims], &perm);
            strides_perm[ndims] = ndims;
            return AsStrided(
                utils.arrayPermute(ndims, shape, perm),
                utils.arrayPermute(ndims + 1, strides, strides_perm),
            );
        }
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
        pub fn transpose(x: *const Self, comptime dim1: u8, comptime dim2: u8) Transpose(dim1, dim2) {
            if (dim1 != dim2) {
                const Out = Transpose(dim1, dim2);
                return x.asStrided(Out.shape, Out.strides);
            } else {
                return x.*;
            }
        }
        pub fn view(x: *const Self, comptime new_shape: anytype) InferredStrides(dtype, new_shape) {
            comptime {
                const Out = InferredStrides(dtype, new_shape);
                if (x.is_contiguous) {
                    const traceFn = struct {
                        fn trace(out: *const Out) void {
                            Graph.trace(x);
                            Graph.Vertex.new(out, .{ .TypeOp = .{
                                .op = .View,
                                .x = Graph.Vertex.get(x),
                            } }, Out);
                        }
                    }.trace;
                    return Out.init(traceFn);
                } else {
                    @compileError("Must be contiguous to view");
                }
            }
        }
        pub fn reshape(x: *const Self, comptime new_shape: anytype) InferredStrides(dtype, new_shape) {
            comptime {
                if (x.is_contiguous) {
                    return x.view(new_shape);
                } else {
                    // return x.copy().view(new_shape);
                }
            }
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
            return Tensor(dtype, new_shape.len, new_shape, new_strides);
        }
        pub fn asStrided(comptime x: *const Self, comptime new_shape: anytype, comptime new_strides: anytype) AsStrided(new_shape, new_strides) {
            const Out = AsStrided(new_shape, new_strides);
            const traceFn = struct {
                fn trace(out: *const Out) void {
                    Graph.trace(x);
                    Graph.Vertex.new(out, .{ .TypeOp = .{
                        .op = .AsStrided,
                        .x = Graph.Vertex.get(x),
                    } }, Out);
                }
            }.trace;
            return Out.init(traceFn);
        }

        pub fn asType(comptime x: *const Self, comptime new_dtype: dtypes.DType) Tensor(new_dtype, ndims, shape, strides) {
            const Out: type = Tensor(new_dtype, ndims, shape, strides);
            const traceFn = struct {
                fn trace(out: *const Out) void {
                    Graph.trace(x);
                    Graph.Vertex.new(out, .{ .TypeOp = .{
                        .op = .AsType,
                        .x = Graph.Vertex.get(x),
                    } }, Out);
                }
            }.trace;
            return Out.init(traceFn);
        }
        pub fn map(comptime x: *const Self, comptime op: ops.MapOp) Self {
            comptime {
                const Out: type = @TypeOf(x.*);
                const traceFn = struct {
                    fn trace(out: *const Out) void {
                        Graph.trace(x);
                        Graph.Vertex.new(out, .{ .MapOp = .{
                            .op = op,
                            .x = Graph.Vertex.get(x),
                        } }, Out);
                    }
                }.trace;
                return Out.init(traceFn);
            }
        }

        pub fn Broadcast(comptime Other: type, comptime op: ops.ZipOp) type {
            const new_dtype = switch (op) {
                .Equals, .LessThan => .bool,
                else => dtype,
            };
            const bc_ndims = @max(ndims, Other.ndims);
            var bc_shape: [bc_ndims]usize = undefined;
            for (0..bc_ndims) |i| {
                const dim1 = if (i >= ndims) 1 else shape[ndims - i - 1];
                const dim2 = if (i >= Other.ndims) 1 else Other.shape[Other.ndims - i - 1];
                if (dim1 != 1 and dim2 != 1 and dim1 != dim2) {
                    @compileError(comptimePrint(
                        "Cannot broadcast tensors of shapes {any} and {any}",
                        .{ shape, Other.shape },
                    ));
                }
                bc_shape[bc_ndims - i - 1] = if (dim1 == dim2 or dim2 == 1) dim1 else dim2;
            }
            return InferredStrides(new_dtype, bc_shape);
        }
        pub fn zip(comptime a: *const Self, comptime op: ops.ZipOp, comptime b: anytype) Broadcast(@TypeOf(b), op) {
            const Out: type = Broadcast(@TypeOf(b), op);
            const traceFn = struct {
                fn trace(out: *const Out) void {
                    Graph.trace(a);
                    Graph.trace(b);
                    Graph.Vertex.new(out, .{
                        .ZipOp = .{
                            .op = op,
                            .a = Graph.Vertex.get(a),
                            .b = Graph.Vertex.get(&b),
                        },
                    }, Out);
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
                    Graph.Vertex.new(out, .{ .ReduceOp = .{
                        .op = op,
                        .x = Graph.Vertex.get(x),
                        .dims = reduction_dim_mask[0..],
                    } }, Out);
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
    Graph.trace(tensor2);
    try std.testing.expect(Graph.Vertex.get(&tensor2).edge.MapOp.op == .Neg);
    try std.testing.expect(Graph.Vertex.get(&tensor2).edge.MapOp.x == Graph.Vertex.get(&tensor1));
}

test "zip" {
    const tensor1 = comptime InferredStrides(.i32, .{ 2, 1, 4 }).full(2);
    const tensor2 = comptime InferredStrides(.i32, .{ 3, 1 }).full(3);
    const tensor3 = comptime tensor1.add(tensor2);
    try std.testing.expectEqual([_]usize{ 2, 3, 4 }, tensor3.shape);
    Graph.init(std.testing.allocator);
    defer Graph.deinit();
    Graph.trace(tensor3);
    try std.testing.expect(Graph.Vertex.get(&tensor3).edge.ZipOp.op == .Add);
    try std.testing.expect(Graph.Vertex.get(&tensor3).edge.ZipOp.a == Graph.Vertex.get(&tensor1));
    try std.testing.expect(Graph.Vertex.get(&tensor3).edge.ZipOp.b == Graph.Vertex.get(&tensor2));
}

test "reduce" {
    const tensor1 = comptime InferredStrides(.i32, .{ 2, 3, 4 }).full(5);
    const tensor2 = comptime tensor1.sum(1);
    try std.testing.expectEqual([_]usize{ 2, 1, 4 }, tensor2.shape);
    Graph.init(std.testing.allocator);
    defer Graph.deinit();
    Graph.trace(tensor2);
    try std.testing.expect(Graph.Vertex.get(&tensor2).edge.ReduceOp.op == .Sum);
    try std.testing.expect(Graph.Vertex.get(&tensor2).edge.ReduceOp.x == Graph.Vertex.get(&tensor1));
    try std.testing.expectEqual(Graph.Vertex.get(&tensor2).edge.ReduceOp.dims[0..tensor2.ndims].*, ([_]bool{ false, true, false }));
}

test "multiple dim reduce" {
    const tensor1 = comptime InferredStrides(.i32, .{ 2, 3, 4 }).full(5);
    const tensor2 = comptime tensor1.sum(.{ 0, 1 });
    try std.testing.expectEqual([_]usize{ 1, 1, 4 }, tensor2.shape);
    Graph.init(std.testing.allocator);
    defer Graph.deinit();
    Graph.trace(tensor2);
    try std.testing.expect(Graph.Vertex.get(&tensor2).edge.ReduceOp.op == .Sum);
    try std.testing.expect(Graph.Vertex.get(&tensor2).edge.ReduceOp.x == Graph.Vertex.get(&tensor1));
    try std.testing.expectEqual(Graph.Vertex.get(&tensor2).edge.ReduceOp.dims[0..tensor2.ndims].*, [_]bool{ true, true, false });
}

test "zip reduce" {
    const tensor1 = comptime InferredStrides(.i32, .{ 2, 1, 4 }).full(2);
    const tensor2 = comptime InferredStrides(.i32, .{ 2, 3, 1 }).full(3);
    const tensor3 = comptime tensor1.add(tensor2).sum(1);
    try std.testing.expectEqual([_]usize{ 2, 1, 4 }, tensor3.shape);
    Graph.init(std.testing.allocator);
    defer Graph.deinit();
    Graph.trace(tensor3);
    try std.testing.expect(Graph.Vertex.get(&tensor3).edge.ReduceOp.op == .Sum);
    // Anonymous intermediate tensor that stores tensor1 + tensor2
    const anon = Graph.Vertex.get(&tensor3).edge.ReduceOp.x;
    try std.testing.expect(anon.edge.ZipOp.a == Graph.Vertex.get(&tensor1));
    try std.testing.expect(anon.edge.ZipOp.b == Graph.Vertex.get(&tensor2));
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
    Graph.trace(out);
    // Graph.viz();
}
