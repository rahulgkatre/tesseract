const std = @import("std");
const Allocator = std.mem.Allocator;
const comptimePrint = std.fmt.comptimePrint;
const utils = @import("utils.zig");
const ops = @import("ops.zig");
const Graph = @import("Graph.zig");
const dtypes = @import("dtypes.zig");

pub fn constant(comptime dtype: dtypes.DType, comptime value: anytype) Tensor(dtype, .{1}) {
    return Tensor(dtype, .{1}).full(value);
}

pub fn range(comptime dtype: dtypes.DType, comptime start: dtype, comptime stop: dtype) Tensor(dtype, .{stop - start}) {
    @setEvalBranchQuota(@as(u32, 2 * stop));
    const data: [stop - start]dtype = std.simd.iota(dtype, stop - start) + @as(@Vector(stop - start, dtype), @splat(start));
    return Tensor(dtype, .{stop - start}).from(data[0..]);
}

pub fn Tensor(comptime dtype: dtypes.DType, comptime shape: anytype) type {
    return AsStrided(dtype, shape, utils.stridesFromShape(shape));
}

fn AsStrided(comptime dtype: dtypes.DType, comptime shape: anytype, comptime strides: anytype) type {
    if (shape.len + 1 != strides.len) {
        @compileError("Provided shape ndims not compatible with provided strides ndims, you may be missing the storage offset (strides[ndims])");
    }
    return TensorView(dtype, shape.len, shape, strides);
}

// A Tensor is actually a TensorView, this is probably the best name for it because
// its generic parameters directly affect how data is accessed (viewed)
// While TensorView provides the API, the constructor is not the friendliest
// hence there is a simpler Tensor constructor
fn TensorView(
    // These generic parameters are private so they will be redeclared
    // as public constants in the result type
    comptime _dtype: dtypes.DType,
    comptime _ndims: u8,
    comptime _shape: [_ndims]usize,
    comptime _strides: [_ndims + 1]usize,
) type {
    return struct {
        const Self = @This();
        pub const dtype: dtypes.DType = _dtype;
        pub const ndims: u8 = _ndims;
        pub const shape: [ndims]usize = _shape;
        pub const strides: [ndims + 1]usize = _strides;
        pub const size = utils.storageSizeForTensor(ndims, shape, strides);

        ndims: u8 = ndims,
        dtype: dtypes.DType = dtype,
        shape: [ndims]usize = shape,
        size: usize = size,
        strides: [ndims + 1]usize = strides,
        traceFn: *const fn (self: *const Self) void,

        pub fn init(
            comptime traceFn: *const fn (self: *const Self) void,
        ) Self {
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

        pub fn isContiguous(_: *const Self) bool {
            return comptime utils.isContiguous(ndims, strides);
        }

        pub fn Permute(comptime perm: [ndims]u8) type {
            var strides_perm: [ndims + 1]u8 = undefined;
            @memcpy(strides_perm[0..ndims], &perm);
            strides_perm[ndims] = ndims;
            return AsStrided(
                dtype,
                utils.permuteArray(ndims, shape, perm),
                utils.permuteArray(ndims + 1, strides, strides_perm),
            );
        }
        pub fn permute(x: *const Self, comptime perm: [ndims]u8) Permute(perm) {
            comptime {
                const Out = Permute(perm);
                const traceFn = struct {
                    fn trace(out: *const Out) void {
                        Graph.trace(x);
                        Graph.Vertex.new(out, .{
                            .TypeOp = .{
                                .op = .Permute,
                                .x = Graph.Vertex.get(x),
                                .new_info = .{ .Permute = perm[0..] },
                            },
                        }, Out);
                    }
                }.trace;
                return Out.init(traceFn);
            }
        }

        pub fn view(x: *const Self, comptime new_shape: anytype) Tensor(dtype, new_shape) {
            comptime {
                const Out = Tensor(dtype, new_shape);
                std.debug.assert(Out.size == size);
                if (x.isContiguous()) {
                    const traceFn = struct {
                        fn trace(out: *const Out) void {
                            Graph.trace(x);
                            Graph.Vertex.new(out, .{ .TypeOp = .{
                                .op = .View,
                                .x = Graph.Vertex.get(x),
                                .new_info = .{ .View = @as([new_shape.len]usize, new_shape)[0..] },
                            } }, Out);
                        }
                    }.trace;
                    return Out.init(traceFn);
                } else {
                    @compileError("Must be contiguous to view");
                }
            }
        }

        pub fn asStrided(x: *const Self, comptime new_shape: anytype, comptime new_strides: anytype) AsStrided(dtype, new_shape, new_strides) {
            comptime {
                const Out = AsStrided(dtype, new_shape, new_strides);
                const traceFn = struct {
                    fn trace(out: *const Out) void {
                        Graph.trace(x);
                        Graph.Vertex.new(out, .{ .TypeOp = .{
                            .op = .AsStrided,
                            .x = Graph.Vertex.get(x),
                            .new_info = .{ .AsStrided = @as([new_strides.len]usize, new_strides)[0..] },
                        } }, Out);
                    }
                }.trace;
                return Out.init(traceFn);
            }
        }

        pub fn AsType(comptime new_dtype: dtypes.DType) type {
            return TensorView(new_dtype, ndims, shape, strides);
        }
        pub fn asType(x: *const Self, comptime new_dtype: dtypes.DType) AsType(new_dtype) {
            comptime {
                const Out: type = AsType(new_dtype);
                const traceFn = struct {
                    fn trace(out: *const Out) void {
                        Graph.trace(x);
                        Graph.Vertex.new(out, .{ .TypeOp = .{
                            .op = .AsType,
                            .x = Graph.Vertex.get(x),
                            .new_info = .{ .AsType = new_dtype },
                        } }, Out);
                    }
                }.trace;
                return Out.init(traceFn);
            }
        }

        pub fn map(x: *const Self, comptime op: ops.MapOp) Self {
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
        pub fn exp2(x: *const Self) Self {
            return x.map(.Exp2);
        }
        pub fn log2(x: *const Self) Self {
            return x.map(.Log2);
        }
        pub fn neg(x: *const Self) Self {
            return x.map(.Neg);
        }
        pub fn recip(x: *const Self) Self {
            return x.map(.Recip);
        }
        pub fn sin(x: *const Self) Self {
            return x.map(.Sin);
        }
        pub fn sqrt(x: *const Self) Self {
            return x.map(.Sqrt);
        }

        pub fn Broadcast(comptime OtherTensorType: type, comptime new_dtype: dtypes.DType) type {
            if (dtype != OtherTensorType.dtype) {
                @compileError("Cannot broadcast tensors as they do not have the same dtype, please cast first");
            }
            const bc_ndims = @max(ndims, OtherTensorType.ndims);
            var bc_shape: [bc_ndims]usize = undefined;
            for (0..bc_ndims) |i| {
                const dim1 = if (i >= ndims) 1 else shape[ndims - i - 1];
                const dim2 = if (i >= OtherTensorType.ndims) 1 else OtherTensorType.shape[OtherTensorType.ndims - i - 1];
                if (dim1 != 1 and dim2 != 1 and dim1 != dim2) {
                    @compileError(comptimePrint(
                        "Cannot broadcast tensors of shapes {any} and {any}",
                        .{ shape, OtherTensorType.shape },
                    ));
                }
                bc_shape[bc_ndims - i - 1] = if (dim1 == dim2 or dim2 == 1) dim1 else dim2;
            }
            return Tensor(new_dtype, bc_shape);
        }
        fn ZipNewDType(comptime op: ops.ZipOp) dtypes.DType {
            return switch (op) {
                .Equals, .LessThan => .bool,
                else => dtype,
            };
        }
        pub fn zip(a: *const Self, comptime op: ops.ZipOp, b: anytype) Broadcast(@TypeOf(b), ZipNewDType(op)) {
            comptime {
                const Out: type = Broadcast(@TypeOf(b), ZipNewDType(op));
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
        }
        // ZipOps
        pub fn add(a: *const Self, b: anytype) Broadcast(@TypeOf(b), dtype) {
            return a.zip(.Add, b);
        }
        pub fn mul(a: *const Self, b: anytype) Broadcast(@TypeOf(b), dtype) {
            return a.zip(.Mul, b);
        }
        pub fn maximum(a: *const Self, b: anytype) Broadcast(@TypeOf(b), dtype) {
            return a.zip(.Maximum, b);
        }
        pub fn mod(a: *const Self, b: anytype) Broadcast(@TypeOf(b)) {
            return a.zip(.Mod, b);
        }
        pub fn less_than(a: *const Self, b: anytype) Broadcast(@TypeOf(b), bool) {
            return a.zip(.LessThan, b);
        }
        pub fn equals(a: *const Self, b: anytype) Broadcast(@TypeOf(b), bool) {
            return a.zip(.Equals, b);
        }
        pub fn xor(a: *const Self, b: anytype) Broadcast(@TypeOf(b), dtype) {
            return a.zip(.Xor, b);
        }

        pub fn Reduce(comptime reduction: anytype) type {
            const ReductionType = @TypeOf(reduction);
            switch (@typeInfo(ReductionType)) {
                .ComptimeInt, .Int => {
                    const dim = reduction;
                    if (dim < 0 or dim >= ndims) {
                        @compileError("Dimension index for single dimension reduce is out of bounds");
                    }
                    var reduced_shape: [ndims]usize = undefined;
                    @memcpy(&reduced_shape, &shape);
                    reduced_shape[dim] = 1;
                    return Tensor(dtype, reduced_shape);
                },
                .Null, .Void => {
                    return Tensor(dtype, [_]usize{1} ** ndims);
                },
                else => {
                    const dims = reduction;
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
                    return Tensor(dtype, reduced_shape);
                },
            }
        }
        pub fn reduce(x: *const Self, comptime op: ops.ReduceOp, comptime reduction: anytype) Reduce(reduction) {
            comptime {
                const Out: type = Reduce(reduction);
                const reduction_dim_mask: [ndims]bool = switch (@typeInfo(@TypeOf(reduction))) {
                    .ComptimeInt, .Int => blk: {
                        var tmp_mask: [ndims]bool = [_]bool{false} ** ndims;
                        tmp_mask[reduction] = true;
                        break :blk tmp_mask;
                    },
                    .Null, .Void => [_]bool{true} ** ndims,
                    else => blk: {
                        var tmp_mask: [ndims]bool = [_]bool{false} ** ndims;
                        for (reduction) |dim| {
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
        }

        // ReduceOps
        pub fn sum(x: *const Self, comptime reduction: anytype) Reduce(reduction) {
            return x.reduce(.Sum, reduction);
        }
        pub fn max(x: *const Self, comptime reduction: anytype) Reduce(reduction) {
            return x.reduce(.Max, reduction);
        }

        // Compound functions that use the ops
        pub fn exp(x: *const Self) Self {
            // 1 / log(2) = 1.44269504089
            // e^x = 2^(x / log(2))
            return x.mul(constant(dtype, 1.44269504089)).exp2();
        }
        pub fn ln(x: *const Self) Self {
            // log(2) = 0.69314718056
            // log(x) = log2(x)log(2)
            return x.log2().mul(constant(dtype, 0.69314718056));
        }
        pub fn div(a: *const Self, b: anytype) Broadcast(@TypeOf(b), dtype) {
            return a.mul(b.recip());
        }
        pub fn sub(a: *const Self, b: anytype) Broadcast(@TypeOf(b), dtype) {
            return a.add(b.neg());
        }
    };
}

test "same tensors assignable" {
    // This test catches regressions caused by comptime slices with the same values not being
    // equal to teach other, which would cause this test to not compile
    // Note that the fill value is different: this should have no effect
    comptime {
        const tensor1 = Tensor(.i32, .{ 2, 3, 4 }).full(0);
        var tensor2 = Tensor(.i32, .{ 2, 3, 4 }).full(1);
        tensor2 = tensor1;
    }
}

test "permute" {
    comptime {
        const tensor1 = Tensor(.i32, .{ 2, 3, 4 }).full(0);
        const tensor2 = tensor1.permute(.{ 0, 2, 1 });
        try std.testing.expectEqual([_]usize{ 2, 4, 3 }, tensor2.shape);
        try std.testing.expectEqual([_]usize{ 12, 1, 4, 0 }, tensor2.strides);
    }
}

test "view" {
    comptime {
        const tensor1 = Tensor(.i32, .{ 2, 3, 4 }).full(0);
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
        const tensor1 = Tensor(.i32, .{ 3, 3 }).full(0);
        const tensor2 = tensor1.asStrided(.{ 2, 2 }, .{ 1, 2, 0 });

        try std.testing.expectEqual([_]usize{ 2, 2 }, tensor2.shape);
        try std.testing.expectEqual(false, tensor2.isContiguous());

        const test_indices = [_][2]usize{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 } };
        const expected_flat_indices1 = [_]usize{ 0, 2, 1, 3 };
        for (expected_flat_indices1, test_indices) |expected_flat_i, test_i| {
            try std.testing.expectEqual(expected_flat_i, utils.ravelMultiIndex(tensor2.ndims, tensor2.strides, test_i));
        }

        const tensor3 = tensor1.asStrided(.{ 2, 2 }, .{ 1, 2, 1 });
        try std.testing.expectEqual([_]usize{ 2, 2 }, tensor2.shape);
        try std.testing.expectEqual(false, tensor2.isContiguous());

        const expected_flat_indices2 = [_]usize{ 1, 3, 2, 4 };
        for (expected_flat_indices2, test_indices) |expected_flat_i, test_i| {
            try std.testing.expectEqual(expected_flat_i, utils.ravelMultiIndex(tensor3.ndims, tensor3.strides, test_i));
        }
    }
}

test "map" {
    const tensor1 = comptime Tensor(.i32, .{ 2, 3, 4 }).full(3);
    const tensor2 = comptime tensor1.neg();
    try std.testing.expectEqual([_]usize{ 2, 3, 4 }, tensor2.shape);
    Graph.init();
    defer Graph.deinit();
    Graph.trace(tensor2);
    try std.testing.expect(Graph.Vertex.get(&tensor2).edge.MapOp.op == .Neg);
    try std.testing.expect(Graph.Vertex.get(&tensor2).edge.MapOp.x == Graph.Vertex.get(&tensor1));
}

test "zip" {
    const tensor1 = comptime Tensor(.i32, .{ 2, 1, 4 }).full(2);
    const tensor2 = comptime Tensor(.i32, .{ 3, 1 }).full(3);
    const tensor3 = comptime tensor1.add(tensor2);
    try std.testing.expectEqual([_]usize{ 2, 3, 4 }, tensor3.shape);
    Graph.init();
    defer Graph.deinit();
    Graph.trace(tensor3);
    try std.testing.expect(Graph.Vertex.get(&tensor3).edge.ZipOp.op == .Add);
    try std.testing.expect(Graph.Vertex.get(&tensor3).edge.ZipOp.a == Graph.Vertex.get(&tensor1));
    try std.testing.expect(Graph.Vertex.get(&tensor3).edge.ZipOp.b == Graph.Vertex.get(&tensor2));
}

test "reduce" {
    const tensor1 = comptime Tensor(.i32, .{ 2, 3, 4 }).full(5);
    const tensor2 = comptime tensor1.sum(1);
    try std.testing.expectEqual([_]usize{ 2, 1, 4 }, tensor2.shape);
    Graph.init();
    defer Graph.deinit();
    Graph.trace(tensor2);
    try std.testing.expect(Graph.Vertex.get(&tensor2).edge.ReduceOp.op == .Sum);
    try std.testing.expect(Graph.Vertex.get(&tensor2).edge.ReduceOp.x == Graph.Vertex.get(&tensor1));
    try std.testing.expectEqual(Graph.Vertex.get(&tensor2).edge.ReduceOp.dims[0..tensor2.ndims].*, ([_]bool{ false, true, false }));
}

test "multiple dim reduce" {
    const tensor1 = comptime Tensor(.i32, .{ 2, 3, 4 }).full(5);
    const tensor2 = comptime tensor1.sum(.{ 0, 1 });
    try std.testing.expectEqual([_]usize{ 1, 1, 4 }, tensor2.shape);
    Graph.init();
    defer Graph.deinit();
    Graph.trace(tensor2);
    try std.testing.expect(Graph.Vertex.get(&tensor2).edge.ReduceOp.op == .Sum);
    try std.testing.expect(Graph.Vertex.get(&tensor2).edge.ReduceOp.x == Graph.Vertex.get(&tensor1));
    try std.testing.expectEqual(Graph.Vertex.get(&tensor2).edge.ReduceOp.dims[0..tensor2.ndims].*, [_]bool{ true, true, false });
}

test "zip reduce" {
    const tensor1 = comptime Tensor(.i32, .{ 2, 1, 4 }).full(2);
    const tensor2 = comptime Tensor(.i32, .{ 2, 3, 1 }).full(3);
    const tensor3 = comptime tensor1.add(tensor2).sum(1);
    try std.testing.expectEqual([_]usize{ 2, 1, 4 }, tensor3.shape);
    Graph.init();
    defer Graph.deinit();
    Graph.trace(tensor3);
    try std.testing.expect(Graph.Vertex.get(&tensor3).edge.ReduceOp.op == .Sum);
    // Anonymous intermediate tensor that stores tensor1 + tensor2
    const anon = Graph.Vertex.get(&tensor3).edge.ReduceOp.x;
    try std.testing.expect(anon.edge.ZipOp.a == Graph.Vertex.get(&tensor1));
    try std.testing.expect(anon.edge.ZipOp.b == Graph.Vertex.get(&tensor2));
}

test "as_type" {
    const tensor1 = comptime Tensor(.bool, .{3}).full(true);
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

fn fn1() Tensor(.i32, .{ 2, 1, 4 }) {
    const tensor1 = Tensor(.i32, .{ 2, 1, 4 }).full(1);
    const tensor2 = Tensor(.i32, .{ 2, 3, 1 }).full(2);
    const tensor3 = tensor1.add(tensor2).sum(1);
    return tensor3;
}

fn fn2(input: anytype) Tensor(.i32, .{ 2, 3, 4 }) {
    return comptime blk: {
        const tensor4 = Tensor(.i32, .{ 2, 1, 4 }).full(4);
        const tensor5 = Tensor(.i32, .{ 2, 3, 1 }).full(5);
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
    Graph.trace(out);
    // Graph.viz();
}
