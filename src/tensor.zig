const std = @import("std");
const comptimePrint = std.fmt.comptimePrint;
const utils = @import("utils.zig");
const ops = @import("ops.zig");
const Graph = @import("Graph.zig");
const dtypes = @import("dtypes.zig");
const functions = @import("functions.zig");
const anytensor = @import("anytensor.zig");
const Record = @import("record.zig").Record;

pub fn TensorType(comptime val: anytype) type {
    if (isTensor(@TypeOf(val))) {
        return Tensor(val.dtype, val.ndims, val.shape[0..val.ndims].*);
    } else {
        const default = dtypes.default;
        const inferred = dtypes.inferDType(val);
        if ((dtypes.isBool(inferred) and dtypes.isBool(default)) or (dtypes.isInt(inferred) and dtypes.isInt(default)) or (dtypes.isFloat(inferred) and dtypes.isFloat(default))) {
            return Scalar(default);
        } else {
            return Scalar(inferred);
        }
    }
}

pub fn FloatTensor(comptime TT: type) type {
    std.debug.assert(isTensor(TT));
    if (!dtypes.isFloat(TT.dtype)) {
        return TT.AsType(dtypes.default);
    } else {
        return TT;
    }
}

pub fn BoolTensor(comptime TT: type) type {
    std.debug.assert(isTensor(TT));
    if (!dtypes.isBool(TT.dtype)) {
        @compileError("Must be bool datatype");
    } else {
        return TT;
    }
}

pub fn IntTensor(comptime TT: type) type {
    std.debug.assert(isTensor(TT));
    if (!dtypes.isInt(TT.dtype)) {
        @compileError("Must cast to int datatype first");
    } else {
        return TT;
    }
}

pub fn range(
    comptime start: comptime_int,
    comptime stop: comptime_int,
) _Tensor(.i32, .{stop - start}) {
    return _Tensor(.i32, .{stop - start}).initContiguous(.{ .InitOp = .Range }, {}, .{
        .Range = .{
            .start = std.fmt.comptimePrint("{d}", .{start}),
            .stop = std.fmt.comptimePrint("{d}", .{stop}),
        },
    });
}

fn Scalar(comptime dtype: dtypes.DType) type {
    return _Tensor(dtype, .{1});
}

pub fn isTensor(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Pointer => |ptr| isTensor(ptr.child),
        .Struct => Tensor(T.dtype, T.ndims, T.shape) == T,
        else => false,
    };
}

/// Used for wrapping immediate values in single size tensors with the same dtype as the current tensor
pub fn tensorOf(comptime val: anytype) TensorType(val) {
    return if (!isTensor(@TypeOf(val))) TensorType(val).full(val) else val;
}

pub fn randLike(comptime other: anytype) @TypeOf(other) {
    std.debug.assert(isTensor(@TypeOf(other)));
    return @TypeOf(other).rand();
}

pub fn fullLike(comptime other: anytype, value: dtypes.ZigType(other.dtype)) @TypeOf(other) {
    std.debug.assert(isTensor(@TypeOf(other)));
    return @TypeOf(other).full(value);
}

pub fn _Tensor(comptime dtype: dtypes.DType, comptime shape: anytype) type {
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
        const Self = @This();
        pub usingnamespace functions.Functions(Self);

        // Type level constants for comptime shape logic (e.g. @TypeOf(a).ndims)
        pub const dtype: dtypes.DType = _dtype;
        pub const ndims: u8 = _ndims;
        pub const shape: [ndims]u64 = _shape;

        ndims: u8 = ndims,
        dtype: dtypes.DType = dtype,
        shape: []const u64 = &shape,
        strides: []const u64,
        offset: u64,
        record: Record,

        pub fn initContiguous(record: Record) Self {
            return .{ .record = record, .strides = &utils.contiguousStrides(ndims, shape), .offset = 0 };
        }

        pub fn trace(self: *const Self) void {
            anytensor.trace(@ptrCast(self));
        }

        pub fn isContiguous(self: Self) bool {
            // Strides need to be decreasing unless its a broadcasted stride (0)
            var prev: u64 = std.math.maxInt(u64);
            for (self.strides) |s| {
                if (s > 0) {
                    if (s > prev) {
                        return false;
                    }
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

        /// Supports negative indexing sugar (e.g. -1 = ndims - 1)
        pub fn dimSize(_: Self, d: i16) u64 {
            return shape[normalizedDim(d)];
        }

        /// Used to mark a tensor as an input to a graph,
        /// codegen will make this an argument of the function
        pub fn input() Self {
            return initContiguous(Record.init(
                .InitOp,
                .Input,
                {},
                .{ .Input = {} },
            ));
        }

        pub fn param() Self {
            return initContiguous(Record.init(
                .InitOp,
                .Parameter,
                {},
                .{ .Parameter = {} },
            ));
        }

        /// Fill a tensor with a value
        pub fn full(comptime value: dtypes.ZigType(dtype)) Self {
            return initContiguous(
                Record.init(
                    .InitOp,
                    .Full,
                    {},
                    .{ .Full = std.fmt.comptimePrint("{}", .{value}) },
                ),
            );
        }

        pub fn rand() Self {
            std.debug.assert(dtypes.isFloat(dtype));
            return initContiguous(Record.init(
                .InitOp,
                .Rand,
                .{ .Rand = {} },
            ));
        }

        fn Permute(comptime perm: [ndims]u8) type {
            return View(utils.arrayPermute(u64, ndims, shape, perm));
        }
        /// Permute the dimensions of the tensor. A valid permutation must contain
        /// values from 0 to ndims and each value must appear exactly once.
        pub fn permute(a: Self, comptime perm: [ndims]u8) Permute(perm) {
            return a.asStrided(
                Permute(perm).shape,
                utils.arrayPermute(u64, ndims, a.strides[0..ndims].*, perm),
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
                var new_strides = a.strides[0..a.ndims].*;
                new_strides[norm1] = a.strides[norm2];
                new_strides[norm2] = a.strides[norm1];
                return a.asStrided(
                    Transpose(norm1, norm2).shape,
                    new_strides,
                    a.offset,
                );
            } else {
                return a;
            }
        }

        pub fn View(comptime new_shape: anytype) type {
            return _Tensor(dtype, new_shape);
        }
        /// View the tensor as a different shape.
        pub fn view(a: Self, comptime new_shape: anytype) View(new_shape) {
            if (!isContiguous(a)) {
                return a.copy().view(new_shape);
            } else {
                return a.asStrided(new_shape, utils.contiguousStrides(new_shape.len, new_shape), a.offset);
            }
        }

        pub fn Flatten(comptime start_dim: i16, comptime end_dim: i16) type {
            const norm_start = normalizedDim(start_dim);
            const norm_end = normalizedDim(end_dim);
            if (norm_start == norm_end) {
                return Self;
            } else {
                var new_shape: [ndims - (norm_end - norm_start)]u64 = undefined;
                new_shape[norm_start] = 1;
                for (0..ndims) |d| {
                    if (d < norm_start or d > norm_end) {
                        new_shape[d] = shape[d];
                    } else {
                        new_shape[norm_start] *= shape[d];
                    }
                }
                return View(new_shape);
            }
        }

        pub fn flattenPartial(a: Self, comptime start_dim: i16, comptime end_dim: i16) Flatten(start_dim, end_dim) {
            return a.view(Flatten(start_dim, end_dim).shape);
        }

        pub fn flatten(a: Self) Flatten(0, -1) {
            return a.view(Flatten(0, -1).shape);
        }

        pub fn Unsqueeze(comptime dim: i16) type {
            return View(utils.arrayInsert(ndims, shape, normalizedDim(dim), 1));
        }
        /// Insert a dim of size 1 into the shape of the tensor.
        pub fn unsqueeze(a: Self, comptime dim: i16) Unsqueeze(dim) {
            return a.asStrided(
                Unsqueeze(dim).shape,
                utils.arrayInsert(ndims, a.strides[0..ndims].*, normalizedDim(dim), 0),
                a.offset,
            );
        }

        pub fn Squeeze(comptime dim: i16) type {
            if (shape[normalizedDim(dim)] != 1) {
                @compileError("Cannot squeeze as dimension size is not 1");
            }
            return View(utils.arrayDelete(ndims, shape, normalizedDim(dim)));
        }
        /// Remove a dim of size 1 from the shape of the tensor.
        pub fn squeeze(a: Self, comptime dim: i16) Squeeze(dim) {
            return a.asStrided(
                Squeeze(dim).shape,
                utils.arrayDelete(ndims, a.strides[0..ndims].*, normalizedDim(dim)),
                a.offset,
            );
        }

        /// Changes the shape and stride of the tensor to change how the underlying memory is accessed.
        /// Powerful enough to be used to implement any reshaping or windowing operation on a tensor.
        /// There are guiderails to prevent out of bounds access into underlying memory.
        pub fn asStrided(a: Self, comptime new_shape: anytype, comptime new_strides: [new_shape.len]u64, new_offset: u64) View(new_shape) {
            var out = View(new_shape){
                .record = Record.init(.TypeOp, .AsStrided, .{@ptrCast(&a)}, {}),
                .strides = &new_strides,
                .offset = new_offset,
            };
            if (out.size() > a.size()) {
                @compileError(comptimePrint(
                    \\New shape and strides will go out of bounds of the underlying memory
                    \\Old shape: {any}
                    \\Old strides: {any}
                    \\Old storage offset: {}
                    \\Old memory size: {}
                    \\New shape: {any}
                    \\New strides: {any}
                    \\New storage offset: {}
                    \\New memory size: {}
                , .{
                    a.shape,
                    a.strides,
                    a.offset,
                    a.size(),
                    out.shape,
                    out.strides,
                    out.offset,
                    out.size(),
                }));
            }
            return out;
        }

        pub fn AsType(comptime new_dtype: dtypes.DType) type {
            return _Tensor(new_dtype, shape);
        }
        ///Cast an array of a datatype to another datatype
        pub fn asType(a: Self, comptime new_dtype: dtypes.DType) AsType(new_dtype) {
            return _Tensor(new_dtype, shape).initContiguous(Record.init(.TypeOp, .AsType, .{@ptrCast(&a)}, {}));
        }

        ///Apply an elementwise map operation
        pub fn map(a: Self, comptime op: ops.MapOp) Self {
            return initContiguous(Record.init(.MapOp, op, .{@ptrCast(&a)}, {}));
        }

        pub fn Broadcast(comptime other_shape: anytype) type {
            if (std.mem.eql(u64, &shape, &other_shape)) {
                return Self;
            }
            const bc_ndims = @max(ndims, other_shape.len);
            var bc_shape: [bc_ndims]u64 = undefined;
            for (0..bc_ndims) |i| {
                const dim1 = if (i >= ndims) 1 else shape[ndims - i - 1];
                const dim2 = if (i >= other_shape.len) 1 else other_shape[other_shape.len - i - 1]; // orelse dim1;
                if (dim1 != 1 and dim2 != 1 and dim1 != dim2) {
                    @compileError(comptimePrint(
                        \\Tensor shapes are not comaptible for broadcasting
                        \\Tensor A shape: {any}
                        \\Tensor B shape: {any}
                    ,
                        .{ shape, other_shape },
                    ));
                }
                bc_shape[bc_ndims - i - 1] = if (dim1 == dim2 or dim2 == 1) dim1 else dim2;
            }
            return View(bc_shape);
        }
        pub fn expand(a: Self, comptime new_shape: anytype) Broadcast(new_shape) {
            const Out = Broadcast(new_shape);
            if (Self == Out) {
                return a;
            }
            var bc_strides: [new_shape.len]u64 = undefined;
            for (0..new_shape.len) |i| {
                bc_strides[new_shape.len - i - 1] = if (i >= ndims) 0 else a.strides[ndims - i - 1];
            }
            return a.asStrided(new_shape, bc_strides, a.offset);
        }

        pub fn normalizedDim(dim: i16) u8 {
            const normalized = if (dim < 0) ndims + dim else dim;
            if (normalized < 0 or normalized > ndims) {
                @compileError(comptimePrint(
                    "Dimension index {d} is out of bounds {d}",
                    .{ normalized, ndims },
                ));
            }
            return @intCast(normalized);
        }

        pub fn Zip(comptime Other: type, comptime op: ops.ZipOp) type {
            std.debug.assert(isTensor(Other));
            const bc_shape = Broadcast(Other.shape).shape;
            const new_dtype = utils.zipResultDType(op, Self.dtype, Other.dtype);
            return _Tensor(new_dtype, bc_shape);
        }

        /// Apply an elementwise zip (binary) operation on two arrays, with broadcasting
        pub fn zip(a: Self, b: anytype, comptime op: ops.ZipOp) Zip(TensorType(b), op) {
            const b_tensor = comptime tensorOf(b);
            // Expand a and b to match the output shape
            // const a_expand = comptime a.expand(bc_shape);
            // const b_expand = comptime b_tensor.expand(bc_shape);
            return Zip(TensorType(b), op).initContiguous(Record.init(
                .ZipOp,
                op,
                .{ @ptrCast(&a), @ptrCast(&b_tensor) },
                {},
            ));
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
                    return View(.{1});
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
        pub fn reduce(
            a: Self,
            comptime op: ops.ReduceOp,
            comptime reduce_dims: anytype,
        ) Reduce(reduce_dims) {
            const reduce_dim_mask: [ndims]bool = switch (@typeInfo(@TypeOf(reduce_dims))) {
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

            return Reduce(reduce_dims).initContiguous(Record.init(.ReduceOp, op, .{@ptrCast(&a)}, &reduce_dim_mask));
        }

        pub fn MatMul(comptime other: anytype) type {
            // Matrix multiplication invariant
            // (n a m1) matmul (m2 a p) -> (n a p) iff m1 = m2
            // otherwise matmul is invalid, compile error
            const n = if (ndims == 1) 1 else shape[ndims - 2];
            const m = shape[ndims - 1];
            const other_m = if (other.ndims == 1) 1 else other.shape[other.ndims - 2];
            const p = other.shape[other.ndims - 1];

            if (m == other_m) {
                const mm_ndims = @max(ndims, other.ndims);
                var mm_shape: [mm_ndims]u64 = undefined;
                // Broadcasting check, look only at batch dimensions (everything before last 2 dimensions)
                for (0..mm_ndims - 2) |i| {
                    const dim1 = if (i >= ndims - 2) 1 else shape[ndims - i - 3];
                    const dim2 = if (i >= other.ndims - 2) 1 else other.shape[other.ndims - i - 3];
                    if (dim1 == dim2 or dim1 == 1 or dim2 == 1) {
                        mm_shape[mm_ndims - i - 3] = if (dim1 == dim2 or dim2 == 1) dim1 else dim2;
                    } else {
                        @compileError(comptimePrint(
                            \\Tensors have incompatible shapes for batch matrix multiplication
                            \\Tensor A: {any}
                            \\Tensor B: {any}
                        , .{ shape, other.shape }));
                    }
                }
                mm_shape[mm_ndims - 2] = n;
                mm_shape[mm_ndims - 1] = p;
                return View(mm_shape);
            }
            @compileError(comptimePrint(
                \\Tensors have incompatible shapes for batch matrix multiplication
                \\Tensor A: {any}
                \\Tensor B: {any}
            , .{ shape, other.shape }));
        }
        pub fn matmul(a: Self, b: anytype) MatMul(b) {
            return a
                .unsqueeze(a.ndims - 1)
                .mul(b.transpose(b.ndims - 2, b.ndims - 1).copy().unsqueeze(b.ndims - 2))
                .sum(a.ndims)
                .squeeze(a.ndims);
        }

        pub fn Where(comptime true_value: anytype, comptime false_value: anytype) type {
            const true_tensor = tensorOf(true_value);
            const false_tensor = tensorOf(false_value);
            std.debug.assert(true_tensor.dtype == false_tensor.dtype);
            std.debug.assert(dtypes.isBool(Self.dtype));
            const T = @TypeOf(true_tensor);
            const F = @TypeOf(false_tensor);
            const TF = T.Broadcast(F.shape);
            return _Tensor(TF.dtype, Broadcast(TF.shape).shape);
        }
        /// Conditional elementwise operator
        /// out[i] = if (mask[i]) true_value[i] else false_value[i]
        /// Supports broadcasting between all 3 tensors, but true value and false value are broadcasted together first and must also have the same dtype
        pub fn where(mask: Self, true_value: anytype, false_value: anytype) Where(true_value, false_value) {
            const Out = Where(true_value, false_value);
            const mask_expand = mask.expand(Out.shape);
            const true_expand = tensorOf(true_value).expand(Out.shape);
            const false_expand = tensorOf(false_value).expand(Out.shape);
            return Out.initContiguous(Record.init(.TernaryOp, .Where, .{ @ptrCast(&mask_expand), @ptrCast(&true_expand), @ptrCast(&false_expand) }, {}));
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
        const tensor1 = _Tensor(.i32, .{ 2, 3, 4 }).full(0);
        var tensor2 = _Tensor(.i32, .{ 2, 3, 4 }).full(1);
        var tensor3 = _Tensor(tensor2.dtype, tensor2.shape[0..tensor2.ndims].*).full(2);
        tensor2 = tensor1;
        tensor3 = tensor2;
    }
}

test "permute" {
    const tensor1 = comptime _Tensor(.i32, .{ 2, 3, 4 }).full(0);
    const tensor2 = comptime tensor1.permute(.{ 0, 2, 1 });
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 4, 3 }, tensor2.shape);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 12, 1, 4 }, tensor2.strides);
}

test "view" {
    const tensor1 = comptime _Tensor(.i32, .{ 2, 3, 4 }).full(0);
    const tensor2 = comptime tensor1.view(.{ 12, 2 });
    const tensor3 = comptime tensor2.view(.{24});
    try std.testing.expectEqualSlices(u64, &[_]u64{ 12, 2 }, tensor2.shape);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 1 }, tensor2.strides);
    try std.testing.expectEqualSlices(u64, &[_]u64{24}, tensor3.shape);
    try std.testing.expectEqualSlices(u64, &[_]u64{1}, tensor3.strides);
}

test "as strided" {
    // Based on example from https://pytorch.org/docs/stable/generated/torch.as_strided.html
    const tensor1 = comptime _Tensor(.i32, .{ 3, 3 }).full(0);
    const tensor2 = comptime tensor1.asStrided(.{ 2, 2 }, .{ 1, 2 }, 0);

    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 2 }, tensor2.shape);
    try std.testing.expectEqual(false, tensor2.isContiguous());

    const test_indices = [_][2]u64{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 } };
    const expected_flat_indices1 = &[_]u64{ 0, 2, 1, 3 };
    for (expected_flat_indices1, test_indices) |expected_flat_i, test_i| {
        try std.testing.expectEqual(expected_flat_i, utils.ravelMultiIndex(tensor2.ndims, tensor2.strides[0..tensor2.ndims].*, tensor2.offset, test_i));
    }

    const tensor3 = comptime tensor1.asStrided(.{ 2, 2 }, .{ 1, 2 }, 1);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 2 }, tensor2.shape);
    try std.testing.expectEqual(false, tensor2.isContiguous());

    const expected_flat_indices2 = &[_]u64{ 1, 3, 2, 4 };
    for (expected_flat_indices2, test_indices) |expected_flat_i, test_i| {
        try std.testing.expectEqual(expected_flat_i, utils.ravelMultiIndex(tensor3.ndims, tensor3.strides[0..tensor3.ndims].*, tensor3.offset, test_i));
    }
}

test "map" {
    const tensor1 = comptime _Tensor(.i32, .{ 2, 3, 4 }).full(3);
    const tensor2 = comptime tensor1.neg();
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 4 }, tensor2.shape);
    Graph.init();
    defer Graph.deinit();
    tensor2.trace();
    try std.testing.expect(tensor2.record.MapOp.op == .Neg);
    try std.testing.expectEqual(tensor2.record.MapOp.a.*, tensor1.node());
}

test "zip" {
    const tensor1 = comptime _Tensor(.i32, .{ 2, 1, 4 }).full(2);
    const tensor2 = comptime _Tensor(.i32, .{ 3, 1 }).full(3);
    const tensor3 = comptime tensor1.add(tensor2);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 4 }, tensor3.shape);
    Graph.init();
    defer Graph.deinit();
    tensor3.trace();
    try std.testing.expect(tensor3.record.ZipOp.op == .Add);
    try std.testing.expectEqual(tensor3.record.ZipOp.a.record.TypeOp.a.*, tensor1.node());
    try std.testing.expectEqual(tensor3.record.ZipOp.b.record.TypeOp.a.*, tensor2.node());
}

test "reduce" {
    const tensor1 = comptime _Tensor(.i32, .{ 2, 3, 4 }).full(5);
    const tensor2 = comptime tensor1.sum(1);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 1, 4 }, tensor2.shape);
    Graph.init();
    defer Graph.deinit();
    tensor2.trace();
    try std.testing.expect(tensor2.record.ReduceOp.op == .Sum);
    try std.testing.expectEqual(tensor2.record.ReduceOp.a.*, tensor1.node());
    try std.testing.expectEqual(tensor2.record.ReduceOp.dims[0..tensor2.ndims].*, ([_]bool{ false, true, false }));
}

test "multiple dim reduce" {
    const tensor1 = comptime _Tensor(.i32, .{ 2, 3, 4 }).full(5);
    const tensor2 = comptime tensor1.sum(.{ 0, 1 });
    try std.testing.expectEqualSlices(u64, &[_]u64{ 1, 1, 4 }, tensor2.shape);
    Graph.init();
    defer Graph.deinit();
    tensor2.trace();
    try std.testing.expect(tensor2.record.ReduceOp.op == .Sum);
    try std.testing.expectEqual(tensor2.record.ReduceOp.a.*, tensor1.node());
    try std.testing.expectEqual(tensor2.record.ReduceOp.dims[0..tensor2.ndims].*, [_]bool{ true, true, false });
}

test "zip reduce" {
    const tensor1 = comptime _Tensor(.i32, .{ 2, 1, 4 }).full(2);
    const tensor2 = comptime _Tensor(.i32, .{ 2, 3, 1 }).full(3);
    const tensor3 = comptime tensor1.add(tensor2).sum(1);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 1, 4 }, tensor3.shape);
    Graph.init();
    defer Graph.deinit();
    tensor3.trace();
    try std.testing.expect(tensor3.record.ReduceOp.op == .Sum);
    // Anonymous intermediate tensor that stores tensor1 + tensor2
    const anon = tensor3.record.ReduceOp.a;
    try std.testing.expectEqual(anon.record.ZipOp.a.record.TypeOp.a.*, tensor1.node());
    try std.testing.expectEqual(anon.record.ZipOp.b.record.TypeOp.a.*, tensor2.node());
}

test "as_type" {
    const tensor1 = comptime _Tensor(.bool, .{3}).full(true);
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

fn fn1() _Tensor(.i32, .{ 2, 1, 4 }) {
    const tensor1 = _Tensor(.i32, .{ 2, 1, 4 }).full(1);
    const tensor2 = _Tensor(.i32, .{ 2, 3, 1 }).full(2);
    const tensor3 = tensor1.add(tensor2).sum(1);
    return tensor3;
}

fn fn2(input: anytype) _Tensor(.i32, .{ 2, 3, 4 }) {
    return comptime blk: {
        const tensor4 = _Tensor(.i32, .{ 2, 1, 4 }).full(4);
        const tensor5 = _Tensor(.i32, .{ 2, 3, 1 }).full(5);
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
