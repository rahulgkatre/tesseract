const std = @import("std");
const Allocator = std.mem.Allocator;
const comptimePrint = std.fmt.comptimePrint;
const utils = @import("utils.zig");
const ops = @import("ops.zig");
const Graph = @import("Graph.zig");
const dtypes = @import("dtypes.zig");
const Dim = @import("symbolic.zig").Dim;

pub fn constant(comptime dtype: dtypes.DType, comptime value: anytype) InferredStrides(dtype, .{1}) {
    return InferredStrides(dtype, .{1}).full(value);
}

pub fn range(
    comptime dtype: dtypes.DType,
    comptime start: if (dtypes.isInt(dtype)) comptime_int else @compileError("Range tensor must have int dtype"),
    comptime stop: if (dtypes.isInt(dtype)) comptime_int else @compileError("Range tensor must have int dtype"),
) InferredStrides(dtype, .{stop - start}) {
    return InferredStrides(dtype, 1, .{.{ .constant = stop - start }}).rand(start, stop);
}

fn symbolicDims(comptime shape: anytype) [shape.len]Dim {
    var symbolic_shape: [shape.len]Dim = undefined;
    for (shape, 0..) |shape_d, d| {
        symbolic_shape[d] = switch (@typeInfo(@TypeOf(shape_d))) {
            .Struct => {
                //TODO: Support for named dimensions
                @compileError("Named dimensions are not supported yet");
            },
            // Strings will get converted to an enum field for eventual named dimension support
            // Enum field value is the dim number, so it can be passed into any function that requires a dim by index
            .Pointer => .{ .variable = shape_d },
            .ComptimeInt, .Int => .{
                .constant = shape_d,
            },
            else => .{ .variable = std.fmt.comptimePrint("{any}", .{shape_d}) },
        };
    }
    return symbolic_shape;
}

fn symbolicStrides(comptime ndims: u8, comptime symbolic_shape: [ndims]Dim) [ndims + 1]Dim {
    var strides: [ndims + 1]Dim = undefined;
    var offset: Dim = .{ .constant = 1 };
    for (0..ndims - 1) |d| {
        const stride = Dim.mul(symbolic_shape[ndims - d - 1], offset);
        strides[ndims - d - 2] = stride;
        offset = stride;
    }
    strides[ndims - 1] = .{ .constant = 1 };
    strides[ndims] = .{ .constant = 0 };
    for (0..ndims) |d| {
        if (symbolic_shape[d].equalsConstant(0) or symbolic_shape[d].equalsConstant(1)) {
            strides[d] = .{ .constant = 0 };
        }
    }
    return strides;
}

pub fn UserTensor(comptime dtype: dtypes.DType, comptime shape: anytype) type {
    return InferredStrides(dtype, shape.len, symbolicDims(shape));
}

/// Utility function for defining a scalar (a 1-size tensor)
pub fn Scalar(comptime dtype: dtypes.DType) type {
    return InferredStrides(dtype, 1, .{.{ .constant = 1 }});
}

fn InferredStrides(comptime dtype: dtypes.DType, comptime ndims: u8, comptime shape: [ndims]Dim) type {
    const strides = symbolicStrides(ndims, shape);
    return Tensor(dtype, shape.len, shape, strides);
}

fn isSymbolic(comptime ndims: comptime_int, comptime dims: [ndims]Dim) bool {
    for (dims) |dim| {
        switch (dim) {
            .constant => {},
            else => return true,
        }
    }
    return false;
}

fn Tensor(
    // These generic parameters are private so they will be redeclare as pub consts in the result type
    comptime tensor_dtype: dtypes.DType,
    comptime tensor_ndims: u8,
    comptime tensor_shape: [tensor_ndims]Dim,
    comptime tensor_strides: [tensor_ndims + 1]Dim,
) type {
    return struct {
        // All the functions for operations are implemented separately
        const Self = @This();
        pub usingnamespace @import("functions.zig");

        pub const dtype = tensor_dtype;
        pub const ndims = tensor_ndims;
        pub const shape = tensor_shape;
        pub const strides = tensor_strides;
        pub const size = get_size: {
            // The storage size is 1 + last index calculated by the strides and shape
            // shape[d] - 1 is the last index in dimension d
            // Also incorporate the storage offset
            var symbolic_size: Dim = strides[ndims].add(.{ .constant = 1 });
            for (0..ndims) |d| {
                symbolic_size = symbolic_size.add((shape[d].add(.{ .constant = -1 })).mul(strides[d]));
            }
            // The result is the size of the storage needed to visit all indices of the tensor
            break :get_size symbolic_size;
        };
        pub const contiguous: ?bool = is_contiguous: {
            if (!isSymbolic(ndims + 1, strides)) {
                var prev: u64 = std.math.minInt(u64);
                for (strides[0..ndims]) |s| {
                    if (s.constant > prev and s.constant > 0) {
                        break :is_contiguous false;
                    }
                    if (s.constant > 0) {
                        prev = s.constant;
                    }
                }
                break :is_contiguous true;
            } else {
                break :is_contiguous null;
            }
        };

        dtype: dtypes.DType = dtype,
        ndims: u8 = ndims,
        shape: [ndims]Dim = shape,
        strides: [ndims + 1]Dim = strides,
        size: Dim = size,
        contiguous: ?bool = contiguous,

        pub fn any(_: *const Self) Graph.AnyTensor {
            return .{
                .dtype = dtype,
                .ndims = ndims,
                .shape = shape[0..],
                .strides = strides[0..],
                .size = size,
                .contiguous = contiguous,
            };
        }

        pub fn trace(self: *const Self) void {
            self.ref.trace();
        }

        pub fn init(comptime last_op: Graph.TensorOp) Self {
            var self = Self{};
            self.ref.last_op = last_op;
            return self;
        }

        /// Used to mark a tensor as an input to a graph,
        /// codegen will make this an argument of the function
        pub fn input() Self {
            return init(.{ .InitOp = .{ .op = .Input, .args = .{ .Input = {} } } });
        }

        /// Fill a tensor with a value
        pub fn full(comptime value: anytype) Self {
            return init(.{ .InitOp = .{ .op = .Full, .args = .{ .Full = std.fmt.comptimePrint("{}", .{value}) } } });
        }
        /// Keeps the same shape as another tensor, but initializes a new tensor
        pub fn fullLike(_: *const Self, comptime value: anytype) Self {
            return Self.full(value);
        }

        /// Internal function to fill with range, this is not publicly exposed
        /// as shape of range tensor must be constrained
        fn range(comptime start: comptime_int, comptime stop: comptime_int) Self {
            if (tensor_ndims != 1) {
                @compileError("Can only use range() on a tensor with exactly 1 dimension");
            }
            return init(.{ .InitOp = .{ .op = .Range, .args = .{
                .Range = .{
                    .start = std.fmt.comptimePrint("{d}", .{start}),
                    .stop = std.fmt.comptimePrint("{d}", .{stop}),
                },
            } } });
        }

        pub fn rand() Self {
            std.debug.assert(dtypes.isFloat(tensor_dtype));
            return init(.{ .InitOp = .{ .op = .Rand, .args = .{ .Rand = tensor_dtype } } });
        }
        pub fn randLike(_: *const Self) Self {
            return Self.rand();
        }

        /// A copy is only needed to make a non-contiguous tensor contiguous again.
        /// Each tensor is immutable and operations already produce new tensors
        /// but intermediate tensors can be eliminated through optimization.
        pub fn copy(x: *const Self) InferredStrides(tensor_dtype, tensor_ndims, tensor_shape) {
            const Out = InferredStrides(tensor_dtype, tensor_ndims, tensor_shape);
            return Out.init(.{ .MapOp = .{ .op = .Id, .x = &x.ref } });
        }

        fn Permute(comptime perm: [tensor_ndims]u8) type {
            var strides_perm: [tensor_ndims + 1]u8 = undefined;
            @memcpy(strides_perm[0..tensor_ndims], &perm);
            strides_perm[tensor_ndims] = tensor_ndims;
            return AsStridedSymbolic(
                ndims,
                utils.arrayPermute(Dim, tensor_ndims, tensor_shape, perm),
                utils.arrayPermute(Dim, tensor_ndims + 1, tensor_strides, strides_perm),
            );
        }
        /// Permute the dimensions of the tensor. A valid permutation must contain
        /// values from 0 to ndims and each value must appear exactly once.
        pub fn permute(x: *const Self, comptime perm: [tensor_ndims]u8) Permute(perm) {
            const Out = Permute(perm);
            return x.asStridedSymbolic(ndims, Out.shape, Out.strides);
        }

        pub fn Transpose(comptime dim1: u8, comptime dim2: u8) type {
            if (dim1 == dim2) {
                return Self;
            } else {
                var new_shape = tensor_shape;
                new_shape[dim1] = tensor_shape[dim2];
                new_shape[dim2] = tensor_shape[dim1];
                var new_strides = tensor_strides;
                new_strides[dim1] = tensor_strides[dim2];
                new_strides[dim2] = tensor_strides[dim1];
                return AsStridedSymbolic(ndims, new_shape, new_strides);
            }
        }
        /// Transpose two dimensions of the tensor. Similar to permute, but only for two dimensions.
        pub fn transpose(x: *const Self, comptime dim1: u8, comptime dim2: u8) Transpose(dim1, dim2) {
            if (dim1 != dim2) {
                const Out = Transpose(dim1, dim2);
                return x.asStridedSymbolic(ndims, Out.shape, Out.strides);
            } else {
                return x.*;
            }
        }
        /// View the tensor as a different shape.
        pub fn view(x: *const Self, comptime new_shape: anytype) UserTensor(tensor_dtype, new_shape) {
            const Out = UserTensor(tensor_dtype, new_shape);
            return Out.init(.{ .TypeOp = .{ .op = .AsStrided, .x = &x.ref } });
        }

        fn Unsqueeze(comptime dim: u8) type {
            if (dim > tensor_ndims) {
                @compileError("dim to unsqueeze at is out of range");
            }
            return AsStridedSymbolic(
                tensor_ndims + 1,
                utils.arrayInsert(Dim, tensor_ndims, tensor_shape, dim, .{ .constant = 1 }),
                utils.arrayInsert(Dim, tensor_ndims + 1, tensor_strides, dim, .{ .constant = 0 }),
            );
        }
        /// Insert a dim of size 1 into the shape of the tensor.
        pub fn unsqueeze(x: *const Self, comptime dim: u8) Unsqueeze(dim) {
            const Out = Unsqueeze(dim);
            return x.asStridedSymbolic(Out.ndims, Out.shape, Out.strides);
        }

        fn Squeeze(comptime dim: u8) type {
            if (dim >= tensor_ndims) {
                @compileError("dim to squeeze at is out of range");
            }
            if (!tensor_shape[dim].equalsConstant(1) or !tensor_strides[dim].equalsConstant(0)) {
                @compileError(
                    \\[TESSERACT COMPILE ERROR]
                    \\Cannot squeeze as dimension size is not 1 or stride for dimension is not 0
                    \\
                    ++
                    comptimePrint(
                    \\dim = {}
                    \\size = {}
                    \\stride = {}
                , .{
                    dim,
                    tensor_shape[dim],
                    tensor_strides[dim],
                }));
            }
            return AsStridedSymbolic(
                tensor_ndims - 1,
                utils.arrayDelete(Dim, tensor_ndims, tensor_shape, dim),
                utils.arrayDelete(Dim, tensor_ndims + 1, tensor_strides, dim),
            );
        }
        /// Remove a dim of size 1 from the shape of the tensor.
        pub fn squeeze(x: *const Self, comptime dim: u8) Squeeze(dim) {
            const Out = Squeeze(dim);
            return x.asStridedSymbolic(ndims - 1, Out.shape, Out.strides);
        }

        fn AsStrided(comptime new_shape: anytype, comptime new_strides: anytype) type {
            return AsStridedSymbolic(new_shape.len, symbolicDims(new_shape), symbolicDims(new_strides));
        }

        fn AsStridedSymbolic(comptime new_ndims: u8, comptime new_shape: [new_ndims]Dim, comptime new_strides: [new_ndims + 1]Dim) type {
            if (new_shape.len + 1 != new_strides.len) {
                @compileError(
                    \\[TESSERACT COMPILE ERROR]
                    \\Provided shape ndims not compatible with provided strides ndims
                    \\You may be missing the storage offset (strides[ndims])
                );
            }
            const Out = Tensor(tensor_dtype, new_shape.len, new_shape, new_strides);
            switch (Self.size) {
                .constant => switch (Out.size) {
                    .constant => {
                        if (Out.size.constant > Self.size.constant) {
                            @compileError(
                                \\[TESSERACT COMPILE ERROR]
                                \\Provided strides will go out of bounds of the current tensor's underlying memory
                            );
                        }
                    },
                    else => {},
                },
                else => {},
            }

            return Out;
        }
        /// Changes the shape and stride of the tensor to change how the underlying memory is accessed.
        /// Powerful enough to be used to implement any reshaping or windowing operation on a tensor.
        /// There are guiderails to prevent out of bounds access into underlying memory.
        pub fn asStrided(comptime x: *const Self, comptime new_shape: anytype, comptime new_strides: anytype) AsStrided(new_shape, new_strides) {
            const Out = AsStrided(new_shape, new_strides);
            return Out.init(.{ .TypeOp = .{ .op = .AsStrided, .x = &x.ref } });
        }

        fn asStridedSymbolic(comptime x: *const Self, comptime new_ndims: u8, comptime new_shape: [new_ndims]Dim, comptime new_strides: [new_ndims + 1]Dim) AsStridedSymbolic(new_ndims, new_shape, new_strides) {
            const Out = AsStridedSymbolic(new_ndims, new_shape, new_strides);
            return Out.init(.{ .TypeOp = .{ .op = .AsStrided, .x = &x.ref } });
        }

        ///Cast an array of a datatype to another datatype
        pub fn asType(comptime x: *const Self, comptime new_dtype: dtypes.DType) Tensor(new_dtype, tensor_ndims, tensor_shape, tensor_strides) {
            const Out = Tensor(new_dtype, tensor_ndims, tensor_shape, tensor_strides);
            return Out.init(.{ .TypeOp = .{ .op = .AsType, .x = &x.ref } });
        }

        ///Apply an elementwise map operation
        pub fn map(comptime x: *const Self, comptime op: ops.MapOp) Self {
            const Out = @TypeOf(x.*);
            return Out.init(.{ .MapOp = .{ .op = op, .x = &x.ref } });
        }

        pub fn Broadcast(comptime new_ndims: u8, comptime new_shape: [new_ndims]Dim) type {
            const bc_ndims = @max(tensor_ndims, new_shape.len);
            var bc_shape: [bc_ndims]Dim = undefined;
            for (0..bc_ndims) |i| {
                const dim1: Dim = if (i >= tensor_ndims) .{ .constant = 1 } else tensor_shape[tensor_ndims - i - 1];
                const dim2: Dim = if (i >= new_shape.len) .{ .constant = 1 } else new_shape[new_shape.len - i - 1];
                if (!dim1.equalsConstant(1) and !dim2.equalsConstant(1) and !dim1.equals(dim2)) {
                    @compileError(comptimePrint(
                        \\Cannot broadcast tensors of shapes:
                        \\t1: {any}
                        \\t2: {any}
                        \\shapes differ at dim {d}
                    ,
                        .{ tensor_shape, new_shape, i },
                    ));
                }
                bc_shape[bc_ndims - i - 1] = if (dim1.equals(dim2) or dim2.equalsConstant(1)) dim1 else dim2;
            }
            return InferredStrides(dtype, bc_ndims, bc_shape);
        }
        pub fn expand(comptime x: *const Self, comptime new_shape: anytype) Broadcast(new_shape.len, new_shape) {
            const Out = Broadcast(new_shape.len, new_shape);
            if (Self == Out) {
                return x.*;
            }
            return Out.init(.{ .TypeOp = .{ .op = .AsStrided, .x = &x.ref } });
        }

        fn Zip(comptime op: ops.ZipOp, comptime other: anytype) type {
            const Other = @TypeOf(other);
            const new_dtype = switch (op) {
                .Equals, .LessThan => .bool,
                else => tensor_dtype,
            };
            const new_layout = Broadcast(Other.ndims, Other.shape);
            return Tensor(new_dtype, new_layout.ndims, new_layout.shape, new_layout.strides);
        }

        /// Apply an elementwise zip (binary) operation on two arrays, with broadcasting
        pub fn zip(comptime a: *const Self, comptime op: ops.ZipOp, comptime b: anytype) Zip(op, b) {
            const Out = Zip(op, b);
            // Expand a and b to match the output shape
            const a_expand = comptime a.expand(Out.shape);
            const b_expand = comptime b.expand(Out.shape);
            return Out.init(.{ .ZipOp = .{ .op = op, .a = &a_expand.ref, .b = &b_expand.ref } });
        }

        pub fn Reduce(comptime reduce_dims: anytype) type {
            switch (@typeInfo(@TypeOf(reduce_dims))) {
                .ComptimeInt, .Int => {
                    const dim = reduce_dims;
                    if (dim < 0 or dim >= tensor_ndims) {
                        @compileError("Dimension index for single dimension reduce is out of bounds");
                    }
                    var reduced_shape: [tensor_ndims]Dim = undefined;
                    @memcpy(&reduced_shape, &tensor_shape);
                    reduced_shape[dim] = .{ .constant = 1 };
                    return InferredStrides(tensor_dtype, ndims, reduced_shape);
                },
                .Null, .Void => {
                    return InferredStrides(tensor_dtype, ndims, [_]Dim{.{ .constant = 1 }} ** tensor_ndims);
                },
                else => {
                    const dims = reduce_dims;
                    if (dims.len > tensor_ndims) {
                        @compileError("Length of dimension index array for multi dimension reduce is out of bounds");
                    }
                    var reduce_dim_mask: [tensor_ndims]bool = [_]bool{false} ** tensor_ndims;
                    var reduced_shape: [tensor_ndims]Dim = undefined;
                    @memcpy(&reduced_shape, &tensor_shape);
                    for (0..dims.len) |d| {
                        if (d < 0 or d >= tensor_ndims) {
                            @compileError("Dimension index for multi dimension reduce is out of bounds");
                        }
                        if (reduce_dim_mask[d]) {
                            @compileError("Cannot reuse dimension index for multi dimensional reduce");
                        }
                        reduce_dim_mask[d] = true;
                        reduced_shape[d] = .{ .constant = 1 };
                    }
                    return InferredStrides(tensor_dtype, ndims, reduced_shape);
                },
            }
        }
        /// Perform a reduction across 1 or more (or all) dimensions of a tensor.
        /// Dimensions to reduce can be passed as a int for 1 dim, tuple for multiple dims, or null/void for all dims
        pub fn reduce(comptime x: *const Self, comptime op: ops.ReduceOp, comptime reduce_dims: anytype) Reduce(reduce_dims) {
            const Out = Reduce(reduce_dims);
            const reduction_dim_mask: [tensor_ndims]bool = switch (@typeInfo(@TypeOf(reduce_dims))) {
                .ComptimeInt, .Int => blk: {
                    var tmp_mask: [tensor_ndims]bool = [_]bool{false} ** tensor_ndims;
                    tmp_mask[reduce_dims] = true;
                    break :blk tmp_mask;
                },
                .Null, .Void => [_]bool{true} ** tensor_ndims,
                else => blk: {
                    var tmp_mask: [tensor_ndims]bool = [_]bool{false} ** tensor_ndims;
                    for (reduce_dims) |dim| {
                        tmp_mask[dim] = true;
                    }
                    break :blk tmp_mask;
                },
            };
            return Out.init(.{ .ReduceOp = .{ .op = op, .x = &x.ref, .dims = &reduction_dim_mask } });
        }

        pub fn MatMul(comptime other: anytype) type {
            // Matrix multiplication invariant
            // (n x m1) matmul (m2 x p) -> (n x p) iff m1 = m2
            // otherwise matmul is invalid, compile error
            const n = if (tensor_ndims == 1) 1 else tensor_shape[tensor_ndims - 2];
            const m = tensor_shape[tensor_ndims - 1];
            const other_m = if (other.ndims == 1) 1 else other.shape[other.ndims - 2];
            const p = other.shape[other.ndims - 1];

            if (m.equals(other_m)) {
                const mm_ndims = @max(tensor_ndims, other.ndims);
                var mm_shape: [mm_ndims]Dim = undefined;
                // Broadcasting check, look only at batch dimensions (everything before last 2 dimensions)
                for (0..mm_ndims - 2) |i| {
                    const dim1: Dim = if (i >= tensor_ndims - 2) .{ .constant = 1 } else tensor_shape[tensor_ndims - i - 3];
                    const dim2: Dim = if (i >= other.ndims - 2) .{ .constant = 1 } else other.shape[other.ndims - i - 3];
                    if (dim1.equals(dim2) or dim1.equalsConstant(1) or dim2.equalsConstant(1)) {
                        mm_shape[mm_ndims - i - 3] = if (dim1.equals(dim2) or dim2.isConstantValue(1)) dim1 else dim2;
                    } else {
                        @compileError(comptimePrint(
                            \\[TESSERACT COMPILE ERROR]
                            \\Cannot perform matrix multiplication on these two tensors
                            \\tensor1: {any}
                            \\tensor2: {any}
                        , .{ tensor_shape, other.shape }));
                    }
                }
                mm_shape[mm_ndims - 2] = n;
                mm_shape[mm_ndims - 1] = p;
                return InferredStrides(tensor_dtype, mm_ndims, mm_shape);
            }
            @compileError(comptimePrint(
                \\[TESSERACT COMPILE ERROR]
                \\Cannot perform matrix multiplication on these two tensors
                \\tensor1: {any}
                \\tensor2: {any}
            , .{ tensor_shape, other.shape }));
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
        try std.testing.expectEqual([_]Dim{ .{ .constant = 2 }, .{ .constant = 4 }, .{ .constant = 3 } }, tensor2.shape);
        try std.testing.expectEqual([_]Dim{ .{ .constant = 12 }, .{ .constant = 1 }, .{ .constant = 4 }, .{ .constant = 0 } }, tensor2.strides);
    }
}

test "view" {
    comptime {
        const tensor1 = UserTensor(.i32, .{ 2, 3, 4 }).full(0);
        const tensor2 = tensor1.view(.{ 12, 2 });
        const tensor3 = tensor2.view(.{24});
        try std.testing.expectEqual([_]Dim{ .{ .constant = 12 }, .{ .constant = 2 } }, tensor2.shape);
        try std.testing.expectEqual([_]Dim{ .{ .constant = 2 }, .{ .constant = 1 }, .{ .constant = 0 } }, tensor2.strides);
        try std.testing.expectEqual([_]Dim{.{ .constant = 24 }}, tensor3.shape);
        try std.testing.expectEqual([_]Dim{ .{ .constant = 1 }, .{ .constant = 0 } }, tensor3.strides);
    }
}

test "as strided" {
    // Based on example from https://pytorch.org/docs/stable/generated/torch.as_strided.html
    comptime {
        const tensor1 = UserTensor(.i32, .{ 3, 3 }).full(0);
        const tensor2 = tensor1.asStrided(.{ 2, 2 }, .{ 1, 2, 0 });

        try std.testing.expectEqual([_]Dim{ .{ .constant = 2 }, .{ .constant = 2 } }, tensor2.shape);
        try std.testing.expectEqual(false, tensor2.contiguous);

        // const test_indices = [_][2]u64{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 } };
        // const expected_flat_indices1 = [_]u64{ 0, 2, 1, 3 };
        // for (expected_flat_indices1, test_indices) |expected_flat_i, test_i| {
        //     try std.testing.expectEqual(expected_flat_i, utils.ravelMultiIndex(tensor2.ndims, tensor2.strides, test_i));
        // }

        const tensor3 = tensor1.asStrided(.{ 2, 2 }, .{ 1, 2, 1 });
        try std.testing.expectEqual([_]Dim{ .{ .constant = 2 }, .{ .constant = 2 } }, tensor2.shape);
        try std.testing.expectEqual(false, tensor2.contiguous);
        _ = tensor3;

        // const expected_flat_indices2 = [_]u64{ 1, 3, 2, 4 };
        // for (expected_flat_indices2, test_indices) |expected_flat_i, test_i| {
        //     try std.testing.expectEqual(expected_flat_i, utils.ravelMultiIndex(tensor3.ndims, tensor3.strides, test_i));
        // }
    }
}

test "map" {
    const tensor1 = comptime UserTensor(.i32, .{ 2, 3, 4 }).full(3);
    const tensor2 = comptime tensor1.neg();
    try std.testing.expectEqual([_]Dim{ .{ .constant = 2 }, .{ .constant = 3 }, .{ .constant = 4 } }, tensor2.shape);
    Graph.init();
    defer Graph.deinit();
    tensor2.trace();
    try std.testing.expect(Graph.AnyTensor.get((&tensor2)).last_op.MapOp.op == .Neg);
    try std.testing.expect(Graph.AnyTensor.get((&tensor2)).last_op.MapOp.x == Graph.AnyTensor.get((&tensor1)));
}

test "zip" {
    const tensor1 = comptime UserTensor(.i32, .{ 2, 1, 4 }).full(2);
    const tensor2 = comptime UserTensor(.i32, .{ 3, 1 }).full(3);
    const tensor3 = comptime tensor1.add(tensor2);
    try std.testing.expectEqual([_]Dim{ .{ .constant = 2 }, .{ .constant = 3 }, .{ .constant = 4 } }, tensor3.shape);
    Graph.init();
    defer Graph.deinit();
    tensor3.trace();
    try std.testing.expect(Graph.AnyTensor.get((&tensor3)).last_op.ZipOp.op == .Add);
    try std.testing.expect(Graph.AnyTensor.get((&tensor3)).last_op.ZipOp.a.last_op.TypeOp.x == Graph.AnyTensor.get((&tensor1)));
    try std.testing.expect(Graph.AnyTensor.get((&tensor3)).last_op.ZipOp.b.last_op.TypeOp.x == Graph.AnyTensor.get((&tensor2)));
}

test "reduce" {
    const tensor1 = comptime UserTensor(.i32, .{ 2, 3, 4 }).full(5);
    const tensor2 = comptime tensor1.sum(1);
    try std.testing.expectEqual([_]Dim{ .{ .constant = 2 }, .{ .constant = 1 }, .{ .constant = 4 } }, tensor2.shape);
    Graph.init();
    defer Graph.deinit();
    tensor2.trace();
    try std.testing.expect(Graph.AnyTensor.get(&tensor2).last_op.ReduceOp.op == .Sum);
    try std.testing.expect(Graph.AnyTensor.get((&tensor2)).last_op.ReduceOp.x == Graph.AnyTensor.get((&tensor1)));
    try std.testing.expectEqual(Graph.AnyTensor.get((&tensor2)).last_op.ReduceOp.dims[0..tensor2.ndims].*, ([_]bool{ false, true, false }));
}

test "multiple dim reduce" {
    const tensor1 = comptime UserTensor(.i32, .{ 2, 3, 4 }).full(5);
    const tensor2 = comptime tensor1.sum(.{ 0, 1 });
    try std.testing.expectEqual([_]Dim{ .{ .constant = 1 }, .{ .constant = 1 }, .{ .constant = 4 } }, tensor2.shape);
    Graph.init();
    defer Graph.deinit();
    tensor2.trace();
    try std.testing.expect(Graph.AnyTensor.get((&tensor2)).last_op.ReduceOp.op == .Sum);
    try std.testing.expect(Graph.AnyTensor.get((&tensor2)).last_op.ReduceOp.x == Graph.AnyTensor.get((&tensor1)));
    try std.testing.expectEqual(Graph.AnyTensor.get((&tensor2)).last_op.ReduceOp.dims[0..tensor2.ndims].*, [_]bool{ true, true, false });
}

test "zip reduce" {
    const tensor1 = comptime UserTensor(.i32, .{ 2, 1, 4 }).full(2);
    const tensor2 = comptime UserTensor(.i32, .{ 2, 3, 1 }).full(3);
    const tensor3 = comptime tensor1.add(tensor2).sum(1);
    try std.testing.expectEqual([_]Dim{ .{ .constant = 2 }, .{ .constant = 1 }, .{ .constant = 4 } }, tensor3.shape);
    Graph.init();
    defer Graph.deinit();
    tensor3.trace();
    try std.testing.expect(Graph.AnyTensor.get((&tensor3)).last_op.ReduceOp.op == .Sum);
    // Anonymous intermediate tensor that stores tensor1 + tensor2
    const anon = Graph.AnyTensor.get((&tensor3)).last_op.ReduceOp.x;
    try std.testing.expect(anon.last_op.ZipOp.a.last_op.TypeOp.x == Graph.AnyTensor.get((&tensor1)));
    try std.testing.expect(anon.last_op.ZipOp.b.last_op.TypeOp.x == Graph.AnyTensor.get((&tensor2)));
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
