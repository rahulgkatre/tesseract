const std = @import("std");
const comptimePrint = std.fmt.comptimePrint;
const AnyTensor = @import("anytensor.zig").AnyTensor;
const utils = @import("utils.zig");
const ops = @import("ops.zig");
const dtypes = @import("dtypes.zig");
const OpTracker = @import("tracker.zig").OpTracker;
const OpGroupTracker = @import("tracker.zig").OpGroupTracker;

pub const Metadata = struct {
    op_tracker: OpTracker,
    op_group_tracker: OpGroupTracker,
    constant: bool,
    label: ?[]const u8,

    pub fn defaults() Metadata {
        return .{
            .op_tracker = OpTracker.init(.InitOp, .Empty, .{}, .{ .Empty = {} }),
            .op_group_tracker = .{},
            .constant = false,
            .label = null,
        };
    }
};

pub fn AsTensor(any: anytype) type {
    const Type = @TypeOf(any);
    switch (@typeInfo(Type)) {
        .Struct => return TensorType(any.dtype, any.shape[0..any.ndims]),
        .Int,
        .Float,
        .Bool,
        .ComptimeInt,
        .ComptimeFloat,
        => {},
        else => @compileError(comptimePrint("Cannot convert {any} to a tensor type", .{any})),
    }
    return Tensor(Type);
}

/// Used for wrapping immediate values in single size tensors with the same dtype as the current tensor
pub fn asTensor(any: anytype) AsTensor(any) {
    const Type = @TypeOf(any);
    return if (!isTensor(Type)) Tensor(Type).full(any) else any;
}

pub fn isTensor(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Pointer => |ptr| Tensor(ptr.child.ArrayType()) == ptr.child or AnyTensor == ptr.child,
        .Struct => Tensor(T.ArrayType()) == T or AnyTensor == T,
        else => false,
    };
}

pub fn range(
    comptime start: comptime_int,
    comptime stop: comptime_int,
) Tensor([stop - start]comptime_int) {
    return .{ .op_tracker = &OpTracker.init(.InitOp, .Range, {}, .{ .Range = .{
        .start = std.fmt.comptimePrint("{d}", .{start}),
        .stop = std.fmt.comptimePrint("{d}", .{stop}),
    } }) };
}

pub fn randLike(comptime other: anytype) AsTensor(other) {
    std.debug.assert(isTensor(@TypeOf(other)));
    return AsTensor(other).rand();
}

pub fn fullLike(comptime other: anytype, value: dtypes.ZigType(other.dtype)) AsTensor(other) {
    std.debug.assert(isTensor(@TypeOf(other)));
    return AsTensor(other).full(value);
}

pub fn TensorType(dtype: dtypes.DType, shape: anytype) type {
    var ArrayType = dtypes.ZigType(dtype);
    for (0..shape.len) |dim| {
        ArrayType = [shape[shape.len - dim - 1]]ArrayType;
    }
    return Tensor(ArrayType);
}

pub fn Tensor(comptime TensorArrayType: type) type {
    return extern struct {
        const Self = @This();
        const contiguous_strides: [ndims]u64 = utils.contiguousStrides(ndims, shape);
        const num_entries = utils.numEntries(ndims, shape);

        // All the functions for operations are implemented separately
        pub usingnamespace @import("functions.zig").Functions(Self);

        pub const dtype: dtypes.DType = utils.extractDType(TensorArrayType);
        pub const ndims: u8 = utils.extractNdims(TensorArrayType);
        pub const shape: [ndims]u64 = utils.extractShape(TensorArrayType);

        dtype: dtypes.DType = dtype,
        ndims: u8 = ndims,
        shape: *const [ndims]u64 = &shape,
        strides: *const [ndims]u64 = &contiguous_strides,
        offset: u64 = 0,
        meta: *const Metadata = &Metadata.defaults(),

        pub fn widen(comptime self: Self) AnyTensor {
            return @as(*const AnyTensor, @ptrCast(&self)).*;
        }

        pub fn ArrayType() type {
            var Child = dtypes.ZigType(dtype);
            for (0..ndims) |dim| {
                Child = [shape[ndims - dim - 1]]Child;
            }
            return Child;
        }

        /// Allows for negative dimension indexing to work by normalizing it to [0,ndims)
        pub fn signedToUnsignedDim(dim: i16) u8 {
            const value = if (dim < 0) ndims + dim else dim;
            if (value < 0 or value > ndims) {
                @compileError(comptimePrint(
                    "Dimension index {d} is out of bounds {d}",
                    .{ value, ndims },
                ));
            }
            return @intCast(value);
        }

        /// Determine if the stride pattern of the tensor defines a fully contiguous section of memory at runtime
        pub fn isContiguous(self: Self) bool {
            // Strides need to be decreasing unless its a broadcasted stride (0)
            var prev: u64 = std.math.maxInt(u64);
            for (self.strides) |stride| {
                if (stride > 0) {
                    if (stride > prev) {
                        return false;
                    }
                    prev = stride;
                }
            }
            return true;
        }

        pub fn storageSize(self: Self) u128 {
            // The storage size is 1 + last index calculated by the strides and shape
            // This is different from size() because there can be elements that are not
            // accessed by the stride pattern but are still contained in the underlying memory
            // An example is padding for alignment
            // Conversely, a convolution using stride tricks will index the same element multiple times
            // but the underlying memory has not increased in size
            // shape[d] - 1 is the last index in dimension d
            // Also incorporate the storage offset
            var _size: u128 = self.offset + 1;
            for (0..ndims) |d| {
                _size += (shape[d] - 1) * self.strides[d];
            }
            // The result is the size of the storage needed to visit all indices of the tensor
            return _size;
        }

        pub fn size(_: Self) u128 {
            return num_entries;
        }

        /// Supports negative indexing sugar (e.g. -1 = ndims - 1)
        pub fn dimSize(_: Self, d: i16) u64 {
            return shape[signedToUnsignedDim(d)];
        }

        /// Supports negative indexing sugar (e.g. -1 = ndims - 1)
        pub fn dimStride(self: Self, d: i16) u64 {
            return self.strides[signedToUnsignedDim(d)];
        }

        /// This is used to determine which higher level function the tensor is part of
        /// which is useful for finding a better gradient implementation if one exists
        /// than the one that would be found through simple backtracking of the graph.
        /// By default the op_group is null, so setting the op_group isn't necessary
        /// for simple functions with trivial gradients (e.g suogtraction)
        pub fn startGroup(self: Self, op_group_name: []const u8) Self {
            // OpGroup will not modify the computation graph so pass the other fields unmodified
            return .{
                .meta = &.{
                    .op_tracker = self.meta.op_tracker,
                    .op_group_tracker = self.meta.op_group_tracker.startGroup(op_group_name ++ if (self.meta.label) |label| "_" ++ label else ""),
                    .constant = self.meta.constant,
                    .label = self.meta.label,
                },
                .strides = self.strides,
                .offset = self.offset,
            };
        }

        pub fn foldConstIntoGroupOf(self: Self, group_of_tensor: anytype) Self {
            if (!self.meta.constant) {
                return self;
            }
            const tensor = asTensor(group_of_tensor);
            return .{
                .meta = &.{
                    .op_tracker = self.meta.op_tracker,
                    .op_group_tracker = self.meta.op_group_tracker.foldIntoGroup(tensor.meta.op_group_tracker.next),
                    .constant = self.meta.constant,
                    .label = self.meta.label,
                },
                .strides = self.strides,
                .offset = self.offset,
            };
        }

        /// End the current op_group by setting it to the outer op_group.
        /// Compile error if the current op_group is null.
        pub fn endGroup(self: *const Self) Self {
            return .{
                .meta = &.{
                    .op_tracker = self.meta.op_tracker,
                    .op_group_tracker = self.meta.op_group_tracker.endGroup(),
                    .constant = self.meta.constant,
                    .label = self.meta.label,
                },
                .strides = self.strides,
                .offset = self.offset,
            };
        }

        pub fn assignLabel(self: Self, comptime label: []const u8) Self {
            return .{
                .meta = &.{
                    .op_tracker = self.meta.op_tracker,
                    .op_group_tracker = self.meta.op_group_tracker,
                    .constant = self.meta.constant,
                    .label = label,
                },
                .strides = self.strides,
                .offset = self.offset,
            };
        }

        //
        // Initialization functions (init ops)
        //

        /// Used to mark a tensor as an input to a graph,
        /// codegen will make this an argument of the function
        /// A label can be given to make two tensors of the same shape/dtype
        /// correspond to different arrays at runtime (e.g. for two input images )
        pub fn input(comptime label: ?[]const u8) Self {
            return .{
                .meta = &.{
                    .op_tracker = OpTracker.init(
                        .InitOp,
                        .Input,
                        .{},
                        .{ .Input = {} },
                    ),
                    .op_group_tracker = OpGroupTracker{},
                    .constant = false,
                    .label = label,
                },
            };
        }

        /// Used to mark a tensor as a learnable parameter,
        /// codegen will make this an argument of the function,
        /// gradients can be accumulated for it,
        /// and optimizers can detect it,
        pub fn param(label: []const u8) Self {
            return .{ .meta = &.{
                .op_tracker = OpTracker.init(
                    .InitOp,
                    .Parameter,
                    .{},
                    .{ .Parameter = {} },
                ),
                .op_group_tracker = OpGroupTracker{},
                .constant = false,
                .label = label,
            } };
        }

        /// Fill a tensor with a value
        /// By default, full tensors will be constant folded in codegen
        /// unless they are marked as requires_grad
        pub fn full(comptime value: dtypes.ZigType(dtype)) Self {
            return .{
                .meta = &.{
                    .op_tracker = OpTracker.init(
                        .InitOp,
                        .Full,
                        .{},
                        .{ .Full = .{ .value = std.fmt.comptimePrint("{}", .{value}) } },
                    ),
                    .op_group_tracker = OpGroupTracker{},
                    .constant = true,
                    .label = null,
                },
            };
        }

        /// Fill a tensor with random generated numbers
        /// By default, random tensors will be constant folded in codegen
        /// unless they are marked as requires_grad
        /// Do not use this for random initialization of parameters!
        /// Note that some device backends do not support this
        pub fn random(label: []const u8) Self {
            std.debug.assert(dtypes.isFloat(dtype));
            return .{ .meta = &.{
                .op_tracker = &OpTracker.init(
                    .InitOp,
                    .Rand,
                    .{},
                    .{ .Rand = {} },
                ),
                .op_group_tracker = .{},
                .constant = false,
                .label = label,
            } };
        }

        //
        // Type / shape manipulation functions
        //

        ///Cast an array of a datatype to another datatype
        pub fn cast(comptime a: Self, comptime new_dtype: dtypes.DType) TensorType(new_dtype, shape) {
            if (new_dtype != a.dtype) {
                return .{
                    .meta = &.{
                        .op_tracker = OpTracker.init(.TypeOp, .Cast, .{&a.widen()}, .{ .Cast = {} }),
                        .op_group_tracker = a.meta.op_group_tracker.keepGroup(),
                        .constant = a.meta.constant,
                        .label = a.meta.label,
                    },
                    .strides = a.strides,
                    .offset = a.offset,
                };
            } else {
                return a;
            }
        }

        /// Make an array contguous (a full new copy) if it is not already
        pub fn contiguous(comptime a: Self) Self {
            if (a.isContiguous()) return a;
            return Self{ .meta = &Metadata{ .constant = false, .label = a.meta.label, .op_group_tracker = a.meta.op_group_tracker.keepGroup(), .op_tracker = .{ .TypeOp = .{
                .in = .{&a.widen()},
                .args = .{ .Contiguous = {} },
                .op = .Contiguous,
            } } } };
        }

        const DimRange = struct {
            from: i16 = 0,
            to: i16 = -1,
        };

        /// Expand a tensor along 1 or more dimensions with size 1 and stride 0
        /// The new shape must broadcast with the old shape
        pub fn expand(a: Self, comptime new_shape: anytype) TensorType(dtype, utils.broadcastShape(shape, new_shape)) {
            const Out = TensorType(dtype, utils.broadcastShape(shape, new_shape));
            if (Self == Out) {
                return a;
            }
            const bc_strides: [new_shape.len]u64 = blk: {
                var bc_strides: [new_shape.len]u64 = undefined;
                for (0..new_shape.len) |i| {
                    bc_strides[new_shape.len - i - 1] = if (i >= ndims) 0 else a.strides[ndims - i - 1];
                }
                break :blk bc_strides;
            };

            return a.view(
                new_shape,
                bc_strides,
                a.offset,
            );
        }

        pub fn Flatten(comptime dim_range: DimRange) type {
            const from = signedToUnsignedDim(dim_range.from);
            const to = signedToUnsignedDim(dim_range.to);
            if (from == to) {
                return @This();
            }
            var new_shape: [ndims - (to - from)]u64 = undefined;
            new_shape[from] = 1;
            for (0..ndims) |d| {
                if (d < from or d > to) {
                    new_shape[d] = shape[d];
                } else {
                    new_shape[from] *= shape[d];
                }
            }
            return Reshape(new_shape);
        }
        /// Flatten a range of dims, collapsing them to 1 dimension
        pub fn flatten(comptime a: Self, comptime dim_range: DimRange) Flatten(dim_range) {
            return a.reshape(Flatten(dim_range).shape);
        }

        const PaddingMode = union(ops.PadModeEnum) {
            constant: dtypes.ZigType(dtype),
            reflect: void,
            replicate: void,
            circular: void,
        };

        pub fn Pad(padding: anytype) type {
            const padded_dims = padding.len;
            const padding_tuple: [padded_dims][2]u64 = padding;
            std.debug.assert(padded_dims <= ndims);
            var new_shape: [ndims]usize = shape;
            for (0..padded_dims) |dim| {
                new_shape[ndims - dim - 1] += padding_tuple[dim][0] + padding_tuple[dim][1];
            }
            return TensorType(dtype, new_shape);
        }
        pub fn pad(comptime a: Self, comptime padding: anytype, comptime mode: PaddingMode) Pad(padding) {
            return .{
                .meta = &.{
                    .op_tracker = OpTracker.init(.TypeOp, .Pad, .{&a.widen()}, .{
                        .Pad = .{
                            .padding = &padding,
                            .mode = switch (mode) {
                                .constant => |constant| .{ .constant = comptimePrint("{}", .{constant}) },
                                else => mode,
                            },
                        },
                    }),
                    .op_group_tracker = a.meta.op_group_tracker.keepGroup(),
                    .constant = false,
                    .label = a.meta.label,
                },
            };
        }

        pub fn Permute(comptime perm: [ndims]u8) type {
            return Reshape(utils.arrayPermute(u64, ndims, shape, perm));
        }
        /// Permute the dimensions of the tensor. A valid permutation must contain
        /// values from 0 to ndims and each value must appear exactly once.
        pub fn permute(comptime a: Self, comptime perm: [ndims]u8) Permute(perm) {
            return a.view(
                Permute(perm).shape,
                utils.arrayPermute(u64, ndims, a.strides[0..ndims].*, perm),
                a.offset,
            );
        }

        pub fn Reshape(comptime new_shape: anytype) type {
            const Type = TensorType(dtype, new_shape);
            std.debug.assert(Type.num_entries == num_entries);
            return Type;
        }
        /// Change the shape of the tensor. This changes the type too.
        pub fn reshape(comptime a: Self, comptime new_shape: anytype) Reshape(new_shape) {
            return a.contiguous().view(new_shape, Reshape(new_shape).contiguous_strides, a.offset);
        }

        pub fn Squeeze(comptime dim: i16) type {
            if (shape[signedToUnsignedDim(dim)] != 1) {
                @compileError("Cannot squeeze as dimension size is not 1");
            }
            return Reshape(utils.arrayDelete(ndims, shape, signedToUnsignedDim(dim)));
        }
        /// Remove a dim of size 1 from the shape of the tensor.
        pub fn squeeze(comptime a: Self, comptime dim: i16) Squeeze(dim) {
            return a.view(
                Squeeze(dim).shape,
                utils.arrayDelete(ndims, a.strides[0..ndims].*, signedToUnsignedDim(dim)),
                a.offset,
            );
        }

        pub fn Transpose(comptime dim1: i16, comptime dim2: i16) type {
            const norm1 = signedToUnsignedDim(dim1);
            const norm2 = signedToUnsignedDim(dim2);
            var new_shape = shape;
            new_shape[norm1] = shape[norm2];
            new_shape[norm2] = shape[norm1];
            return Reshape(new_shape);
        }
        /// Transpose two dimensions of the tensor. Similar to permute, but only for two dimensions.
        pub fn transpose(comptime a: Self, comptime dim1: i16, comptime dim2: i16) Transpose(dim1, dim2) {
            const norm1 = signedToUnsignedDim(dim1);
            const norm2 = signedToUnsignedDim(dim2);
            if (norm1 != norm2) {
                var new_strides = a.strides[0..a.ndims].*;
                new_strides[norm1] = a.strides[norm2];
                new_strides[norm2] = a.strides[norm1];
                return a.view(
                    Transpose(norm1, norm2).shape,
                    new_strides,
                    a.offset,
                );
            } else {
                return a;
            }
        }
        /// Shorthand for transposing rightmost dimensions of tensor
        pub fn T(comptime a: Self) Transpose(-2, -1) {
            return a.transpose(-2, -1);
        }

        fn Unsqueeze(comptime dim: i16) type {
            return Reshape(utils.arrayInsert(ndims, shape, signedToUnsignedDim(dim), 1));
        }
        /// Insert a dim of size 1 into the shape of the tensor.
        pub fn unsqueeze(comptime a: Self, comptime dim: i16) Unsqueeze(dim) {
            return a.view(
                Unsqueeze(dim).shape,
                utils.arrayInsert(ndims, a.strides[0..ndims].*, signedToUnsignedDim(dim), 0),
                a.offset,
            );
        }

        /// Changes the shape and stride of the tensor to change how the underlying memory is accessed.
        /// Powerful enough to be used to implement any reshaping or windowing operation on a tensor.
        /// There are guardrails to prevent out of bounds access into underlying memory!
        pub fn view(comptime a: Self, comptime new_shape: anytype, comptime new_strides: [new_shape.len]u64, new_offset: u64) TensorType(dtype, new_shape) {
            var out = TensorType(dtype, new_shape){
                .meta = &.{
                    .op_tracker = OpTracker.init(.TypeOp, .View, .{&a.widen()}, .{ .View = {} }),
                    .op_group_tracker = a.meta.op_group_tracker.keepGroup(),
                    .constant = a.meta.constant,
                    .label = a.meta.label,
                },
                .strides = &new_strides,
                .offset = new_offset,
            };
            if (out.storageSize() > a.storageSize()) {
                @compileError(comptimePrint(
                    \\View indexes elements outside defined storage
                    \\Old shape: {any}
                    \\Old strides: {any}
                    \\Old storage offset: {}
                    \\Old storage size: {}
                    \\New shape: {any}
                    \\New strides: {any}
                    \\New storage offset: {}
                    \\New storage size: {}
                , .{
                    a.shape[0..a.ndims],
                    a.strides[0..a.ndims],
                    a.offset,
                    a.storageSize(),
                    out.shape[0..out.ndims],
                    out.strides[0..out.ndims],
                    out.offset,
                    out.storageSize(),
                }));
            }
            return out;
        }

        ///Apply an elementwise unary operation
        pub fn unaryFn(a: Self, comptime op: ops.UnaryOp) Self {
            return .{
                .meta = &.{
                    .op_tracker = OpTracker.init(.UnaryOp, op, .{&a.widen()}, {}),
                    .op_group_tracker = a.meta.op_group_tracker.keepGroup(),
                    .constant = a.meta.constant,
                    .label = a.meta.label,
                },
                .strides = a.strides,
                .offset = a.offset,
            };
        }

        pub fn BinaryFnResult(comptime b: anytype, comptime op: ops.BinaryOp) type {
            const Other = AsTensor(b);
            const bc_shape = utils.broadcastShape(shape, Other.shape);
            const new_dtype: dtypes.DType = switch (op) {
                .Eq, .Lt => .bool,
                else => dtypes.resultDType(Self.dtype, Other.dtype),
            };
            return TensorType(new_dtype, bc_shape);
        }
        /// Apply an elementwise binary operation on two arrays, with broadcasting
        /// a and b must have the same "dtype class" meaning both must be float, bool, or int
        /// though different sizes are allowed.
        pub fn binaryFn(a: Self, b: anytype, comptime op: ops.BinaryOp) BinaryFnResult(b, op) {
            const tb = asTensor(b).foldConstIntoGroupOf(a);
            const bc_shape = utils.broadcastShape(shape[0..ndims].*, tb.shape[0..tb.ndims].*);

            const ta_bc = a.expand(bc_shape);
            const tb_bc = tb.expand(bc_shape);

            const Result = BinaryFnResult(tb, op);
            return Result{
                .strides = &Result.contiguous_strides,
                .offset = 0,
                .meta = &.{
                    .op_tracker = OpTracker.init(.BinaryOp, op, .{ &ta_bc.widen(), &tb_bc.widen() }, {}),
                    .op_group_tracker = a.meta.op_group_tracker.keepGroup(),
                    .constant = a.meta.constant and tb.meta.constant,
                    .label = null,
                },
            };
        }

        pub fn ReduceFnResult(comptime reduce_dims: anytype) type {
            const reduced_shape: [ndims]u64 = switch (@typeInfo(@TypeOf(reduce_dims))) {
                .ComptimeInt, .Int => blk: {
                    const dim = signedToUnsignedDim(reduce_dims);
                    if (dim < 0 or dim >= ndims) {
                        @compileError("Dimension index for single dimension reduce is out of bounds");
                    }
                    var reduced_shape: [ndims]u64 = undefined;
                    @memcpy(&reduced_shape, &shape);
                    reduced_shape[dim] = 1;
                    break :blk reduced_shape;
                },
                .Null, .Void => blk: {
                    break :blk .{1} ** ndims;
                },
                else => blk: {
                    const dims = reduce_dims;
                    if (dims.len > ndims) {
                        @compileError("Length of dimension index array for multi dimension reduce is out of bounds");
                    }
                    var reduce_dim_mask: [ndims]bool = [_]bool{false} ** ndims;
                    var reduced_shape: [ndims]u64 = undefined;
                    @memcpy(&reduced_shape, &shape);
                    for (0..dims.len) |d| {
                        const norm = signedToUnsignedDim(d);
                        if (reduce_dim_mask[norm]) {
                            @compileError("Cannot reuse dimension index for multi dimensional reduce");
                        }
                        reduce_dim_mask[d] = true;
                        reduced_shape[d] = 1;
                    }
                    break :blk reduced_shape;
                },
            };
            return TensorType(dtype, reduced_shape);
        }
        /// Perform a reduction across 1 or more (or all) dimensions of a tensor.
        /// Dimensions to reduce can be passed as a int for 1 dim, tuple for multiple dims, or null/void for all dims
        pub fn reduceFn(
            a: Self,
            comptime op: ops.ReduceOp,
            comptime reduce_dims: anytype,
        ) ReduceFnResult(reduce_dims) {
            // Use u16 here because []const u8 shows up as a string
            const reduce_dims_array: []const u16 = switch (@typeInfo(@TypeOf(reduce_dims))) {
                .ComptimeInt, .Int => &[1]u16{signedToUnsignedDim(reduce_dims)},
                .Null, .Void => @as([ndims]u16, std.simd.iota(u16, ndims))[0..],
                else => &reduce_dims,
            };
            const reduce_dim_mask: [ndims]bool = switch (@typeInfo(@TypeOf(reduce_dims))) {
                .ComptimeInt, .Int => blk: {
                    var tmp_mask: [ndims]bool = [_]bool{false} ** ndims;
                    const dim = reduce_dims;
                    tmp_mask[signedToUnsignedDim(dim)] = true;
                    break :blk tmp_mask;
                },
                .Null, .Void => [_]bool{true} ** ndims,
                else => blk: {
                    var tmp_mask: [ndims]bool = [_]bool{false} ** ndims;
                    for (reduce_dims) |dim| {
                        tmp_mask[signedToUnsignedDim(dim)] = true;
                    }
                    break :blk tmp_mask;
                },
            };

            return .{
                .meta = &.{
                    .op_tracker = OpTracker.init(.ReduceOp, op, .{&a.widen()}, .{ .dims = reduce_dims_array, .mask = &reduce_dim_mask }),
                    .op_group_tracker = a.meta.op_group_tracker.keepGroup(),
                    .constant = false,
                    .label = a.meta.label,
                },
            };
        }

        pub fn MatMul(b: anytype) type {
            const TensorB = AsTensor(b);

            // Matrix multiplication invariant
            // (n x m1) matmul (m2 x p) -> (n x p) iff m1 = m2
            // otherwise matmul is invalid, compile error
            const n = if (ndims == 1) 1 else shape[ndims - 2];
            const m = shape[ndims - 1];
            const b_m = if (TensorB.ndims == 1) 1 else TensorB.shape[TensorB.ndims - 2];
            const p = TensorB.shape[TensorB.ndims - 1];

            if (m == b_m) {
                const mm_ndims = @max(ndims, TensorB.ndims);
                var mm_shape: [mm_ndims]u64 = undefined;
                // Expanding check, look only at batch dimensions (everything before last 2 dimensions)
                const mm_bc_shape: [mm_ndims - 2]u64 = utils.broadcastShape(shape[0 .. ndims - 2].*, TensorB.shape[0 .. TensorB.ndims - 2].*);
                @memcpy(mm_shape[0 .. mm_ndims - 2], &mm_bc_shape);
                mm_shape[mm_ndims - 2] = n;
                mm_shape[mm_ndims - 1] = p;
                return TensorType(dtype, mm_shape);
            }
            @compileError(comptimePrint(
                \\Tensors have incompatible shapes for batch matrix multiplication
                \\Tensor A: {any}
                \\Tensor B: {any}
                \\n = {d}, m = {d}, other_m = {d}, p = {d}
            , .{ shape, TensorB.shape, n, m, b_m, p }));
        }
        pub fn matmul(a: Self, comptime b: anytype) MatMul(b) {
            return a.startGroup("matmul")
                .unsqueeze(a.ndims - 1)
                .mul(b.transpose(b.ndims - 2, b.ndims - 1).unsqueeze(b.ndims - 2))
                .sum(a.ndims)
                .squeeze(a.ndims)
                .endGroup();
        }

        pub fn Where(comptime true_value: anytype, comptime false_value: anytype) type {
            const True = AsTensor(true_value);
            const False = AsTensor(false_value);
            std.debug.assert(True.dtype == False.dtype);
            const bc_value_shape = utils.broadcastShape(True.shape, False.shape);
            const bc_result_shape = utils.broadcastShape(shape, bc_value_shape);
            return TensorType(True.dtype, bc_result_shape);
        }
        /// Conditional elementwise operator
        /// out[i] = if (mask[i]) true_value[i] else false_value[i]
        /// Supports broadcasting between all 3 tensors, but true value and false value are broadcasted together first and must also have the same dtype
        pub fn where(mask: dtypes.BoolTensor(Self), true_value: anytype, false_value: anytype) Where(true_value, false_value) {
            const Out = Where(true_value, false_value);
            const mask_expand = mask.expand(Out.shape);
            const true_expand = asTensor(true_value).foldConstIntoGroupOf(mask).expand(Out.shape);
            const false_expand = asTensor(false_value).foldConstIntoGroupOf(mask).expand(Out.shape);
            return .{
                .meta = &.{
                    .op_tracker = OpTracker.init(.TernaryOp, .Where, .{ &mask_expand.widen(), &true_expand.widen(), &false_expand.widen() }, {}),
                    .op_group_tracker = mask.meta.op_group_tracker.keepGroup(),
                    .constant = false,
                    .label = null,
                },
            };
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
    };
}

test "permute" {
    const tensor1 = comptime Tensor([2][3][4]i32).full(0);
    const tensor2 = comptime tensor1.permute(.{ 0, 2, 1 });
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 4, 3 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 12, 1, 4 }, tensor2.strides[0..tensor2.ndims]);
}

test "view" {
    const tensor1 = comptime Tensor([2][3][4]i32).full(0);
    const tensor2 = comptime tensor1.reshape(.{ 12, 2 });
    const tensor3 = comptime tensor2.reshape(.{24});
    try std.testing.expectEqualSlices(u64, &[_]u64{ 12, 2 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 1 }, tensor2.strides[0..tensor2.ndims]);
    try std.testing.expectEqualSlices(u64, &[_]u64{24}, tensor3.shape[0..tensor3.ndims]);
    try std.testing.expectEqualSlices(u64, &[_]u64{1}, tensor3.strides[0..tensor3.ndims]);
}

test "as strided" {
    // Based on example from https://pytorch.org/docs/stable/generated/torch.as_strided.html
    const tensor1 = comptime TensorType(.i32, .{ 3, 3 }).full(0);
    const tensor2 = comptime tensor1.view(.{ 2, 2 }, .{ 1, 2 }, 0);

    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 2 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expectEqual(false, tensor2.isContiguous());

    const test_indices = [_][2]u64{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 } };
    const expected_flat_indices1 = &[_]u64{ 0, 2, 1, 3 };
    for (expected_flat_indices1, test_indices) |expected_flat_i, test_i| {
        try std.testing.expectEqual(expected_flat_i, utils.ravelMultiIndex(tensor2.ndims, tensor2.strides[0..tensor2.ndims].*, tensor2.offset, test_i));
    }

    const tensor3 = comptime tensor1.view(.{ 2, 2 }, .{ 1, 2 }, 1);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 2 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expectEqual(false, tensor2.isContiguous());

    const expected_flat_indices2 = &[_]u64{ 1, 3, 2, 4 };
    for (expected_flat_indices2, test_indices) |expected_flat_i, test_i| {
        try std.testing.expectEqual(expected_flat_i, utils.ravelMultiIndex(tensor3.ndims, tensor3.strides[0..tensor3.ndims].*, tensor3.offset, test_i));
    }
}

test "unaryFn" {
    const tensor1 = comptime Tensor([2][3][4]i32).full(3);
    const tensor2 = comptime tensor1.neg();
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 4 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expect(tensor2.meta.op_tracker.UnaryOp.op == .Neg);
    try std.testing.expectEqual(tensor2.meta.op_tracker.UnaryOp.in[0].narrow().*, tensor1);
}

test "binaryFn" {
    const tensor1 = comptime Tensor([2][1][4]i32).full(2);
    const tensor2 = comptime Tensor([3][1]i32).full(3);
    const tensor3 = comptime tensor1.add(tensor2);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 4 }, tensor3.shape[0..tensor3.ndims]);
    try std.testing.expect(tensor3.meta.op_tracker.BinaryOp.op == .Add);
    try std.testing.expectEqualDeep(tensor3.meta.op_tracker.BinaryOp.in[0].meta.op_tracker.TypeOp.in[0].narrow().*, tensor1);
    try std.testing.expectEqualDeep(tensor3.meta.op_tracker.BinaryOp.in[1].meta.op_tracker.TypeOp.in[0].narrow().*, tensor2);
}

test "reduce" {
    const tensor1 = comptime Tensor([2][3][4]i32).full(5);
    const tensor2 = comptime tensor1.sum(1);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 1, 4 }, tensor2.shape[0..tensor1.ndims]);
    try std.testing.expect(tensor2.meta.op_tracker.ReduceOp.op == .Add);
    try std.testing.expectEqual(tensor2.meta.op_tracker.ReduceOp.in[0].narrow().*, tensor1);
    try std.testing.expectEqual(tensor2.meta.op_tracker.ReduceOp.args.mask[0..tensor2.ndims].*, ([_]bool{ false, true, false }));
}

test "multiple dim reduce" {
    const tensor1 = comptime Tensor([2][3][4]i32).full(5);
    const tensor2 = comptime tensor1.sum(.{ 0, 1 });
    try std.testing.expectEqualSlices(u64, &[_]u64{ 1, 1, 4 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expect(tensor2.meta.op_tracker.ReduceOp.op == .Add);
    try std.testing.expectEqual(tensor2.meta.op_tracker.ReduceOp.in[0].narrow().*, tensor1);
    try std.testing.expectEqualDeep(tensor2.meta.op_tracker.ReduceOp.args.mask[0..tensor2.ndims], &[_]bool{ true, true, false });
}

test "binaryFn reduce" {
    const tensor1 = comptime Tensor([2][1][4]i32).full(2);
    const tensor2 = comptime Tensor([2][3][1]i32).full(3);
    const tensor3 = comptime tensor1.add(tensor2).sum(1);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 1, 4 }, tensor3.shape[0..tensor3.ndims]);
    try std.testing.expect(tensor3.meta.op_tracker.ReduceOp.op == .Add);
    // Anonymous intermediate tensor that stores tensor1 + tensor2
    const anon = tensor3.meta.op_tracker.ReduceOp.in[0];
    try std.testing.expectEqualDeep(anon.meta.op_tracker.BinaryOp.in[0].meta.op_tracker.TypeOp.in[0].narrow().*, tensor1);
    try std.testing.expectEqualDeep(anon.meta.op_tracker.BinaryOp.in[1].meta.op_tracker.TypeOp.in[0].narrow().*, tensor2);
}

test "cast" {
    const tensor1 = comptime TensorType(.bool, .{3}).full(true);
    try std.testing.expect(tensor1.dtype == .bool);
    const tensor2 = comptime tensor1.cast(.i32);
    try std.testing.expect(tensor2.dtype == .i32);
    const tensor3 = comptime tensor2.cast(.i8);
    try std.testing.expect(tensor3.dtype == .i8);
    const tensor4 = comptime tensor3.cast(.f16);
    try std.testing.expect(tensor4.dtype == .f16);
    const tensor5 = comptime tensor4.cast(.f32);
    try std.testing.expect(tensor5.dtype == .f32);
}

fn fn1() Tensor([2][1][4]i32) {
    const tensor1 = Tensor([2][1][4]i32).full(1);
    const tensor2 = Tensor([2][3][1]i32).full(2);
    const tensor3 = tensor1.add(tensor2).sum(1);
    return tensor3;
}

fn fn2(input: anytype) Tensor([2][3][4]i32) {
    return comptime blk: {
        const tensor4 = Tensor([2][1][4]i32).full(4);
        const tensor5 = Tensor([2][3][1]i32).full(5);
        const tensor6 = tensor4.mul(input).sum(1).add(tensor5);
        break :blk tensor6;
    };
}

test "tensors from functions" {
    _ = comptime blk: {
        const tensor3 = fn1();
        const tensor6 = fn2(tensor3);
        break :blk tensor6;
    };
}

test "transpose" {
    const tensor1 = comptime Tensor([2][1][4]i32).full(1);
    const tensor2 = comptime tensor1.T();
    const ndims = tensor1.ndims;
    try std.testing.expectEqualDeep(tensor2, comptime tensor1.transpose(-2, -1));
    try std.testing.expectEqualDeep(tensor1.shape[0..ndims], comptime tensor2.T().shape[0..ndims]);
    try std.testing.expectEqualDeep(tensor1.strides[0..ndims], comptime tensor2.T().strides[0..ndims]);
}

test "runtime" {
    const tensor1 = comptime Tensor([2][1][4]i32).full(1);
    const tensor2 = comptime tensor1.T();
    _ = tensor2;
}

test "array type" {
    const Tensor1 = TensorType(.i32, .{ 2, 3, 4 });
    try std.testing.expectEqual(Tensor1.ArrayType(), [2][3][4]i32);
}

test "bf16" {
    const Tensor1 = comptime Tensor([2][3]dtypes.bf16);
    try std.testing.expectEqual(Tensor1.dtype, .bf16);
}

test "padding" {
    // https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    const t4d = comptime Tensor([3][3][4][2]f32){};
    const p1d = comptime .{.{ 1, 1 }};
    const out1 = comptime t4d.pad(p1d, .{ .constant = 0 });
    try std.testing.expectEqualDeep(@TypeOf(out1).shape, .{ 3, 3, 4, 4 });

    const p2d = comptime .{ .{ 1, 1 }, .{ 2, 2 } };
    const out2 = comptime t4d.pad(p2d, .{ .constant = 0 });
    try std.testing.expectEqualDeep(@TypeOf(out2).shape, .{ 3, 3, 8, 4 });

    const p3d = comptime .{ .{ 0, 1 }, .{ 2, 1 }, .{ 3, 3 } };
    const out3 = comptime t4d.pad(p3d, .{ .constant = 0 });
    try std.testing.expectEqualDeep(@TypeOf(out3).shape, .{ 3, 9, 7, 3 });
}

test "unique input" {
    const tensor1 = comptime Tensor([2][1][4]i32).input("tensor1");
    const tensor2 = comptime Tensor([2][1][4]i32).input("tensor2");
    const out = comptime tensor1.add(tensor2);
    const writer = std.io.Writer(std.fs.File, std.fs.File.WriteError, std.fs.File.write){ .context = std.io.getStdOut() };
    try @import("utils.zig").dataflowViz(&[_]*const AnyTensor{&out.widen()}, writer, std.testing.allocator, true);
}
