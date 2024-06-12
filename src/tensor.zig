const std = @import("std");
const AnyTensor = @import("anytensor.zig").AnyTensor;
const utils = @import("utils.zig");
const ops = @import("ops.zig");
const dtypes = @import("dtypes.zig");
const graph = @import("graph.zig");

const meta = @import("meta.zig");
const Metadata = meta.Metadata;

// =============================================================================
// Type level utilities specifically for Tensor types
// Equivalents for @as, @TypeOf, @Type
// =============================================================================

/// Like @TypeOf but any must either be a scalar (to construct a scalar Tensor)
/// or a tensor (to reconstruct the Tensor from its dtype shape and ndims)
pub fn TensorTypeOf(any: anytype) type {
    var Type = @TypeOf(any);
    switch (@typeInfo(Type)) {
        .Pointer => |info| Type = info.child,
        else => {},
    }
    switch (@typeInfo(Type)) {
        .Struct => {},
        .Int,
        .Float,
        .Bool,
        .ComptimeInt,
        .ComptimeFloat,
        => return Tensor(Type),
        else => @compileError(std.fmt.comptimePrint("Cannot convert {any} to a tensor type", .{Type})),
    }
    return TensorType(any.dtype, any.shape[0..any.ndims]);
}

/// Like @as but for casting to matching Tensor type
/// Used for wrapping immediate values in single size tensors with the same dtype as the current tensor
/// Will cause a compile error if any is not a Tensor or a scalar number.
pub fn asTensor(any: anytype) TensorTypeOf(any) {
    @setEvalBranchQuota(std.math.maxInt(u32));
    return switch (@typeInfo(@TypeOf(any))) {
        .Pointer => any.*,
        .Struct => any,
        .Int, .Float, .Bool, .ComptimeInt, .ComptimeFloat => TensorTypeOf(any).full(any),
        else => unreachable,
    };
}

/// Test if a type is a Tensor type
pub fn isTensorType(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Struct => Tensor(T.ArrayType()) == T,
        else => false,
    };
}

/// Like @Type but for constructing a Tensor type from its "type info"
/// Given dtype and shape, recreate the array type and return the corresponing Tensor type
pub fn TensorType(dtype: dtypes.DType, shape: anytype) type {
    var ArrayType = dtypes.ZigType(dtype);
    for (0..shape.len) |dim| {
        ArrayType = [shape[shape.len - dim - 1]]ArrayType;
    }
    return Tensor(ArrayType);
}

pub fn TensorTuple(comptime tensors: anytype) type {
    comptime var types: [tensors.len]type = undefined;
    for (tensors, 0..) |in, i| {
        types[i] = TensorTypeOf(in);
    }
    return std.meta.Tuple(&types);
}

pub fn Tensor(comptime TensorArrayType: type) type {
    return extern struct {
        const Self = @This();
        pub const contiguous_strides: [ndims]u64 = utils.contiguousStrides(&shape);
        pub const num_entries = utils.numEntries(ndims, shape);

        // All the functions for operations (that do not modify metadata directly)
        // are implemented in another file
        pub usingnamespace @import("functions.zig");

        pub const dtype: dtypes.DType = utils.extractDType(TensorArrayType);
        pub const ndims: u8 = utils.extractNdims(TensorArrayType);
        pub const shape: [ndims]u64 = utils.extractShape(TensorArrayType);

        dtype: dtypes.DType = dtype,
        ndims: u8 = ndims,
        shape: *const [ndims]u64 = &shape,
        strides: *const [ndims]u64 = &contiguous_strides,
        offset: u64 = 0,
        meta: *const Metadata,

        pub fn toAny(comptime self: Self) *const AnyTensor {
            return @as(*const AnyTensor, @ptrCast(&self));
        }

        pub fn ArrayType() type {
            var Child = dtypes.ZigType(dtype);
            for (0..ndims) |dim| {
                Child = [shape[ndims - dim - 1]]Child;
            }
            return Child;
        }

        /// Determine if the stride pattern of the tensor defines a fully contiguous section of memory at runtime
        pub fn isContiguous(self: Self) bool {
            // Strides need to be decreasing unless its a broadcasted stride (0)
            return utils.isContiguous(self.strides);
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

        /// Allows for negative dimension indexing to work by normalizing it to [0,ndims)
        fn signedToUnsignedDim(dim: i16) u8 {
            return utils.signedToUnsignedDim(Self.ndims, dim);
        }

        /// Supports negative indexing sugar (e.g. -1 = ndims - 1)
        pub fn dimSize(_: Self, d: i16) u64 {
            return shape[signedToUnsignedDim(d)];
        }

        /// Supports negative indexing sugar (e.g. -1 = ndims - 1)
        pub fn dimStride(self: Self, d: i16) u64 {
            return self.strides[signedToUnsignedDim(d)];
        }

        //
        // Initialization functions (init ops)
        //

        /// Create an empty tensor (i.e. allocate).
        /// Do not make any assumptions about data in the empty
        pub fn empty() Self {
            return .{
                .meta = &meta.Metadata.init(.{
                    .InitOp = .{
                        .op = .empty,
                        .args = .{ .empty = {} },
                    },
                }, false, null),
            };
        }

        /// Fill a tensor with a value
        /// By default, full tensors will be constant folded in codegen
        /// unless they are marked as requires_grad
        pub fn full(comptime value: dtypes.ZigType(dtype)) Self {
            return .{
                .meta = &meta.Metadata.init(.{
                    .InitOp = .{
                        .op = .full,
                        .args = .{ .full = std.fmt.comptimePrint("{}", .{value}) },
                    },
                }, true, null),
            };
        }

        /// Used to mark a tensor as an input to a graph,
        /// codegen will make this an argument of the function
        /// A label can be given to make two tensors of the same shape/dtype
        /// correspond to different arrays at runtime (e.g. for two input images )
        pub fn input(comptime label: ?[]const u8) Self {
            return .{
                .meta = &meta.Metadata.init(
                    .{
                        .InitOp = .{
                            .op = .input,
                            .args = .{ .input = {} },
                        },
                    },
                    false,
                    label,
                ),
            };
        }

        /// Used to mark a tensor as a learnable parameter,
        /// codegen will make this an argument of the function,
        /// gradients can be accumulated for it,
        /// and optimizers can detect it,
        pub fn param(label: []const u8) Self {
            return .{
                .meta = &meta.Metadata.init(.{
                    .InitOp = .{
                        .op = .parameter,
                        .args = .{ .parameter = {} },
                    },
                }, false, label),
            };
        }

        /// Fill a tensor with random generated numbers
        /// By default, random tensors will be constant folded in codegen
        /// unless they are marked as requires_grad
        /// Do not use this for random initialization of parameters!
        /// Note that some device backends do not support this
        pub fn random(label: []const u8) Self {
            std.debug.assert(dtypes.isFloat(dtype));
            return .{
                .meta = &meta.Metadata.init(.{
                    .InitOp = .{
                        .op = .random,
                        .args = .{ .random = {} },
                    },
                }, false, label),
            };
        }

        //
        // Type / shape manipulation functions
        //

        ///Cast an array of a datatype to another datatype
        pub fn cast(comptime self: Self, comptime new_dtype: dtypes.DType) TensorType(new_dtype, shape) {
            if (new_dtype != self.dtype) {
                return .{
                    .meta = &meta.Metadata.init(.{
                        .TypeOp = .{
                            .src = .{self.toAny()},
                            .op = .cast,
                            .args = .{ .cast = {} },
                        },
                    }, self.meta.constant, self.meta.label),
                    .strides = self.strides,
                    .offset = self.offset,
                };
            } else {
                return self;
            }
        }

        /// Make an array contguous (a full new copy) if it is not already
        pub fn contiguous(comptime self: Self) Self {
            if (self.isContiguous()) return self;
            return .{
                .meta = &meta.Metadata.init(.{
                    .TypeOp = .{
                        .src = .{self.toAny()},
                        .op = .contiguous,
                        .args = .{ .contiguous = {} },
                    },
                }, false, self.meta.label),
            };
        }

        const PadMode = union(ops.TypeOp.Args.Pad.Mode) {
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
        pub fn pad(comptime self: Self, comptime padding: anytype, comptime mode: PadMode) Pad(padding) {
            return .{
                .meta = &meta.Metadata.init(.{
                    .TypeOp = .{
                        .src = .{self.toAny()},
                        .op = .pad,
                        .args = .{
                            .pad = .{
                                .padding = &padding,
                                .mode = switch (mode) {
                                    .constant => |constant| .{ .constant = std.fmt.comptimePrint("{}", .{constant}) },
                                    else => mode,
                                },
                            },
                        },
                    },
                }, false, self.meta.label),
            };
        }

        pub fn View(comptime new_shape: anytype) type {
            return TensorType(dtype, new_shape);
        }

        /// Changes the shape and stride of the tensor to change how the underlying memory is accessed.
        /// Powerful enough to be used to implement any reshaping or windowing operation on a
        /// There are guardrails to prevent out of bounds access into underlying memory!
        pub fn view(comptime self: Self, comptime new_shape: anytype, comptime new_strides: [new_shape.len]u64, comptime new_offset: u64) View(new_shape) {
            // It is possible to directly view the first tensor that is not the result of a view op
            // View ops only rely on the new shape and new strides, broadcasting rules no longer apply
            // This greatly simplifies the graph as view ops are essentially compressed
            var first_not_view_tensor = self.toAny();
            while (std.meta.activeTag(first_not_view_tensor.meta.instr) == .TypeOp and first_not_view_tensor.meta.instr.TypeOp.op == .view) {
                first_not_view_tensor = first_not_view_tensor.meta.instr.TypeOp.src[0];
            }
            var dst = View(new_shape){
                .meta = &meta.Metadata.init(
                    .{
                        .TypeOp = .{
                            .src = .{first_not_view_tensor},
                            .op = .view,
                            .args = .{ .view = {} },
                        },
                    },
                    self.meta.constant,
                    self.meta.label,
                ),
                .strides = &new_strides,
                .offset = new_offset,
            };
            if (dst.storageSize() > self.storageSize()) {
                @compileError(std.fmt.comptimePrint(
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
                    self.shape[0..self.ndims],
                    self.strides[0..self.ndims],
                    self.offset,
                    self.storageSize(),
                    dst.shape[0..dst.ndims],
                    dst.strides[0..dst.ndims],
                    dst.offset,
                    dst.storageSize(),
                }));
            }
            return dst;
        }

        ///Apply an elementwise unary operation
        pub fn unaryFn(self: Self, comptime op: ops.UnaryOp) Self {
            return .{
                .meta = &meta.Metadata.init(
                    .{ .UnaryOp = .{
                        .src = .{self.toAny()},
                        .op = op,
                    } },
                    self.meta.constant,
                    self.meta.label,
                ),
                .strides = self.strides,
                .offset = self.offset,
            };
        }

        pub fn BinaryFnResultType(comptime other: anytype, comptime op: ops.BinaryOp) type {
            const Other = TensorTypeOf(other);
            const new_dtype: dtypes.DType = switch (op) {
                .equals, .less_than => .bool,
                else => dtypes.resultDType(Self.dtype, Other.dtype),
            };
            return TensorType(new_dtype, utils.broadcastShape(shape, Other.shape));
        }
        /// Apply an elementwise binary operation on two arrays, with broadcasting
        /// a and b must have the same "dtype class" meaning both must be float, bool, or int
        /// though different sizes are allowed.
        pub fn binaryFn(self: Self, other: anytype, comptime op: ops.BinaryOp) BinaryFnResultType(other, op) {
            const Other = TensorTypeOf(other);
            const bc_shape = utils.broadcastShape(shape, Other.shape);

            const a = self.expand(bc_shape);
            const b = asTensor(other).expand(bc_shape);
            return .{
                .meta = &meta.Metadata.init(
                    .{ .BinaryOp = .{
                        .src = .{ a.toAny(), b.toAny() },
                        .op = op,
                    } },
                    a.meta.constant and b.meta.constant,
                    null,
                ),
            };
        }

        pub fn ReduceFnResultType(comptime reduce_dims: anytype) type {
            const reduced_shape: [ndims]u64 = switch (@typeInfo(@TypeOf(reduce_dims))) {
                .ComptimeInt, .Int => blk: {
                    const dim = signedToUnsignedDim(reduce_dims);
                    if (dim < 0 or dim >= ndims) {
                        @compileError("Dimension index for single dimension reduce is out of bounds");
                    }
                    var reduced_shape: [ndims]u64 = shape;
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
                    var reduced_shape: [ndims]u64 = shape;
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
        /// Perform a reduction across 1 or more (or all) dimensions of a
        /// Dimensions to reduce can be passed as a int for 1 dim, tuple for multiple dims, or null/void for all dims
        pub fn reduceFn(
            self: Self,
            comptime op: ops.ReduceOp,
            comptime reduce_dims: anytype,
        ) ReduceFnResultType(reduce_dims) {
            // Use u16 here because []const u8 shows up as a string
            const reduce_dims_array: []const u16 = switch (@typeInfo(@TypeOf(reduce_dims))) {
                .ComptimeInt, .Int => &[1]u16{signedToUnsignedDim(reduce_dims)},
                .Null, .Void => @as([ndims]u16, std.simd.iota(u16, ndims))[0..],
                else => &reduce_dims,
            };
            const reduce_dims_mask: [ndims]bool = switch (@typeInfo(@TypeOf(reduce_dims))) {
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
                .meta = &meta.Metadata.init(
                    .{
                        .ReduceOp = .{
                            .src = .{self.toAny()},
                            .op = op,
                            .args = .{
                                .dims = reduce_dims_array,
                                .mask = &reduce_dims_mask,
                            },
                        },
                    },
                    false,
                    self.meta.label,
                ),
            };
        }

        pub fn Where(comptime true_value: anytype, comptime false_value: anytype) type {
            const True = TensorTypeOf(true_value);
            const False = TensorTypeOf(false_value);
            std.debug.assert(True.dtype == False.dtype);
            const bc_value_shape = utils.broadcastShape(True.shape, False.shape);
            const bc_result_shape = utils.broadcastShape(shape, bc_value_shape);
            return TensorType(True.dtype, bc_result_shape);
        }
        /// Conditional elementwise operator
        /// dst[i] = if (mask[i]) true_value[i] else false_value[i]
        /// Supports broadcasting between all 3 tensors, but true value and false value are broadcasted together first and must also have the same dtype
        pub fn where(mask: dtypes.BoolTensor(Self), true_value: anytype, false_value: anytype) where(true_value, false_value) {
            const Out = Where(true_value, false_value);
            const mask_expand = mask.expand(Out.shape);
            const true_expand = asTensor(true_value).expand(Out.shape);
            const false_expand = asTensor(false_value).expand(Out.shape);
            return .{
                .meta = &meta.Metadata.init(
                    .{ .TernaryOp = .{
                        .src = .{ mask_expand.toAny(), true_expand.toAny(), false_expand.toAny() },
                        .op = .where,
                    } },
                    false,
                    null,
                ),
            };
        }
    };
}

test "view" {
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
    const t4d = comptime Tensor([3][3][4][2]f32).empty();
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

test "sameness and uniqueness of input" {
    const tensor1 = comptime Tensor([2][1][4]i32).input("tensor1");
    const tensor1_1 = comptime Tensor([2][1][4]i32).input("tensor1");
    const tensor2 = comptime Tensor([2][1][4]i32).input("tensor2");

    try std.testing.expect(@intFromPtr(&tensor1) == @intFromPtr(&tensor1_1));
    try std.testing.expect(@intFromPtr(&tensor1) != @intFromPtr(&tensor2));
}
