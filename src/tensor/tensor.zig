const std = @import("std");
const F = @import("functions.zig");
const tensor_typing = @import("tensor_typing.zig");

const utils = @import("../utils.zig");
const ops = @import("../ops.zig");
const dtypes = @import("../dtypes.zig");
const graph = @import("../graph.zig");
const autograd = @import("../autograd.zig");

test Tensor {
    _ = @import("tensor_testing.zig");
}

pub const Layout = struct {
    dtype: dtypes.DType,
    ndims: u8,
    shape: []const u64,
    strides: []const u64,
    offset: u64,
};

pub const Labels = struct {
    name: ?[]const u8,
    dim_names: ?[]const ?[]const u8,
};

pub const Json = struct {
    ptr: usize,
    dtype: dtypes.DType,
    ndims: u8,
    shape: []const u64,
    strides: []const u64,
    offset: u64,
};

pub const AnyTensor = Tensor(anyopaque);

pub fn Tensor(comptime TensorArrayType: type) type {
    const isAnyTensor = TensorArrayType == anyopaque;
    return extern struct {
        const Self = @This();

        // All the functions for operations that do not modify metadata directly
        pub usingnamespace F;
        pub const _dtype: dtypes.DType = if (isAnyTensor) dtypes.DType.anyopaque else utils.extractDType(TensorArrayType);
        pub const _ndims: u8 = if (isAnyTensor) 0 else utils.extractNdims(TensorArrayType);
        pub const _shape: [_ndims]u64 = if (isAnyTensor) .{} else utils.extractShape(TensorArrayType);
        pub const contiguous_strides: [_ndims]u64 = if (isAnyTensor) .{} else utils.contiguousStrides(&_shape);
        pub const num_elements = if (isAnyTensor) 0 else utils.numElements(&_shape);

        instr: *const ops.Instruction,
        layout: *const Layout = &.{
            .dtype = _dtype,
            .ndims = _ndims,
            .shape = &_shape,
            .strides = &contiguous_strides,
            .offset = 0,
        },
        labels: *const Labels = &.{
            .name = null,
            .dim_names = null,
        },
        autograd: *const autograd.Autograd = &.{
            .grad_fn = autograd.noGrad,
            .constant = false,
        },

        pub inline fn dtype(self: Self) dtypes.DType {
            return self.layout.dtype;
        }

        pub inline fn ndims(self: Self) u8 {
            return self.layout.ndims;
        }

        pub inline fn shape(self: Self) *const [_ndims]u64 {
            return @ptrCast(self.layout.shape);
        }

        pub inline fn strides(self: Self) *const [_ndims]u64 {
            return @ptrCast(self.layout.strides);
        }

        pub inline fn offset(self: Self) u64 {
            return self.layout.offset;
        }

        pub inline fn toAnyTensor(comptime self: *const Self) *const AnyTensor {
            return @ptrCast(self);
        }

        pub inline fn toTensor(comptime self: *const AnyTensor) *const tensor_typing.TensorTypeOf(self) {
            return @ptrCast(self);
        }

        pub fn toJson(self: *const Self) Json {
            return .{
                .ptr = @intFromPtr(self),
                .dtype = self.dtype(),
                .ndims = self.ndims(),
                .shape = self.shape(),
                .strides = self.strides(),
                .offset = self.offset(),
            };
        }

        pub fn ArrayType() type {
            var Child = dtypes.ZigType(_dtype);
            for (0.._ndims) |dim| {
                Child = [_shape[_ndims - dim - 1]]Child;
            }
            return Child;
        }

        pub fn DimsEnumType(maybe_dim_names: ?[]const ?[]const u8) type {
            var dim_enum_fields: [_ndims]std.builtin.Type.EnumField = undefined;
            var enum_idx: usize = 0;
            if (maybe_dim_names) |dim_names| {
                for (dim_names, 0..) |maybe_name, dim_idx| {
                    if (maybe_name) |name| {
                        dim_enum_fields[enum_idx] = std.builtin.Type.EnumField{ .name = name[0.. :0], .value = dim_idx };
                        enum_idx += 1;
                    }
                }
                return @Type(std.builtin.Type{ .Enum = .{ .fields = dim_enum_fields[0..enum_idx], .is_exhaustive = false, .tag_type = u8, .decls = &.{} } });
            } else {
                return void;
            }
        }

        pub fn setDimNames(self: Self, comptime dim_names: std.meta.Tuple(&[_]type{?[]const u8} ** _ndims)) Self {
            return .{
                .instr = self.instr,
                .labels = &.{
                    .name = self.labels.name,
                    .dim_names = &dim_names,
                },
                .autograd = self.autograd,
                .layout = self.layout,
            };
        }

        pub fn namedDim(comptime self: Self, dim: DimsEnumType(self.labels.dim_names)) u64 {
            return @intFromEnum(dim);
        }

        /// Determine if the stride pattern of the tensor defines a fully contiguous section of memory at runtime
        pub fn isContiguous(self: Self) bool {
            // Strides need to be decreasing unless its a broadcasted stride (0)
            return utils.isContiguous(self.layout.strides);
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
            var _size: u128 = self.layout.offset + 1;
            for (0.._ndims) |d| {
                _size += (_shape[d] - 1) * self.layout.strides[d];
            }
            // The result is the size of the storage needed to visit all indices of the tensor
            return _size;
        }

        pub fn size(_: Self) u128 {
            return num_elements;
        }

        /// Allows for negative dimension indexing to work by normalizing it to [0,ndims)
        pub fn signedToUnsignedDim(dim: i16) u8 {
            return utils.signedToUnsignedDimNdims(_ndims, dim);
        }

        /// Supports negative indexing sugar (e.g. -1 = ndims - 1)
        pub fn dimSize(_: Self, d: i16) u64 {
            return _shape[signedToUnsignedDim(d)];
        }

        /// Supports negative indexing sugar (e.g. -1 = ndims - 1)
        pub fn dimStride(self: Self, d: i16) u64 {
            return self.layout.strides[signedToUnsignedDim(d)];
        }

        pub fn setName(self: Self, name: ?[]const u8) Self {
            return .{
                .instr = self.instr,
                .labels = &.{
                    .name = name,
                    .dim_names = self.labels.dim_names,
                },
                .autograd = self.autograd,
                .layout = self.layout,
            };
        }

        pub fn requiresGrad(self: Self, name: []const u8) Self {
            const new_autograd = blk: {
                var new_autograd = self.autograd.*;
                new_autograd.constant = false;
                new_autograd.grad_fn = struct {
                    pub fn gradFnImpl(grad: anytype, param_grads: []const *const AnyTensor) []const *const AnyTensor {
                        return autograd.accumulateGrad(name, grad, param_grads);
                    }
                }.gradFnImpl;
                break :blk new_autograd;
            };

            return (Self{
                .instr = self.instr,
                .autograd = &new_autograd,
            }).setName(name);
        }

        //
        // Initialization functions (init ops)
        //

        /// Create an empty tensor (i.e. allocate).
        /// Do not make any assumptions about data in the empty
        pub fn empty() Self {
            return .{
                .instr = &.{
                    .InitOp = .{
                        .op = .empty,
                        .args = .{ .empty = {} },
                    },
                },
            };
        }

        /// Fill a tensor with a value
        /// By default, full tensors will be constant folded in codegen
        /// unless they are marked as requires_grad
        pub fn full(comptime value: dtypes.ZigType(_dtype)) Self {
            const str = std.fmt.comptimePrint("{}", .{value});
            return (Self{
                .instr = &.{
                    .InitOp = .{
                        .op = .full,
                        .args = .{ .full = str },
                    },
                },
            }).setName("const_" ++ str[0..@min(str.len, 6)]);
        }

        /// Used to mark a tensor as an input to a graph,
        /// codegen will make this an argument of the function
        /// A label can be given to make two tensors of the same shape/dtype
        /// correspond to different arrays at runtime (e.g. for two input images )
        pub fn input(comptime name: []const u8) Self {
            return (Self{
                .instr = &.{
                    .InitOp = .{
                        .op = .input,
                        .args = .{ .input = {} },
                    },
                },
            }).setName(name);
        }

        /// Used to mark a tensor as a learnable param,
        /// codegen will make this an argument of the function,
        /// gradients can be accumulated for it,
        /// and optimizers can detect it,
        pub fn param(name: []const u8) Self {
            return (Self{
                .instr = &.{
                    .InitOp = .{
                        .op = .param,
                        .args = .{ .param = {} },
                    },
                },
            }).requiresGrad(name);
        }

        /// Fill a tensor with random generated numbers
        /// By default, random tensors will be constant folded in codegen
        /// unless they are marked as requires_grad
        /// Do not use this for random initialization of param_grads!
        pub fn random(name: []const u8) Self {
            std.debug.assert(dtypes.isFloat(_dtype));
            return (Self{
                .instr = &.{
                    .InitOp = .{
                        .op = .random,
                        .args = .{ .random = {} },
                    },
                },
            }).setName(name);
        }

        //
        // Type / shape manipulation functions
        //

        ///Cast an array of a datatype to another datatype
        pub fn cast(comptime self: Self, comptime new_dtype: dtypes.DType) tensor_typing.TensorType(new_dtype, _shape) {
            if (new_dtype == _dtype) return self;
            const new_layout = blk: {
                var new_layout = self.layout.*;
                new_layout.dtype = new_dtype;
                break :blk new_layout;
            };
            return .{
                .instr = &.{
                    .DataOp = .{
                        .in = .{self.toAnyTensor()},
                        .op = .cast,
                        .args = .{ .cast = new_dtype },
                    },
                },
                .layout = &new_layout,
                .autograd = &.{
                    .grad_fn = autograd.noGrad,
                    .constant = self.autograd.constant,
                },
                .labels = self.labels,
            };
        }

        /// Make an array contguous (a full new copy) if it is not already
        pub fn contiguous(comptime self: Self) Self {
            if (self.isContiguous()) return self;
            return .{
                .instr = &.{
                    .DataOp = .{
                        .in = .{self.toAnyTensor()},
                        .op = .contiguous,
                        .args = .{
                            .contiguous = .{
                                .shape = self._shape,
                                .strides = &contiguous_strides,
                                .offset = 0,
                            },
                        },
                    },
                },
                .labels = self.labels,
            };
        }

        const PadMode = union(ops.DataOp.Args.Pad.Mode) {
            constant: dtypes.ZigType(_dtype),
            reflect: void,
            replicate: void,
            circular: void,
        };
        pub fn Pad(padding: anytype) type {
            const padded_dims = padding.len;
            const padding_tuple: [padded_dims][2]u64 = padding;
            std.debug.assert(padded_dims <= _ndims);
            var new_shape: [_ndims]usize = _shape;
            for (0..padded_dims) |dim| {
                new_shape[_ndims - dim - 1] += padding_tuple[dim][0] + padding_tuple[dim][1];
            }
            return tensor_typing.TensorType(_dtype, new_shape);
        }
        pub fn pad(comptime self: Self, comptime padding: anytype, comptime mode: PadMode) Pad(padding) {
            return .{
                .instr = &.{
                    .DataOp = .{
                        .in = .{self.toAnyTensor()},
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
                },
                .labels = self.labels,
                .autograd = self.autograd,
            };
        }

        pub fn View(comptime new_shape: anytype) type {
            return tensor_typing.TensorType(_dtype, new_shape);
        }

        /// Changes the shape and stride of the tensor to change how the underlying memory is accessed.
        /// Powerful enough to be used to implement any reshaping or windowing operation on a
        /// There are guardrails to prevent out of bounds access into underlying memory!
        pub fn view(
            comptime self: Self,
            comptime new_shape: anytype,
            comptime new_strides: [new_shape.len]u64,
            comptime new_offset: u64,
        ) View(new_shape) {
            // It is possible to directly view the first tensor that is not the result of a view op
            // View ops only rely on the new shape and new strides, broadcasting rules no longer apply
            // This greatly simplifies the graph as view ops are essentially compressed
            const first_not_view_tensor = blk: {
                var t = self.toAnyTensor();
                while (std.meta.activeTag(t.instr.*) == .DataOp and t.instr.DataOp.op == .view) {
                    t = t.instr.DataOp.in[0];
                }
                break :blk t;
            };

            const out = View(new_shape){
                .instr = &.{
                    .DataOp = .{
                        .in = .{first_not_view_tensor},
                        .op = .view,
                        .args = .{
                            .view = .{
                                .shape = &new_shape,
                                .strides = &new_strides,
                                .offset = new_offset,
                            },
                        },
                    },
                },
                .layout = &.{
                    .dtype = _dtype,
                    .ndims = new_shape.len,
                    .shape = &new_shape,
                    .strides = &new_strides,
                    .offset = new_offset,
                },
                .autograd = &.{
                    .constant = self.autograd.constant,
                    .grad_fn = autograd.noGrad,
                },
                .labels = self.labels,
            };
            if (out.storageSize() > self.storageSize()) {
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
                    self.layout.shape,
                    self.layout.strides,
                    self.layout.offset,
                    self.storageSize(),
                    out.layout.shape,
                    out.layout.strides,
                    out.layout.offset,
                    out.storageSize(),
                }));
            }
            return out;
        }

        pub fn UnaryOpResultType(comptime op: ops.UnaryOp) type {
            return switch (op) {
                .exp2, .log2, .recip, .sin, .sqrt => tensor_typing.FloatTensor(Self),
                .neg => Self,
            };
        }

        ///Apply an elementwise unary operation
        pub fn applyUnaryOp(self: Self, comptime op: ops.UnaryOp) UnaryOpResultType(op) {
            const grad_fn = struct {
                pub fn gradFnImpl(grad_out: anytype, param_grads: []const *const AnyTensor) []const *const AnyTensor {
                    return autograd.unaryGrad(op, self, grad_out, param_grads);
                }
            }.gradFnImpl;
            return .{
                .instr = &.{
                    .UnaryOp = .{
                        .in = .{self.toAnyTensor()},
                        .op = op,
                    },
                },
                .autograd = &.{
                    .constant = self.autograd.constant,
                    .grad_fn = grad_fn,
                },
                .labels = self.labels,
            };
        }

        pub fn BinaryOpResultType(comptime other: anytype, comptime op: ops.BinaryOp) type {
            const Other = tensor_typing.TensorTypeOf(other);
            const new_dtype: dtypes.DType = switch (op) {
                .eq, .lt => .bool,
                else => dtypes.resultDType(Self._dtype, Other._dtype),
            };
            return tensor_typing.TensorType(new_dtype, utils.broadcastShape(_shape, Other._shape));
        }
        /// Apply an elementwise binary operation on two arrays, with broadcasting
        /// a and b must have the same "dtype class" meaning both must be float, bool, or int
        /// though different sizes are allowed.
        pub fn applyBinaryOp(self: Self, other: anytype, comptime op: ops.BinaryOp) BinaryOpResultType(other, op) {
            const Other = tensor_typing.TensorTypeOf(other);
            const bc_shape = utils.broadcastShape(_shape, Other._shape);
            const a = self.expand(bc_shape);
            const b = F.expand(other, bc_shape);
            const grad_fn = struct {
                pub fn gradFnImpl(grad_out: anytype, param_grads: []const *const AnyTensor) []const *const AnyTensor {
                    return autograd.binaryGrad(op, a, b, grad_out, param_grads);
                }
            }.gradFnImpl;
            return .{
                .instr = &.{
                    .BinaryOp = .{
                        .in = .{ a.toAnyTensor(), b.toAnyTensor() },
                        .op = op,
                    },
                },
                .autograd = &.{
                    .grad_fn = grad_fn,
                    .constant = a.autograd.constant and b.autograd.constant,
                },
            };
        }

        pub fn ReduceOpResultType(comptime reduce_dims: anytype) type {
            const reduced_shape: [_ndims]u64 = switch (@typeInfo(@TypeOf(reduce_dims))) {
                .ComptimeInt, .Int => blk: {
                    const dim = signedToUnsignedDim(reduce_dims);
                    if (dim < 0 or dim >= _ndims) {
                        @compileError("Dimension index for single dimension reduce is out of bounds");
                    }
                    var reduced_shape: [_ndims]u64 = _shape;
                    reduced_shape[dim] = 1;
                    break :blk reduced_shape;
                },
                .Null, .Void => blk: {
                    break :blk .{1} ** _ndims;
                },
                else => blk: {
                    const dims = reduce_dims;
                    if (dims.len > _ndims) {
                        @compileError("Length of dimension index array for multi dimension reduce is out of bounds");
                    }
                    var reduce_dim_mask: [_ndims]bool = [_]bool{false} ** _ndims;
                    var reduced_shape: [_ndims]u64 = _shape;
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
            return tensor_typing.TensorType(_dtype, reduced_shape);
        }
        /// Perform a reduction across 1 or more (or all) dimensions of a
        /// Dimensions to reduce can be passed as a int for 1 dim, tuple for multiple dims, or null/void for all dims
        pub fn applyReduceOp(
            self: Self,
            comptime op: ops.ReduceOp,
            comptime reduce_dims: anytype,
        ) ReduceOpResultType(reduce_dims) {
            // Use u16 here because []const u8 shows up as a string
            const reduce_dims_array: []const u8 = switch (@typeInfo(@TypeOf(reduce_dims))) {
                .ComptimeInt, .Int => &[1]u8{signedToUnsignedDim(reduce_dims)},
                .Void => @as([_ndims]u8, std.simd.iota(u8, _ndims))[0..],
                else => &reduce_dims,
            };
            const reduce_dims_mask: [_ndims]bool = switch (@typeInfo(@TypeOf(reduce_dims))) {
                .ComptimeInt, .Int => blk: {
                    var tmp_mask: [_ndims]bool = [_]bool{false} ** _ndims;
                    const dim = reduce_dims;
                    tmp_mask[signedToUnsignedDim(dim)] = true;
                    break :blk tmp_mask;
                },
                .Void => [_]bool{true} ** _ndims,
                else => blk: {
                    var tmp_mask: [_ndims]bool = [_]bool{false} ** _ndims;
                    for (reduce_dims) |dim| {
                        tmp_mask[signedToUnsignedDim(dim)] = true;
                    }
                    break :blk tmp_mask;
                },
            };
            return .{
                .instr = &.{
                    .ReduceOp = .{
                        .in = .{self.toAnyTensor()},
                        .op = op,
                        .args = .{
                            .dims = reduce_dims_array,
                            .mask = &reduce_dims_mask,
                        },
                    },
                },
                .autograd = &.{
                    .constant = false,
                    .grad_fn = autograd.noGrad,
                },
                .labels = self.labels,
            };
        }

        pub fn Where(comptime true_value: anytype, comptime false_value: anytype) type {
            const True = tensor_typing.TensorTypeOf(true_value);
            const False = tensor_typing.TensorTypeOf(false_value);
            std.debug.assert(True._dtype == False._dtype);
            const bc_value_shape = utils.broadcastShape(True._shape, False._shape);
            const bc_result_shape = utils.broadcastShape(_shape, bc_value_shape);
            return tensor_typing.TensorType(True._dtype, bc_result_shape);
        }
        /// Conditional elementwise operator
        /// out[i] = if (mask[i]) true_value[i] else false_value[i]
        /// Supports broadcasting between all 3 tensors, but true value and false value are broadcasted together first and must also have the same dtype
        pub fn where(mask: dtypes.BoolTensor(Self), true_value: anytype, false_value: anytype) where(true_value, false_value) {
            const Out = Where(true_value, false_value);
            const mask_expand = mask.expand(Out._shape);
            const true_expand = tensor_typing.asTensor(true_value).expand(Out._shape);
            const false_expand = tensor_typing.asTensor(false_value).expand(Out._shape);
            return .{
                .instr = .{
                    .TernaryOp = .{
                        .in = .{ mask_expand.toAnyTensor(), true_expand.toAnyTensor(), false_expand.toAnyTensor() },
                        .op = .where,
                    },
                },
                .autograd = &.{
                    .grad_fn = autograd.noGrad,
                    .constant = false,
                },
            };
        }

        pub fn backwards(self: Self) []const *const AnyTensor {
            return autograd.backwards(self);
        }
    };
}
