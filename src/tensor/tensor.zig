const std = @import("std");
const F = @import("functions.zig");
const types = @import("types.zig");
const tests = @import("tests.zig");

const utils = @import("../utils.zig");
const ops = @import("../ops.zig");
const dtypes = @import("../dtypes.zig");
const graph = @import("../graph.zig");
const autograd = @import("../autograd.zig");

test tests {
    _ = tests;
}

/// An AnyTensor is a shape-erased tensor. This makes it easy to store in graphs,
/// but further computations are not allowed unless casted back to a shaped tensor.
pub const AnyTensor = Tensor(anyopaque);

/// A Tensor is a multidimensional Array.
pub fn Tensor(Array: type) type {
    // Need to be extern to have a well defined layout for pointer casting between AnyTensor and ShapedTensor
    return extern struct {
        const isAnyTensor = Array == anyopaque;
        // Self is used for all functions that involve reading fields or calculating some value.
        // It is fine for AnyTensor to be able to do these things too.
        const Self = @This();
        // To enforce a shaped tensor type for some functions, @This() is AnyTensor
        // then ShapedTensor is void, and AnyTensor won't have access to functions where
        // Self must not be AnyTensor.
        const ShapedTensor = if (isAnyTensor) void else Self;

        // All the functions for operations that do not modify metadata directly
        pub usingnamespace F;
        pub const _dtype: dtypes.DType = if (isAnyTensor) dtypes.DType.anyopaque else utils.extractDType(Array);
        pub const _ndims: u8 = if (isAnyTensor) 0 else utils.extractNdims(Array);
        pub const _shape: [_ndims]u64 = if (isAnyTensor) .{} else utils.extractShape(Array);
        pub const contiguous_strides: [_ndims]u64 = if (isAnyTensor) .{} else utils.contiguousStrides(&_shape);
        pub const num_elements = if (isAnyTensor) 0 else utils.numElements(&_shape);

        instr: *const ops.Instruction,
        layout: *const types.Layout = &.{
            .dtype = _dtype,
            .ndims = _ndims,
            .shape = &_shape,
            .strides = &contiguous_strides,
            .offset = 0,
        },
        labels: *const types.Labels = &.{
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

        pub inline fn toAnyTensor(self: *const ShapedTensor) *const AnyTensor {
            return @ptrCast(self);
        }

        pub inline fn toTensor(comptime self: *const AnyTensor) *const types.TensorTypeOf(self) {
            return @ptrCast(self);
        }

        pub fn toJson(self: *const Self) types.Json {
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
            if (isAnyTensor) unreachable;
            var Child = dtypes.ZigType(_dtype);
            for (0.._ndims) |dim| {
                Child = [_shape[_ndims - dim - 1]]Child;
            }
            return Child;
        }

        pub fn setDimNames(self: ShapedTensor, comptime dim_names: std.meta.Tuple(&[_]type{?[]const u8} ** _ndims)) Self {
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

        pub fn namedDim(comptime self: Self, dim: types.DimEnum(self.labels.dim_names)) u64 {
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
        pub fn dimSize(self: Self, d: i16) u64 {
            return self.layout.shape[signedToUnsignedDim(d)];
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
                .layout = self.layout,
                .autograd = &new_autograd,
            }).setName(name);
        }

        //
        // Initialization functions (init ops)
        //

        /// Create an empty tensor (i.e. allocate).
        /// Do not make any assumptions about data in the empty
        pub fn empty() ShapedTensor {
            std.debug.assert(!isAnyTensor);
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
        pub fn full(comptime value: dtypes.ZigType(_dtype)) ShapedTensor {
            std.debug.assert(!isAnyTensor);
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
        pub fn input(comptime name: []const u8) ShapedTensor {
            std.debug.assert(!isAnyTensor);
            return (ShapedTensor{
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
        pub fn param(name: []const u8) ShapedTensor {
            std.debug.assert(!isAnyTensor);
            return (ShapedTensor{
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
        pub fn random(name: []const u8) ShapedTensor {
            std.debug.assert(!isAnyTensor);
            std.debug.assert(dtypes.isFloat(_dtype));
            return (ShapedTensor{
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
        pub fn cast(comptime self: Self, comptime new_dtype: dtypes.DType) types.TensorType(new_dtype, _shape) {
            if (new_dtype == _dtype) return self;
            return .{
                .instr = &.{
                    .DataOp = .{
                        .in = .{self.toAnyTensor()},
                        .op = .cast,
                        .args = .{ .cast = new_dtype },
                    },
                },
                .layout = &blk: {
                    var new_layout = self.layout.*;
                    new_layout.dtype = new_dtype;
                    break :blk new_layout;
                },
                .autograd = &.{
                    .grad_fn = autograd.noGrad,
                    .constant = self.autograd.constant,
                },
                .labels = self.labels,
            };
        }

        /// Make an array contguous (a full new copy) if it is not already
        pub fn contiguous(self: ShapedTensor) ShapedTensor {
            if (self.isContiguous()) return self;
            return .{
                .instr = &.{
                    .DataOp = .{
                        .in = .{self.toAnyTensor()},
                        .op = .contiguous,
                        .args = .{
                            .contiguous = .{
                                .shape = self.layout.shape,
                                .strides = &contiguous_strides,
                                .offset = 0,
                            },
                        },
                    },
                },
                .labels = self.labels,
            };
        }

        const PadMode = if (!isAnyTensor) union(ops.DataOp.Args.Pad.Mode) {
            constant: dtypes.ZigType(ShapedTensor._dtype),
            reflect: void,
            replicate: void,
            circular: void,
        } else void;

        pub fn pad(comptime self: ShapedTensor, comptime padding: anytype, comptime mode: PadMode) types.Pad(ShapedTensor, padding) {
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

        /// Changes the shape and stride of the tensor to change how the underlying memory is accessed.
        /// Powerful enough to be used to implement any reshaping or windowing operation on a
        /// There are guardrails to prevent out of bounds access into underlying memory!
        pub fn view(
            comptime self: ShapedTensor,
            comptime new_shape: anytype,
            comptime new_strides: [new_shape.len]u64,
            comptime new_offset: u64,
        ) types.View(ShapedTensor, new_shape) {
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

            const out = types.View(ShapedTensor, new_shape){
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

        ///Apply an elementwise unary operation
        pub fn applyUnaryOp(self: ShapedTensor, comptime op: ops.UnaryOp) types.UnaryOpResult(Self, op) {
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

        /// Apply an elementwise binary operation on two arrays, with broadcasting
        /// a and b must have the same "dtype class" meaning both must be float, bool, or int
        /// though different sizes are allowed.
        pub fn applyBinaryOp(self: ShapedTensor, other: anytype, comptime op: ops.BinaryOp) types.BinaryOpResult(Self, types.TensorTypeOf(other), op) {
            const Other = types.TensorTypeOf(other);
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

        /// Perform a reduction across 1 or more (or all) dimensions of a
        /// Dimensions to reduce can be passed as a int for 1 dim, tuple for multiple dims, or null/void for all dims
        pub fn applyReduceOp(
            self: ShapedTensor,
            comptime op: ops.ReduceOp,
            comptime reduce_dims: anytype,
        ) types.ReduceOpResult(Self, reduce_dims) {
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

        pub fn backwards(self: Self) []const *const AnyTensor {
            return autograd.backwards(self);
        }
    };
}
