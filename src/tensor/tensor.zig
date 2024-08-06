const std = @import("std");
const AnyTensor = @import("anytensor.zig").AnyTensor;
const F = @import("functions.zig");
const shared = @import("shared.zig");
const typing = @import("tensor_typing.zig");

const utils = @import("../utils.zig");
const ops = @import("../ops.zig");
const dtypes = @import("../dtypes.zig");
const graph = @import("../graph.zig");
const autograd = @import("../autograd.zig");

test Tensor {
    _ = @import("tensor_testing.zig");
}
pub fn Tensor(comptime TensorArrayType: type) type {
    return extern struct {
        const Self = @This();

        // All the functions for operations that do not modify metadata directly
        pub usingnamespace F;

        pub const dtype: dtypes.DType = utils.extractDType(TensorArrayType);
        pub const ndims: u8 = utils.extractNdims(TensorArrayType);
        pub const shape: [ndims]u64 = utils.extractShape(TensorArrayType);
        pub const contiguous_strides: [ndims]u64 = utils.contiguousStrides(&shape);
        pub const num_elements = utils.numElements(&shape);

        meta: *const shared.Metadata,
        dtype: dtypes.DType = dtype,
        ndims: u8 = ndims,
        shape: *const [ndims]u64 = &shape,
        strides: *const [ndims]u64 = &contiguous_strides,
        offset: u64 = 0,

        pub fn toAnyTensor(self: *const Self) *const AnyTensor {
            return @ptrCast(self);
        }

        /// Determine if the stride pattern of the tensor defines a fully contiguous section of memory at runtime
        pub fn isContiguous(self: *const Self) bool {
            // Strides need to be decreasing unless its a broadcasted stride (0)
            return utils.isContiguous(self.strides);
        }

        pub fn size(_: *const Self) u128 {
            return num_elements;
        }

        /// Allows for negative dimension indexing to work by normalizing it to [0,ndims)
        pub fn signedToUnsignedDim(dim: i16) u8 {
            return utils.signedToUnsignedDimNdims(Self.ndims, dim);
        }

        /// Supports negative indexing sugar (e.g. -1 = ndims - 1)
        pub fn dimSize(_: *const Self, d: i16) u64 {
            return shape[signedToUnsignedDim(d)];
        }

        /// Supports negative indexing sugar (e.g. -1 = ndims - 1)
        pub fn dimStride(self: *const Self, d: i16) u64 {
            return self.strides[signedToUnsignedDim(d)];
        }

        pub fn setLabel(
            self: *const Self,
            label: ?[]const u8,
            allocator: std.mem.Allocator,
        ) *const Self {
            const meta = allocator.create(shared.Metadata) catch unreachable;
            meta.* = self.meta.*;
            meta.label = label;

            const out: *Self = allocator.create(Self) catch unreachable;
            out.* = .{ .meta = meta };
            return out;
        }

        pub fn requiresGrad(
            self: *const Self,
            label: ?[]const u8,
            allocator: std.mem.Allocator,
        ) *const Self {
            const meta = allocator.create(shared.Metadata) catch unreachable;
            meta.* = self.meta.*;
            meta.label = label;
            meta.requires_grad = true;
            meta.grad_fn = struct {
                pub fn gradFnImpl(grad: anytype, param_grads: []const *const AnyTensor) []const *const AnyTensor {
                    return autograd.accumulateGrad(label, grad, param_grads);
                }
            }.gradFnImpl;
            const out: *Self = allocator.create(Self) catch unreachable;
            out.* = .{ .meta = meta };
            return out;
        }

        //
        // Initialization functions (init ops)
        //

        /// Create an empty tensor (i.e. allocate).
        /// Do not make any assumptions about data in the empty
        pub fn empty(allocator: std.mem.Allocator) *const Self {
            const instr = .{
                .InitOp = .{
                    .op = .empty,
                    .args = .{ .empty = {} },
                },
            };
            const meta = allocator.create(shared.Metadata) catch unreachable;
            meta.* = .{
                .instr = instr,
                .constant = false,
                .label = null,
                .grad_fn = autograd.noGrad,
            };
            const out: *Self = allocator.create(Self) catch unreachable;
            out.* = .{ .meta = meta };
            return out;
        }

        /// Fill a tensor with a value
        /// By default, full tensors will be constant folded in codegen
        /// unless they are marked as requires_grad
        pub fn full(
            comptime value: dtypes.ZigType(dtype),
            allocator: std.mem.Allocator,
        ) *const Self {
            const instr = .{
                .InitOp = .{
                    .op = .full,
                    .args = .{ .full = std.fmt.comptimePrint("{}", .{value}) },
                },
            };
            const meta = allocator.create(shared.Metadata) catch unreachable;
            meta.* = .{
                .instr = instr,
                .constant = true,
                .label = null,
                .grad_fn = autograd.noGrad,
            };
            meta.label = "const_" ++ instr.InitOp.args.full[0..@min(instr.InitOp.args.full.len, 6)];
            const out: *Self = allocator.create(Self) catch unreachable;
            out.* = .{ .meta = meta };
            return out;
        }

        /// Used to mark a tensor as an input to a graph,
        /// codegen will make this an argument of the function
        /// A label can be given to make two tensors of the same shape/dtype
        /// correspond to different arrays at runtime (e.g. for two input images )
        pub fn input(
            comptime label: []const u8,
            allocator: std.mem.Allocator,
        ) *const Self {
            const meta = allocator.create(shared.Metadata) catch unreachable;
            meta.* = .{
                .instr = .{
                    .InitOp = .{
                        .op = .input,
                        .args = .{ .input = {} },
                    },
                },
                .grad_fn = autograd.noGrad,
                .constant = false,
                .label = label,
            };
            const out: *Self = allocator.create(Self) catch unreachable;
            out.* = .{ .meta = meta };
            return out;
        }

        /// Used to mark a tensor as a learnable param,
        /// codegen will make this an argument of the function,
        /// gradients can be accumulated for it,
        /// and optimizers can detect it,
        pub fn param(
            label: []const u8,
            allocator: std.mem.Allocator,
        ) *const Self {
            const meta = allocator.create(shared.Metadata) catch unreachable;
            meta.* = .{
                .instr = .{
                    .InitOp = .{
                        .op = .param,
                        .args = .{ .param = {} },
                    },
                },
                .constant = false,
                .label = label,
                .requires_grad = true,
                .grad_fn = struct {
                    pub fn gradFnImpl(grad: anytype, param_grads: []const *const AnyTensor) []const *const AnyTensor {
                        return autograd.accumulateGrad(label, grad, param_grads);
                    }
                }.gradFnImpl,
            };
            const out: *Self = allocator.create(Self) catch unreachable;
            out.* = .{ .meta = meta };
            return out;
        }

        /// Fill a tensor with random generated numbers
        /// By default, random tensors will be constant folded in codegen
        /// unless they are marked as requires_grad
        /// Do not use this for random initialization of param_grads!
        /// Note that some device backends do not support this
        pub fn random(
            label: []const u8,
            allocator: std.mem.Allocator,
        ) *const Self {
            std.debug.assert(dtypes.isFloat(dtype));
            const meta = allocator.create(shared.Metadata) catch unreachable;
            meta.* = .{
                .instr = .{
                    .InitOp = .{
                        .op = .random,
                        .args = .{ .random = {} },
                    },
                },
                .constant = false,
                .label = label,
            };
            const out: *Self = allocator.create(Self) catch unreachable;
            out.* = .{ .meta = meta };
            return out;
        }

        //
        // Type / shape manipulation functions
        //

        ///Cast an array of a datatype to another datatype
        pub fn cast(
            self: *const Self,
            comptime new_dtype: dtypes.DType,
            allocator: std.mem.Allocator,
        ) *const typing.Cast(Self, new_dtype) {
            if (typing.Cast(Self, new_dtype) == Self) return self;
            const meta = allocator.create(shared.Metadata) catch unreachable;
            meta.* = .{
                .instr = .{
                    .DataOp = .{
                        .in = .{self.toAnyTensor()},
                        .op = .cast,
                        .args = .{ .cast = new_dtype },
                    },
                },
                .grad_fn = autograd.noGrad,
                .constant = self.meta.constant,
                .label = self.meta.label,
            };
            const Out = comptime typing.Cast(Self, new_dtype);
            const out: *Out = allocator.create(Out) catch unreachable;
            out.* = .{
                .meta = meta,
                .strides = self.strides,
                .offset = self.offset,
            };
            return out;
        }

        /// Make an array contguous (a full new copy) if it is not already
        pub fn contiguous(
            self: *const Self,
            allocator: std.mem.Allocator,
        ) *const Self {
            if (self.isContiguous()) return self;
            const meta = allocator.create(shared.Metadata) catch unreachable;
            meta.* = .{
                .instr = .{
                    .DataOp = .{
                        .in = .{self.toAnyTensor()},
                        .op = .contiguous,
                        .args = .{
                            .contiguous = .{
                                .shape = self.shape,
                                .strides = &contiguous_strides,
                                .offset = 0,
                            },
                        },
                    },
                },
                .constant = false,
                .label = self.meta.label,
            };
            const out: *Self = allocator.create(Self) catch unreachable;
            out.* = .{ .meta = meta };
            return out;
        }

        pub fn pad(
            self: *const Self,
            comptime padding: anytype,
            comptime mode: shared.PadMode(dtype),
            allocator: std.mem.Allocator,
        ) *const typing.Pad(Self, padding) {
            const meta = allocator.create(shared.Metadata) catch unreachable;
            meta.* = .{
                .instr = .{
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
                .grad_fn = autograd.noGrad,
                .constant = false,
                .label = self.meta.label,
            };
            const Out = comptime typing.Pad(Self, padding);
            const out: *Out = allocator.create(Out) catch unreachable;
            out.* = .{ .meta = meta };
            return out;
        }

        /// Changes the shape and stride of the tensor to change how the underlying memory is accessed.
        /// Powerful enough to be used to implement any reshaping or windowing operation on a
        /// There are guardrails to prevent out of bounds access into underlying memory!
        pub fn view(
            self: *const Self,
            comptime new_shape: anytype,
            new_strides: [new_shape.len]u64,
            new_offset: u64,
            allocator: std.mem.Allocator,
        ) *const typing.View(Self, new_shape) {
            // It is possible to directly view the first tensor that is not the result of a view op
            // View ops only rely on the new shape and new strides, broadcasting rules no longer apply
            // This greatly simplifies the graph as view ops are essentially compressed
            const first_not_view_tensor = blk: {
                var t = self.toAnyTensor();
                while (std.meta.activeTag(t.meta.instr) == .DataOp and t.meta.instr.DataOp.op == .view) {
                    t = t.meta.instr.DataOp.in[0];
                }
                break :blk t;
            };
            const strides: *const [new_shape.len]u64 = @ptrCast(allocator.dupe(u64, &new_strides) catch unreachable);
            const meta = allocator.create(shared.Metadata) catch unreachable;
            meta.* = .{
                .instr = .{
                    .DataOp = .{
                        .in = .{first_not_view_tensor},
                        .op = .view,
                        .args = .{
                            .view = .{
                                .shape = &new_shape,
                                // This part is a bit sus as there is a stack pointer
                                // but it is used as the source for a copy so it is ok
                                .strides = strides,
                                .offset = new_offset,
                            },
                        },
                    },
                },
                .constant = self.meta.constant,
                .label = self.meta.label,
                .grad_fn = autograd.noGrad,
            };
            const Out = typing.View(Self, new_shape);
            const out: *Out = allocator.create(Out) catch unreachable;
            out.* = .{
                .meta = meta,
                .strides = strides,
                .offset = new_offset,
            };
            return out;
        }

        ///Apply an elementwise unary operation
        pub fn applyUnaryOp(
            self: *const Self,
            comptime op: ops.UnaryOp,
            allocator: std.mem.Allocator,
        ) *const typing.UnaryOpResult(Self, op) {
            const meta = allocator.create(shared.Metadata) catch unreachable;
            meta.* = .{
                .instr = .{
                    .UnaryOp = .{
                        .in = .{self.toAnyTensor()},
                        .op = op,
                    },
                },
                .constant = self.meta.constant,
                .label = self.meta.label,
                .grad_fn = struct {
                    pub fn gradFnImpl(grad_out: anytype, param_grads: []const *const AnyTensor) []const *const AnyTensor {
                        return autograd.unaryGrad(op, self, grad_out, param_grads);
                    }
                }.gradFnImpl,
            };
            const Out = comptime typing.UnaryOpResult(Self, op);
            const out: *Out = allocator.create(Out) catch unreachable;
            out.* = .{
                .meta = meta,
                .strides = self.strides,
                .offset = self.offset,
            };
            return out;
        }

        /// Apply an elementwise binary operation on two arrays, with broadcasting
        /// a and b must have the same "dtype class" meaning both must be float, bool, or int
        /// though different sizes are allowed.
        pub fn applyBinaryOp(
            self: *const Self,
            other: anytype,
            comptime op: ops.BinaryOp,
            allocator: std.mem.Allocator,
        ) *const typing.BinaryOpResult(Self, @TypeOf(other), op) {
            const Other = comptime typing.AsTensorType(@TypeOf(other));
            const bc_shape = comptime utils.broadcastShape(Self.shape, Other.shape);
            const a = F.expand(self, bc_shape, allocator);
            const b = F.expand(other, bc_shape, allocator);

            const meta = allocator.create(shared.Metadata) catch unreachable;
            meta.* = .{
                .instr = .{
                    .BinaryOp = .{
                        .in = .{ a.toAnyTensor(), b.toAnyTensor() },
                        .op = op,
                    },
                },
                .grad_fn = struct {
                    pub fn gradFnImpl(grad_out: anytype, param_grads: []const *const AnyTensor) []const *const AnyTensor {
                        return autograd.binaryGrad(op, a, b, grad_out, param_grads);
                    }
                }.gradFnImpl,
                .constant = a.meta.constant and b.meta.constant,
                .label = null,
            };
            const Out = comptime typing.BinaryOpResult(Self, @TypeOf(other), op);
            const out: *Out = allocator.create(Out) catch unreachable;
            out.* = .{ .meta = meta };
            return out;
        }

        /// Perform a reduction across 1 or more (or all) dimensions of a
        /// Dimensions to reduce can be passed as a int for 1 dim, tuple for multiple dims, or null/void for all dims
        pub fn applyReduceOp(
            self: *const Self,
            comptime op: ops.ReduceOp,
            comptime reduce_dims: anytype,
            allocator: std.mem.Allocator,
        ) *const typing.ReduceOpResult(Self, reduce_dims) {
            // Use u16 here because []const u8 shows up as a string
            const reduce_dims_array: []const u8 = switch (@typeInfo(@TypeOf(reduce_dims))) {
                .ComptimeInt, .Int => &[1]u8{signedToUnsignedDim(reduce_dims)},
                .Void => @as([ndims]u8, std.simd.iota(u8, ndims))[0..],
                else => &reduce_dims,
            };
            const reduce_dims_mask: [ndims]bool = switch (@typeInfo(@TypeOf(reduce_dims))) {
                .ComptimeInt, .Int => blk: {
                    var tmp_mask: [ndims]bool = [_]bool{false} ** ndims;
                    const dim = reduce_dims;
                    tmp_mask[signedToUnsignedDim(dim)] = true;
                    break :blk tmp_mask;
                },
                .Void => [_]bool{true} ** ndims,
                else => blk: {
                    var tmp_mask: [ndims]bool = [_]bool{false} ** ndims;
                    for (reduce_dims) |dim| {
                        tmp_mask[signedToUnsignedDim(dim)] = true;
                    }
                    break :blk tmp_mask;
                },
            };
            const meta = allocator.create(shared.Metadata) catch unreachable;
            meta.* = .{
                .instr = .{
                    .ReduceOp = .{
                        .in = .{self.toAnyTensor()},
                        .op = op,
                        .args = .{
                            .dims = reduce_dims_array,
                            .mask = &reduce_dims_mask,
                        },
                    },
                },
                .constant = false,
                .label = self.meta.label,
                .grad_fn = autograd.noGrad,
            };
            const Out = comptime typing.ReduceOpResult(Self, reduce_dims);
            const out: *Out = allocator.create(Out) catch unreachable;
            out.* = .{ .meta = meta };
            return out;
        }

        pub fn backwards(self: *const Self, allocator: std.mem.Allocator) ![]const *const AnyTensor {
            return autograd.backwards(self, allocator);
        }
    };
}
