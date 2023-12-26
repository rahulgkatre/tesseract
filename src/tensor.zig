const std = @import("std");
const Allocator = std.mem.Allocator;
const comptimePrint = std.fmt.comptimePrint;
const ComptimeUtils = @import("tensor_comptime_utils.zig");
const TensorStorage = @import("tensor_storage.zig").TensorStorage;

pub fn tensor(comptime dtype: type, comptime shape: anytype) ret: {
    const strides = ComptimeUtils.defaultStrides(shape.len, shape);
    break :ret Tensor(dtype, shape.len, shape, strides);
} {
    const strides = comptime ComptimeUtils.defaultStrides(shape.len, shape);
    return comptime Tensor(dtype, shape.len, shape, strides).lazy_init(null);
}

pub const MapOp = enum { Neg };
pub const ZipOp = enum { Add };
pub const ReduceOp = enum { Sum };
pub const Op = union(enum) { map: MapOp, zip: ZipOp, reduce: ReduceOp };

fn OpInput(comptime ndims: u8) type {
    return union { tensor: struct {
        ndims: u8,
        shape: [ndims]usize,
        strides: [ndims]usize,
        ptr: *const anyopaque,
    }, arr: [ndims]usize, val: usize };
}
fn History(comptime ndims: u8) type {
    return struct { op: Op, inputs: [2]OpInput(ndims) };
}

pub fn extend_shape(comptime in_ndims: u8, in_shape: [in_ndims]usize, comptime out_ndims: u8) [out_ndims]usize {
    var out_shape: [out_ndims]usize = undefined;
    @memset(&out_shape, 1);
    @memcpy(out_shape[(out_ndims - in_ndims)..], &in_shape);
    return out_shape;
}

pub fn extend_strides(comptime in_ndims: u8, in_strides: [in_ndims]usize, comptime out_ndims: u8) [out_ndims]usize {
    var out_strides: [out_ndims]usize = undefined;
    @memset(&out_strides, 0);
    @memcpy(out_strides[(out_ndims - in_ndims)..], &in_strides);
    return out_strides;
}

// pub fn History(comptime last_op: Op, comptime last_inputs: anytype) type {
//     return struct { last_fn: Op = last_op, last_inputs: @TypeOf(last_inputs) = last_inputs };
// }

// fn History(comptime last_op: Op, comptime last_args: anytype) type {
//     return struct { last_op = last_op, last_args = last_args };
// }

// Each history node will have a known size
// This is because tensors of the same ndims will have the same size
// And all tensors that are inputs to map, zip, reduce will have the same ndims
// If there is a way to specify stuct

// TODO: Add a device type param here
// Should be easy to add this type param everywhere as the device will remain the same unless a to(x)
// method is called
fn Tensor(comptime dtype: type, comptime _ndims: u8, comptime _shape: @Vector(_ndims, usize), comptime _strides: @Vector(_ndims, usize)) type {
    switch (@typeInfo(dtype)) {
        .Bool => {},
        .ComptimeInt => {},
        .Int => {},
        .ComptimeFloat => {},
        .Float => {},
        else => @compileError("Non-numeric or non-bool tensor dtype not supported, received " ++ @typeName(dtype)),
    }
    return struct {
        const Self = @This();
        pub const ndims: u8 = _ndims;
        pub const shape: @Vector(ndims, usize) = _shape;
        pub const strides: @Vector(ndims, usize) = _strides;
        ndims: u8 = ndims,
        shape: @Vector(ndims, usize) = shape,
        strides: @Vector(ndims, usize) = strides,
        storage: ?*TensorStorage(dtype, size()),
        owns_storage: bool,
        real: bool,
        allocator: ?Allocator,
        history: ?History(ndims),

        pub fn size() usize {
            // Used to determine the size of the underlying storage
            return comptime blk: {
                var _size: usize = @reduce(.Mul, shape);
                if (_size == 0) {
                    @compileError("Illegal tensor size of 0");
                }
                break :blk _size;
            };
        }
        fn broadcast_index(bc_ndims: u8, bc_index: [bc_ndims]usize) [ndims]usize {
            // Determine the index in the current tensor given an index in the broadcasted tensor
            // If the current tensor has size of 1 in a dimension, then the index must be 0
            // Otherwise it will be what the broadcasted index is
            return comptime blk: {
                const index: [ndims]usize = undefined;
                for (0..ndims) |i| {
                    index[bc_ndims - i - 1] = if (shape[ndims - i - 1] == 1) 0 else bc_index[bc_ndims - i - 1];
                }
                break :blk index;
            };
        }
        pub inline fn is_contiguous(_: *const Self) bool {
            return comptime ComptimeUtils.isContiguous(ndims, strides);
        }
        pub inline fn lazy_init(history: ?History(ndims)) Self {
            return comptime .{ .storage = null, .owns_storage = false, .allocator = null, .real = false, .history = history };
        }
        pub fn realize(_: *const Self, storage: ?*TensorStorage(dtype, size()), allocator: Allocator) !Self {
            // TODO: Make this async to block thread until tensor is computed
            // Current impl of realize does not trace back up its compute graph to actually get values
            const t = try allocator.create(Self);
            t.* = .{ .storage = storage orelse try TensorStorage(dtype, size()).init(allocator), .allocator = allocator, .real = true, .owns_storage = true, .history = null };
            return t.*;
        }
        pub fn real_init(storage: ?*TensorStorage(dtype, size()), allocator: Allocator) !*const Self {
            const t = try allocator.create(Self);
            errdefer allocator.destroy(t);
            t.* = .{ .storage = storage orelse try TensorStorage(dtype, size()).init(allocator), .owns_storage = storage == null, .allocator = allocator, .real = true, .history = null };
            return t;
        }
        pub fn deinit(self: *const Self) void {
            if (self.real and self.owns_storage) {
                self.storage.?.deinit();
            }
        }
        pub fn permute(_: *const Self, comptime perm: @Vector(ndims, u8)) t: {
            const permute_shape = ComptimeUtils.permuteArray(ndims, shape, perm);
            const permute_strides = ComptimeUtils.permuteArray(ndims, strides, perm);
            break :t Tensor(dtype, ndims, permute_shape, permute_strides);
        } {
            const new_shape = comptime ComptimeUtils.permuteArray(ndims, shape, perm);
            const new_strides = comptime ComptimeUtils.permuteArray(ndims, strides, perm);
            return Tensor(dtype, ndims, new_shape, new_strides).lazy_init(null);
        }
        pub fn mock_map_fn(self: *const Self, map_op: MapOp) Self {
            // Mock implementations of three kinds of tensor ops
            // Map is 1 to 1 (neg, log, exp)
            const out_history: History(ndims) = .{ .op = @unionInit(Op, "map", map_op), .inputs = .{@unionInit(OpInput(ndims), "tensor", .{ shape, strides, self })} };
            return lazy_init(out_history);
        }
        pub fn mock_zip_fn(self: *const Self, zip_op: ZipOp, other: anytype) t: {
            // Zip is 2 to 1 (add, mul, xor)
            const out_shape = ComptimeUtils.shapeTypeBroadcast(Self, @TypeOf(other.*));
            const out_ndims = out_shape.len;
            const out_strides = ComptimeUtils.defaultStrides(out_ndims, out_shape);
            break :t Tensor(dtype, out_ndims, out_shape, out_strides);
        } {
            const other_ndims = @field(other.*, "ndims");
            const other_shape = @field(other.*, "shape");
            const other_strides = @field(other.*, "strides");
            const out_shape = comptime ComptimeUtils.shapeTypeBroadcast(Self, @TypeOf(other.*));
            const out_ndims = out_shape.len;
            const out_strides = ComptimeUtils.defaultStrides(out_ndims, out_shape);
            const out_history: History(out_ndims) = .{ .op = @unionInit(Op, "zip", zip_op), .inputs = .{ @unionInit(OpInput(ndims), "tensor", .{ .ndims = ndims, .shape = extend_shape(ndims, shape, out_ndims), .strides = extend_strides(ndims, strides, out_ndims), .ptr = self }), @unionInit(OpInput(ndims), "tensor", .{ .ndims = other_ndims, .shape = extend_shape(other_ndims, other_shape, out_ndims), .strides = extend_strides(other_ndims, other_strides, out_ndims), .ptr = other }) } };
            return Tensor(dtype, out_ndims, out_shape, out_strides).lazy_init(out_history);
        }
        pub fn mock_reduce_fn(self: *const Self, reduce_op: ReduceOp, comptime reduce_dim: usize) t: {
            // Reduce is many to 1 and collapses a dimension to 1 (sum, prod, max, etc.)
            const out_shape = ComptimeUtils.getReducedShape(ndims, shape, reduce_dim);
            const out_strides = ComptimeUtils.defaultStrides(ndims, out_shape);
            break :t Tensor(dtype, ndims, out_shape, out_strides);
        } {
            const out_shape = comptime ComptimeUtils.getReducedShape(ndims, shape, reduce_dim);
            const out_strides = ComptimeUtils.defaultStrides(ndims, out_shape);
            const out_history: History(ndims) = .{ .op = @unionInit(Op, "reduce", reduce_op), .inputs = .{ @unionInit(OpInput(ndims), "tensor", .{ .ndims = ndims, .shape = shape, .strides = strides, .ptr = self }), @unionInit(OpInput(ndims), "val", reduce_dim) } };
            return Tensor(dtype, out_shape.len, out_shape, out_strides).lazy_init(out_history);
        }
    };
}
