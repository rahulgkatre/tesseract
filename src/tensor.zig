const std = @import("std");
const Allocator = std.mem.Allocator;
const comptimePrint = std.fmt.comptimePrint;
const Utils = @import("tensor_comptime_utils.zig");
const TensorStorage = @import("tensor_storage.zig").TensorStorage;

pub const Type = enum {
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
};

pub fn tensor(comptime _dtype: Type, comptime shape: anytype) ret: {
    const strides = Utils.defaultStrides(shape.len, shape);
    break :ret Tensor(_dtype, shape.len, shape, strides);
} {
    const strides = comptime Utils.defaultStrides(shape.len, shape);
    return comptime Tensor(_dtype, shape.len, shape, strides).lazyInit();
}

pub const MapOp = enum { Neg };
pub const ZipOp = enum { Add };
pub const ReduceOp = enum { Sum };
pub const Op = union(enum) { MapOp: MapOp, ZipOp: ZipOp, ReduceOp: ReduceOp };

const Input = enum {
    Tensor,
    Array,
    Value,
};


const MAX_NDIMS = 8;
const OpInput = union(Input) { 
    Tensor: *const GraphTensor,
    Array: [MAX_NDIMS]usize, 
    Value: usize 
};

// pub fn getInputComptime(comptime input: anytype) t: {
//     break :t switch (input) {
//         .Tensor => |_tensor| *Tensor(_tensor.dtype, _tensor.ndims, _tensor.shape[_tensor.shape.len-_tensor.ndims.._tensor.shape.len].*, _tensor.strides[_tensor.shape.len-_tensor.ndims.._tensor.shape.len].*),
//         .Array => |array| [array.len]usize,
//         .Value => usize
//     };
// } {
//     return switch (input) {
//         .Tensor => |_tensor| @constCast(@alignCast(@ptrCast(_tensor.ptr))),
//         .Array => |array| array,
//         .Value => |value| value,
//     };
// }

// fn getInputRuntime(comptime ndims: u8, input: OpInput(ndims)) t: {
//     break :t switch (input) {
//         .Tensor => |_tensor| *Tensor(_tensor.dtype, _tensor.ndims, _tensor.shape[_tensor.shape.len-_tensor.ndims.._tensor.shape.len].*, _tensor.strides[_tensor.shape.len-_tensor.ndims.._tensor.shape.len].*),
//         .Array => |array| [array.len]usize,
//         .Value => usize
//     };
// } {
//     return switch (input) {
//         .Tensor => |_tensor| @constCast(@alignCast(@ptrCast(_tensor.ptr))),
//         .Array => |array| array,
//         .Value => |value| value,
//     };
// }

const History = struct { op: Op, inputs: [2]OpInput };

pub fn extendShape(comptime in_ndims: u8, in_shape: [in_ndims]usize, comptime out_ndims: u8) [out_ndims]usize {
    var out_shape: [out_ndims]usize = undefined;
    @memset(&out_shape, 1);
    @memcpy(out_shape[(out_ndims - in_ndims)..], &in_shape);
    return out_shape;
}

pub fn extendStrides(comptime in_ndims: u8, in_strides: [in_ndims]usize, comptime out_ndims: u8) [out_ndims]usize {
    var out_strides: [out_ndims]usize = undefined;
    @memset(&out_strides, 0);
    @memcpy(out_strides[(out_ndims - in_ndims)..], &in_strides);
    return out_strides;
}

// Each history node will have a known size
// This is because tensors of the same ndims will have the same size
// And all tensors that are inputs to map, zip, reduce will have the same ndims
// If there is a way to specify stuct

// TODO: Add a device type param here
// Should be easy to add this type param everywhere as the device will remain the same unless a to(x)
// method is called

const GraphTensor = struct {
    const Self = @This();
    ndims: u8,
    dtype: Type,
    shape: []const usize,
    strides:[]const usize,
    history: ?History,
    pub fn print_graph(self: *const Self) void {
        if (self.history != null) {
            const history = self.history.?;
            std.debug.print("\n{any}", .{history.op});
            var next_tensor: GraphTensor = undefined;
            for (0..history.inputs.len) |i| {
                switch (history.inputs[i]) {
                    .Tensor => |_tensor| {
                        const t: *GraphTensor = @constCast(@alignCast(@ptrCast(_tensor)));
                        std.debug.print("\ttensor<", .{});
                        for (t.shape) |s| {
                            std.debug.print("{any}x", .{s});
                        }
                        std.debug.print("{any}>", .{t.dtype});
                        next_tensor = t.*;
                    },
                    .Array => |array| {
                        std.debug.print("\t{any}", .{array});
                    },
                    .Value => |value| {
                        std.debug.print("\t{any}", .{value});
                    },
                }
            }
            std.debug.print("", .{});
            next_tensor.print_graph();
        }
    }
};

fn Tensor(comptime _dtype: Type, comptime _ndims: u8, comptime _shape: [_ndims]usize, comptime _strides: [_ndims]usize) type {
    const dtype: type = switch(_dtype) {
        .Float16 => f16,
        .Float32 => f32,
        .Float64 => f64,
        .Int8 => i8,
        .Int16 => i16,
        .Int32 => i32,
        .Int64 => i64,
    };
    // switch (@typeInfo(dtype)) {
    //     .Bool => {},
    //     .ComptimeInt => {},
    //     .Int => {},
    //     .ComptimeFloat => {},
    //     .Float => {},
    //     else => @compileError("Non-numeric or non-bool tensor dtype not supported, received " ++ @typeName(dtype)),
    // }
    return struct {
        const Self = @This();
        pub const ndims: u8 = _ndims;
        pub const shape: @Vector(ndims, usize) = _shape;
        pub const strides: @Vector(ndims, usize) = _strides;
        ndims: u8 = ndims,
        shape: @Vector(ndims, usize) = shape,
        strides: @Vector(ndims, usize) = strides,
        graph_tensor: GraphTensor = .{
            .ndims = _ndims,
            .dtype = _dtype,
            .shape = _shape[0..],
            .strides = _strides[0..],
            .history = null,
        },
        storage: ?*TensorStorage(dtype, size()),
        owns_storage: bool,
        real: bool,
        allocator: ?Allocator,
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
        fn broadcastIndex(bc_ndims: u8, bc_index: [bc_ndims]usize) [ndims]usize {
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
        pub inline fn isContiguous(_: *const Self) bool {
            // The information for contiguous is in the type itself
            return comptime Utils.isContiguous(ndims, strides);
        }
        pub inline fn lazyInit() Self {
            return .{ .storage = null, .owns_storage = false, .allocator = null, .real = false, };
        }
        pub fn realize(self: *Self, storage: ?*TensorStorage(dtype, size()), allocator: Allocator) !void {
            // TODO: Make this async to block thread until tensor is computed
            // Current impl of realize does not trace back up its compute graph to actually get values
            self.storage = storage orelse try TensorStorage(dtype, size()).init(allocator);
            self.allocator = allocator;
            self.real = true;
            self.owns_storage = storage == null;
        }
        pub fn realInit(storage: ?*TensorStorage(dtype, size()), allocator: Allocator) !Self {
            return .{ 
                .storage = storage orelse try TensorStorage(dtype, size()).init(allocator), 
                .owns_storage = storage == null, 
                .allocator = allocator, 
                .real = true, 
                .history = null 
            };
        }
        pub fn deinit(self: *const Self) void {
            if (self.real and self.owns_storage) {
                self.storage.?.deinit();
            }
        }
        pub fn permute(self: *const Self, comptime perm: @Vector(ndims, u8)) t: {
            const permute_shape = Utils.permuteArray(ndims, shape, perm);
            const permute_strides = Utils.permuteArray(ndims, strides, perm);
            break :t Tensor(_dtype, ndims, permute_shape, permute_strides);
        } {
            _ = self;
            const new_shape = comptime Utils.permuteArray(ndims, shape, perm);
            const new_strides = comptime Utils.permuteArray(ndims, strides, perm);
            return Tensor(_dtype, ndims, new_shape, new_strides).lazyInit(null);
        }
        pub fn map(self: *const Self, map_op: MapOp) Self {
            // Mock implementations of three kinds of tensor ops
            // Map is 1 to 1 (neg, log, exp)
            var out = lazyInit();
            out.graph_tensor.history = .{ 
                .op = .{ .MapOp = map_op }, 
                .inputs = .{
                    .{ .Tensor = .{ shape, strides, self } }
                }
            };
            return out;
        }
        pub fn zip(self: *const Self, zip_op: ZipOp, other: anytype) t: {
            // Zip is 2 to 1 (add, mul, xor)
            const out_shape = Utils.shapeBroadcast(Self, @TypeOf(other.*));
            const out_ndims = out_shape.len;
            const out_strides = Utils.defaultStrides(out_ndims, out_shape);
            break :t Tensor(_dtype, out_ndims, out_shape, out_strides);
        } {
            const out_shape = comptime Utils.shapeBroadcast(Self, @TypeOf(other.*));
            const out_ndims = out_shape.len;
            const out_strides = Utils.defaultStrides(out_ndims, out_shape);
            var out = Tensor(_dtype, out_ndims, out_shape, out_strides).lazyInit();
            out.graph_tensor.history = .{ 
                .op = .{ .ZipOp = zip_op }, 
                .inputs = .{ 
                    .{ .Tensor = &self.graph_tensor }, 
                    .{ .Tensor = &other.graph_tensor }
                }
            };
            return out;
        }
        pub fn reduce(self: *const Self, reduce_op: ReduceOp, comptime reduce_dim: usize) t: {
            // Reduce is many to 1 and collapses a dimension to 1 (sum, prod, max, etc.)
            const out_shape = Utils.getReducedShape(ndims, shape, reduce_dim);
            const out_strides = Utils.defaultStrides(ndims, out_shape);
            break :t Tensor(_dtype, ndims, out_shape, out_strides);
        } {
            const out_shape = comptime Utils.getReducedShape(ndims, shape, reduce_dim);
            const out_strides = Utils.defaultStrides(ndims, out_shape);
            var out = Tensor(_dtype, out_shape.len, out_shape, out_strides).lazyInit();
            out.graph_tensor.history =  .{ 
                .op = .{ .ReduceOp = reduce_op }, 
                .inputs = .{ 
                    .{ .Tensor = &self.graph_tensor }, 
                    .{ .Value = reduce_dim }
                }
            };
            return out;
        }
        
    };
}
