const std = @import("std");
const Allocator = std.mem.Allocator;
const comptimePrint = std.fmt.comptimePrint;
const Utils = @import("utils.zig");
const TensorStorage = @import("storage.zig").TensorStorage;
const ops = @import("ops.zig");
const GraphTensor = @import("graph.zig").GraphTensor;

pub fn tensor(comptime dtype: type, comptime shape: anytype) ret: {
    const strides = Utils.defaultStrides(shape.len, shape);
    break :ret Tensor(dtype, shape.len, shape, strides);
} {
    const strides = comptime Utils.defaultStrides(shape.len, shape);
    return comptime Tensor(dtype, shape.len, shape, strides).init();
}

// TODO: Add a device type param here
// Should be easy to add this type param everywhere as the device will remain the same unless a to(x)
// method is called
fn Tensor(comptime dtype: type, comptime _ndims: u8, comptime _shape: [_ndims]usize, comptime _strides: [_ndims]usize) type {
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
        
        // These just take on the value of the generic arguments
        pub const ndims: u8 = _ndims;
        pub const shape: [ndims]usize = _shape;
        pub const strides: [ndims]usize = _strides;
        ndims: u8 = ndims,
        shape: [ndims]usize = shape,
        strides: [ndims]usize = strides,

        // These need to be populated during init()
        graph_tensor: GraphTensor,
        storage: ?*TensorStorage(dtype, size()),
        owns_storage: bool,
        real: bool,
        allocator: ?Allocator,

        pub fn init() Self {
            const impl = struct {
                pub fn permute(comptime ptr: *const GraphTensor, comptime perm: []u8) GraphTensor {
                    const self = @fieldParentPtr(Self, "graph_tensor", ptr);
                    return self.permute(perm[0..ndims]).graph_tensor;
                }
                pub fn map(comptime ptr: *const GraphTensor, comptime map_op: ops.MapOp) GraphTensor {
                    const self = @fieldParentPtr(Self, "graph_tensor", ptr);
                    return self.map(map_op).graph_tensor;
                }
                pub fn zip(comptime ptr: *const GraphTensor, comptime zip_op: ops.ZipOp, comptime other_ptr: anytype) GraphTensor {
                    const self = @fieldParentPtr(Self, "graph_tensor", ptr);
                    return self.zip(zip_op, other_ptr).graph_tensor;
                }
                pub fn reduce(comptime ptr: *const GraphTensor, comptime reduce_op: ops.ReduceOp, comptime reduce_dim: u8) GraphTensor {
                    const self = @fieldParentPtr(Self, "graph_tensor", ptr);
                    return self.reduce(reduce_op, reduce_dim).graph_tensor;
                }
            };
            return .{ 
                .storage = null, 
                .owns_storage = false, 
                .allocator = null, 
                .real = false, 
                .graph_tensor = .{
                    .permute_fn = impl.permute,
                    .map_fn = impl.map,
                    .zip_fn = impl.zip,
                    .reduce_fn = impl.reduce,
                },
            };
        }
        pub fn realize(self: *Self, storage: ?*TensorStorage(dtype, size()), allocator: Allocator) !void {
            // TODO: Make this async to block thread until tensor is computed
            // Current impl of realize does not trace back up its compute graph to actually get values
            self.storage = storage orelse try TensorStorage(dtype, size()).init(allocator);
            self.allocator = allocator;
            self.real = true;
            self.owns_storage = storage == null;
        }
        pub fn deinit(self: *const Self) void {
            if (self.real and self.owns_storage) {
                self.storage.?.deinit();
            }
        }

        pub fn size() usize {
            // Used to determine the size of the underlying storage
            return comptime blk: {
                const shape_vec: @Vector(ndims, usize) = shape;
                var _size: usize = @reduce(.Mul, shape_vec);
                if (_size == 0) @compileError("Illegal tensor size of 0");
                break :blk _size;
            };
        }
        fn broadcastIndex(bc_ndims: u8, bc_index: [bc_ndims]usize) [ndims]usize {
            // Determine the index in the current tensor given an index in the broadcasted tensor
            // If the current tensor has size of 1 in a dimension, then the index must be 0
            // Otherwise it will be what the broadcasted index is
            const index: [ndims]usize = undefined;
            for (0..ndims) |d| index[bc_ndims - d - 1] = if (shape[ndims - d - 1] == 1) 0 else bc_index[bc_ndims - d - 1];
            return index;
        }
        pub fn flatIndex(index: [ndims]usize) usize {
            // Convert a multidimensional index into a single dimensional index        
            var flat_index: usize = 0;
            for (0..ndims) |d| flat_index += index[d] * strides[d];
            return flat_index;
        }
        pub inline fn isContiguous(_: *const Self) bool {
            // The information for contiguous is in the type itself
            return comptime Utils.isContiguous(ndims, strides);
        }
        pub fn permute(comptime _: *const Self, comptime perm: [ndims]u8) t: {
            // First permute to compute the return type
            const permute_shape = Utils.permuteArray(ndims, shape, perm);
            const permute_strides = Utils.permuteArray(ndims, strides, perm);
            break :t Tensor(dtype, ndims, permute_shape, permute_strides);
        } {
            // Then permute to get the actual object of the return type
            const new_shape = comptime Utils.permuteArray(ndims, shape, perm);
            const new_strides = comptime Utils.permuteArray(ndims, strides, perm);
            return Tensor(dtype, ndims, new_shape, new_strides).init();
        }
        pub fn map(comptime self: *const Self, comptime map_op: ops.MapOp) Self {
            _ = map_op;
            _ = self;
            // Map is 1 to 1 (neg, log, exp) so return type is the same
            var out = init();
            // TODO: In the graph tensor object add a history field and populate it with the function and arguments
            // out.graph_tensor.history = .{ 
            //     .op = .{ .MapOp = map_op }, 
            //     .inputs = .{
            //         .{ .Tensor = .{ shape, strides, self } }
            //     },
            // };
            return out;
        }
        pub fn zip(comptime self: *const Self, comptime zip_op: ops.ZipOp, comptime other: anytype) t: {
            // Zip is 2 to 1 (add, mul, xor)
            // Zip can broadcast so return type needs to be computed
            const out_shape = Utils.shapeBroadcast(Self, @TypeOf(other.*));
            const out_ndims = out_shape.len;
            const out_strides = Utils.defaultStrides(out_ndims, out_shape);
            break :t Tensor(dtype, out_ndims, out_shape, out_strides);
        } {
            _ = zip_op;
            _ = self;
            const out_shape = comptime Utils.shapeBroadcast(Self, @TypeOf(other.*));
            const out_ndims = out_shape.len;
            const out_strides = Utils.defaultStrides(out_ndims, out_shape);
            var out = Tensor(dtype, out_ndims, out_shape, out_strides).init();
            // TODO: In the graph tensor object add a history field and populate it with the function and arguments
            // out.graph_tensor.history = .{ 
            //     .op = .{ .ZipOp = zip_op }, 
            //     .inputs = .{ 
            //         .{ .Tensor = &self.graph_tensor }, 
            //         .{ .Tensor = &other.graph_tensor },
            //     },
            // };
            return out;
        }
        pub fn reduce(comptime self: *const Self, comptime reduce_op: ops.ReduceOp, comptime reduce_dim: usize) t: {
            // Reduce is many to 1 and collapses a dimension to 1 (sum, prod, max, etc.)
            const out_shape = Utils.getReducedShape(ndims, shape, reduce_dim);
            const out_strides = Utils.defaultStrides(ndims, out_shape);
            break :t Tensor(dtype, ndims, out_shape, out_strides);
        } {
            _ = reduce_op;
            _ = self;
            const out_shape = comptime Utils.getReducedShape(ndims, shape, reduce_dim);
            const out_strides = Utils.defaultStrides(ndims, out_shape);
            var out = Tensor(dtype, out_shape.len, out_shape, out_strides).init();
            // TODO: In the graph tensor object add a history field and populate it with the function and arguments
            // out.graph_tensor.history =  .{ 
            //     .op = .{ .ReduceOp = reduce_op }, 
            //     .inputs = .{ 
            //         .{ .Tensor = &self.graph_tensor }, 
            //         .{ .Value = reduce_dim },
            //     },
            // };
            return out;
        }
    };
}
