const std = @import("std");
const Allocator = std.mem.Allocator;
const comptimePrint = std.fmt.comptimePrint;
const utils = @import("utils.zig");
const TensorStorage = @import("storage.zig").TensorStorage;
const ops = @import("ops.zig");
const GraphTensor = @import("graph.zig").GraphTensor;

pub fn tensor(comptime dtype: type, comptime shape: anytype) utils.defaultStridedTensor(dtype, shape.len, shape) {
    // Utility function to create a tensor from an input shape (tuple or array of usize)
    // Infers strides from shape so it will be contiguous
    // Tensors created using this function must be realized manually as they are endpoints of the compute graph
    return utils.defaultStridedTensor(dtype, shape.len, shape).init();
}

// TODO: Add a device type param here
// Should be easy to add this type param everywhere as the device will remain the same unless a to_device() method is called
pub fn Tensor(comptime _dtype: type, comptime _ndims: u8, comptime _shape: [_ndims]usize, comptime _strides: [_ndims]usize) type {
    switch (@typeInfo(_dtype)) {
        .Bool, .ComptimeInt, .Int, .ComptimeFloat, .Float => {},
        else => @compileError("Non-numeric or non-bool tensor dtype not supported, received " ++ @typeName(_dtype)),
    }
    return struct {
        const Self = @This();
        // These just take on the value of the generic arguments
        // Save the dtype here as it is needed for some comptime functions, accessed via @field
        pub const dtype: type = _dtype;
        pub const ndims: u8 = _ndims;
        pub const shape: [ndims]usize = _shape;
        pub const strides: [ndims]usize = _strides;
        pub const size = utils.size(ndims, shape);
        ndims: u8 = ndims,
        shape: [ndims]usize = shape,
        strides: [ndims]usize = strides,
        size: usize = size,

        // These need to be populated during init()
        graph_tensor: GraphTensor,
        storage: ?*TensorStorage(dtype, size),
        owns_storage: bool,
        real: bool,
        allocator: ?Allocator,

        pub fn init() Self {
            const impl = struct {
                // TODO: Make a new function that takes a ShapeOp and executes it
                // pub fn permute(comptime ptr: *const GraphTensor, comptime perm: []u8) GraphTensor {
                //     const self = @fieldParentPtr(Self, "graph_tensor", ptr);
                //     return self.permute(perm[0..ndims]).graph_tensor;
                // }
                pub fn print_info(comptime ptr: *const GraphTensor) void {
                    std.debug.print("tensor<", .{});
                    inline for (0..ndims) |d| {
                        std.debug.print("{any},", .{_shape[d]});
                    }
                    std.debug.print("{any}>, id: {any}", .{ _dtype, @intFromPtr(ptr) });
                }
                pub fn map(comptime ptr: *const GraphTensor, comptime map_op: ops.MapOp) GraphTensor {
                    const self = @fieldParentPtr(Self, "graph_tensor", ptr);
                    return self.map(map_op).graph_tensor;
                }
                pub fn zip(comptime ptr: *const GraphTensor, comptime zip_op: ops.ZipOp, comptime b: anytype) GraphTensor {
                    const self = @fieldParentPtr(Self, "graph_tensor", ptr);
                    return self.zip(zip_op, b).graph_tensor;
                }
                pub fn reduce(comptime ptr: *const GraphTensor, comptime op: ops.ReduceOp, comptime reduce_dim: u8) GraphTensor {
                    const self = @fieldParentPtr(Self, "graph_tensor", ptr);
                    return self.reduce(op, reduce_dim).graph_tensor;
                }
            };
            return .{
                .storage = null,
                .owns_storage = false,
                .allocator = null,
                .real = false,
                .graph_tensor = .{
                    // .permute_fn = impl.permute,
                    .print_info_fn = impl.print_info,
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
        pub inline fn isContiguous(_: *const Self) bool {
            // The information for contiguous is in the type itself
            return comptime utils.isContiguous(ndims, strides);
        }
        pub fn permute(comptime _: *const Self, comptime perm: [ndims]u8) utils.permutedTensor(Self, perm) {
            return utils.permutedTensor(Self, perm).init();
        }
        pub fn map(comptime self: *const Self, comptime op: ops.MapOp) Self {
            var out = init();
            out.graph_tensor.last_op = .{
                .MapOp = .{
                    .op = op,
                    .a = &self.graph_tensor,
                }
            };
            return out;
        }
        pub fn zip(comptime self: *const Self, comptime op: ops.ZipOp, comptime other: anytype) utils.broadcastedTensor(Self, @TypeOf(other)) {
            var out = utils.broadcastedTensor(Self, @TypeOf(other)).init();
            out.graph_tensor.last_op = .{
                .ZipOp = .{
                    .op = op,
                    .a = &self.graph_tensor,
                    .b = &other.graph_tensor,
                }
            };
            return out;
        }
        pub fn reduce(comptime self: *const Self, comptime op: ops.ReduceOp, comptime reduce_dim: usize) utils.reducedTensor(Self, reduce_dim) {
            var out = utils.reducedTensor(Self, reduce_dim).init();
            out.graph_tensor.last_op = .{
                .ReduceOp = .{
                    .op = op,
                    .a = &self.graph_tensor,
                    .reduce_dim = reduce_dim,
                }
            };
            return out;
        }
    };
}
