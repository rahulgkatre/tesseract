const std = @import("std");
const Allocator = std.mem.Allocator;
const comptimePrint = std.fmt.comptimePrint;
const utils = @import("utils.zig");
const TensorStorage = @import("storage.zig").TensorStorage;
const ops = @import("ops.zig");
const GraphTensor = @import("graph.zig").GraphTensor;

pub fn tensor(comptime dtype: type, comptime shape: anytype) DefaultStridedTensor(dtype, shape.len, shape) {
    // Utility function to create a tensor from an input shape (tuple or array of usize)
    // Infers strides from shape so it will be contiguous
    // Tensors created using this function must be realized manually as they are endpoints of the compute graph
    return comptime DefaultStridedTensor(dtype, shape.len, shape).init();
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
                pub fn print_info(ptr: *const GraphTensor) void {
                    std.debug.print("tensor<", .{});
                    for (0..ndims) |d| {
                        std.debug.print("{any},", .{_shape[d]});
                    }
                    std.debug.print("{any}>, id: {any}", .{ _dtype, @intFromPtr(ptr) });
                }
                pub fn eval_map(ptr: *const GraphTensor, op_call: ops.OpCall) void {
                    const self = @fieldParentPtr(Self, "graph_tensor", ptr);
                    return self.eval_map(op_call);
                }
                pub fn eval_zip(ptr: *const GraphTensor, op_call: ops.OpCall) void {
                    const self = @fieldParentPtr(Self, "graph_tensor", ptr);
                    return self.eval_zip(op_call);
                }
                pub fn reduce(ptr: *const GraphTensor, op_call: ops.OpCall) void {
                    const self = @fieldParentPtr(Self, "graph_tensor", ptr);
                    return self.eval_reduce(op_call);
                }
            };
            return .{
                .storage = null,
                .owns_storage = false,
                .allocator = null,
                .real = false,
                .graph_tensor = .{
                    .print_info_fn = impl.print_info,
                    .eval_map = impl.eval_map,
                    .eval_zip = impl.eval_zip,
                    .eval_reduce = impl.reduce,
                },
            };
        }
        pub fn realize(self: *Self, storage: ?*TensorStorage(dtype, size), allocator: Allocator) !void {
            // TODO: Make this async to block thread until tensor is computed
            // Current impl of realize does not trace back up its compute graph to actually get values
            self.storage = storage orelse try TensorStorage(dtype, size).init(allocator);
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
        pub fn permute(comptime _: *const Self, comptime perm: [ndims]u8) PermutedTensor(Self, perm) {
            return PermutedTensor(Self, perm).init();
        }
        pub fn map(self: *const Self, op: ops.MapOp) Self {
            var out = init();
            out.graph_tensor.last_op = .{ .MapOp = .{
                .op = op,
                .a = &self.graph_tensor,
            } };
            return out;
        }
        fn eval_map(self: *const Self, op_call: ops.OpCall) void {
            switch (op_call) {
                .MapOp => |map_op_call| {
                    // TODO: If the self Tensor is realized, execute the operation on its data using the args provided by the op_call
                    // Add any extra args as necessary (e.g. output location)
                    _ = map_op_call;
                },
                else => @panic("Invalid map op call"),
            }
            _ = self;
        }
        pub fn zip(self: *const Self, op: ops.ZipOp, other: anytype) BroadcastedTensor(Self, @TypeOf(other)) {
            var out = BroadcastedTensor(Self, @TypeOf(other)).init();
            out.graph_tensor.last_op = .{ .ZipOp = .{ .op = op, .a = &self.graph_tensor, .b = &other.graph_tensor } };
            return out;
        }
        fn eval_zip(self: *const Self, op_call: ops.OpCall) void {
            switch (op_call) {
                .ZipOp => |zip_op_call| {
                    // TODO: If the self Tensor is realized, execute the operation on its data using the args provided by the op_call
                    // Add any extra args as necessary (e.g. output location)
                    _ = zip_op_call;
                },
                else => @panic("Invalid zip op call"),
            }
            _ = self;
            // var out = BroadcastedTensor(Self, @TypeOf(other)).init();
            // out.graph_tensor.last_op = .{ .ZipOp = .{ .op = op, .a = &self.graph_tensor, .b = &other.graph_tensor } };
            // return out.graph_tensor;
        }
        pub fn reduce(self: *const Self, op: ops.ReduceOp, comptime reduce_dim: usize) ReducedTensor(Self, reduce_dim) {
            var out = ReducedTensor(Self, reduce_dim).init();
            out.graph_tensor.last_op = .{ .ReduceOp = .{
                .op = op,
                .a = &self.graph_tensor,
                .reduce_dim = reduce_dim,
            } };
            return out;
        }
        fn eval_reduce(self: *const Self, op_call: ops.OpCall) void {
            _ = self;
            switch (op_call) {
                .ReduceOp => |reduce_op_call| {
                    // TODO: If the self Tensor is realized, execute the operation on its data using the args provided by the op_call
                    // Add any extra args as necessary (e.g. output location)
                    _ = reduce_op_call;
                },
                else => @panic("Invalid reduce op call"),
            }
        }
    };
}

pub fn DefaultStridedTensor(comptime dtype: type, comptime ndims: u8, comptime shape: [ndims]usize) type {
    return Tensor(dtype, ndims, shape, utils.defaultStrides(ndims, shape));
}

pub fn ReducedTensor(comptime tensor_t: type, comptime reduce_dim: usize) type {
    const dtype = @field(tensor_t, "dtype");
    const ndims = @field(tensor_t, "ndims");
    const shape = utils.reducedShape(ndims, @field(tensor_t, "shape"), reduce_dim);
    const strides = utils.defaultStrides(ndims, shape);
    return Tensor(dtype, ndims, shape, strides);
}

pub fn BroadcastedTensor(comptime tensor1_t: type, comptime tensor2_t: type) type {
    const shape = utils.shapeBroadcast(tensor1_t, tensor2_t);
    return Tensor(@field(tensor1_t, "dtype"), shape.len, shape, utils.defaultStrides(shape.len, shape));
}

pub fn PermutedTensor(comptime tensor_t: type, comptime perm: [@field(tensor_t, "ndims")]u8) type {
    const dtype = @field(tensor_t, "dtype");
    const ndims = @field(tensor_t, "ndims");
    const shape = @field(tensor_t, "shape");
    const strides = @field(tensor_t, "strides");
    const permute_shape = utils.permuteArray(ndims, shape, perm);
    const permute_strides = utils.permuteArray(ndims, strides, perm);
    return Tensor(dtype, ndims, permute_shape, permute_strides);
}
