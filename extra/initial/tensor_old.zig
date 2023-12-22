const std = @import("std");
const utils = @import("./tensor_utils.zig");
const Buffer = @import("buffer.zig").Buffer;

pub fn Tensor(
    comptime T: type,
) type {
    return struct {
        const Self = @This();
        const size = utils.shape2size(shape);
        const shape: []const usize;
        allocator: std.mem.Allocator,
        buffer: *Buffer(T, size),

        fn init(buffer: *Buffer(T, size), allocator: std.mem.Allocator) !*Self {
            const tensor = try allocator.create(Self);
            errdefer allocator.free(tensor);
            tensor.* = .{
                .allocator = allocator,
                .buffer = try Buffer(T, size).from_existing(buffer, allocator),
            };
            return tensor;
        }

        // TODO: Add function and grad stuff here
        pub fn empty(allocator: std.mem.Allocator) !*Self {
            const buffer = try Buffer(T, size).init(shape, allocator);
            errdefer buffer.deinit();
            return try init(buffer, allocator);
        }

        pub fn deinit(self: *Self) void {
            self.buffer.deinit();
            self.allocator.destroy(self);
        }

        pub fn full(fill_value: T, allocator: std.mem.Allocator) !*Self {
            const tensor = try empty(allocator);
            tensor.buffer.fill(fill_value);
            return tensor;
        }

        pub fn zeros(allocator: std.mem.Allocator) !*Self {
            return try full(0, allocator);
        }

        pub fn ones(allocator: std.mem.Allocator) !*Self {
            return try full(1, allocator);
        }

        pub fn rand(allocator: std.mem.Allocator) !*Self {
            var prng = std.rand.DefaultPrng.init(0);
            const tensor = try empty(allocator);
            tensor.buffer.fill_rand(prng.random());
            return tensor;
        }

        pub fn permute(self: *Self, comptime perm: []const usize) *Tensor(T, permute_shape(shape, perm)) {
            const new_shape = comptime permute_shape(shape, perm);
            const tensor: Tensor(T, new_shape) = try Tensor(T, new_shape).init(self.buffer, self.allocator);
            return tensor;
        }
    };
}

fn permute_shape(shape: []const usize, perm: []const usize) []const usize {
    var no_permute = true;
    for (perm, 0..shape.len) |perm_dim, dim| {
        if (perm_dim != dim) {
            no_permute = false;
            break;
        }
    }
    if (no_permute) {
        return shape;
    }

    var tmp: [utils.MAX_NDIMS]usize = undefined;
    @memset(&tmp, 0);
    for (perm, 0..shape.len) |perm_dim, dim| {
        tmp[dim] = shape[perm_dim];
    }
    return tmp[0..shape.len];
}
