const std = @import("std");
const utils = @import("./tensor_utils.zig");

pub fn Buffer(
    comptime T: type,
    comptime size: usize,
    // TODO: Add a device struct that provides a device memory allocator
) type {
    return struct {
        const Self = @This();
        data: []T,
        shape: []usize,
        strides: []usize,
        ndims: u8,
        owns_data: bool,
        allocator: std.mem.Allocator,

        // TODO: Use the device allocator to allocate data but use host allocator for other storage
        pub fn init(shape: []const usize, allocator: std.mem.Allocator) !*Buffer(T, size) {
            const buffer = try allocator.create(Self);
            buffer.* = .{
                .data = try allocator.alloc(T, size),
                .shape = try allocator.alloc(usize, utils.MAX_NDIMS),
                .strides = try allocator.alloc(usize, utils.MAX_NDIMS),
                .ndims = @truncate(shape.len),
                .allocator = allocator,
                .owns_data = true,
            };

            errdefer allocator.free(buffer.data);
            errdefer allocator.free(buffer.shape);
            errdefer allocator.free(buffer.strides);

            @memset(buffer.shape, 0);
            @memcpy(buffer.shape[0..buffer.ndims], shape);
            @memset(buffer.strides, 0);
            try utils.shape2strides(buffer.shape[0..buffer.ndims], buffer.strides[0..buffer.ndims]);
            return buffer;
        }

        pub fn from_existing(other: *Buffer(T, size), allocator: std.mem.Allocator) !*Buffer(T, size) {
            const buffer = try allocator.create(Self);
            buffer.* = .{
                .data = other.data,
                .shape = try allocator.alloc(usize, size),
                .strides = try allocator.alloc(usize, size),
                .ndims = @truncate(size),
                .allocator = allocator,
                .owns_data = false,
            };

            @memset(buffer.shape, 0);
            @memcpy(buffer.shape[0..buffer.ndims], other.shape[0..other.ndims]);
            @memset(buffer.strides, 0);
            try utils.shape2strides(buffer.shape[0..buffer.ndims], buffer.strides[0..buffer.ndims]);
            return buffer;
        }

        pub fn deinit(self: *Self) void {
            if (self.owns_data) {
                self.allocator.free(self.data);
            }
            self.allocator.free(self.shape);
            self.allocator.free(self.strides);
            self.allocator.destroy(self);
        }

        pub fn fill(self: *Self, value: T) void {
            @memset(self.data, value);
        }

        pub fn fill_rand(self: *Self, rng: std.rand.Random) void {
            for (self.data) |*elem| {
                elem.* = rng.floatNorm(T);
            }
        }

        pub fn idx2pos(self: *Self, idx: []const usize) !usize {
            if (idx.len != self.ndims) {
                return utils.TensorError.InequalDimensions;
            } else {
                var pos: usize = 0;
                for (0..idx.len) |dim| {
                    pos += idx[dim] * self.strides[dim];
                }
                return pos;
            }
        }

        fn pos2idx(self: *Self, pos: usize, out_index: []usize) !void {
            if (self.ndims != out_index.len) {
                return utils.TensorError.InequalDimensions;
            } else {
                var rem = pos;
                for (0..out_index.len) |dim| {
                    out_index[dim] = @divFloor(rem, self.strides[dim]);
                    rem = @rem(rem, self.strides[dim]);
                }
            }
        }
    };
}
