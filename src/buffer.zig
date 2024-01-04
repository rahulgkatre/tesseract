const std = @import("std");
const Allocator = std.mem.Allocator;
const BackendTypes = @import("backend.zig").BackendTypes;

// TODO: Make buffers part of an AST in order to fuse ops down the road
pub fn Buffer(comptime dtype: type) type {
    return union(BackendTypes) {
        const Self = @This();
        Zig: ZigBuffer(dtype),

        pub fn deinit(self: *Self) void {
            switch (self.*) {
                inline else => |*b| b.deinit(),
            }
        }
    };
}

pub fn ZigBuffer(comptime dtype: type) type {
    return struct {
        const Self = @This();
        data: []dtype,
        allocator: *const Allocator,
        pub fn init(size: usize, allocator: *const Allocator) !*Buffer(dtype) {
            const data = try allocator.alloc(dtype, size);
            const lazyBuffer = try allocator.create(Buffer(dtype));
            lazyBuffer.* = .{
                .Zig = .{
                    .data = data,
                    .allocator = allocator,
                },
            };
            return lazyBuffer;
        }
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
        }
    };
}
