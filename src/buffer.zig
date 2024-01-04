const std = @import("std");
const Allocator = std.mem.Allocator;
const BackendTypes = @import("backend.zig").BackendTypes;

// TODO: Make this a union like Backend
pub fn LazyBuffer(comptime dtype: type) type {
    return union(BackendTypes) {
        const Self = @This();
        Zig: ZigLazyBuffer(dtype),

        pub fn deinit(self: *Self) void {
            return self.deinitFn(self);
        }
    };
}

pub fn ZigLazyBuffer(comptime dtype: type) type {
    return struct {
        const Self = @This();
        data: []dtype,
        allocator: Allocator,
        pub fn init(size: usize, allocator: Allocator) !*LazyBuffer(dtype) {
            const data = try allocator.alloc(dtype, size);
            const lazyBuffer = try allocator.create(LazyBuffer(dtype));
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
