const std = @import("std");
const Allocator = std.mem.Allocator;
const BackendTypes = @import("backend.zig").BackendTypes;

pub fn Storage(comptime dtype: type) type {
    return union(BackendTypes) {
        const Self = @This();
        Zig: ZigStorage(dtype),

        pub fn deinit(self: *Self) void {
            switch (self.*) {
                inline else => |*b| b.deinit(),
            }
        }
    };
}

pub fn ZigStorage(comptime dtype: type) type {
    return struct {
        const Self = @This();
        data: []dtype,
        allocator: *const Allocator,
        pub fn init(size: usize, allocator: *const Allocator) !*Storage(dtype) {
            const data = try allocator.alloc(dtype, size);
            const storage = try allocator.create(Storage(dtype));
            storage.* = .{
                .Zig = .{
                    .data = data,
                    .allocator = allocator,
                },
            };
            return storage;
        }
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
        }
    };
}
