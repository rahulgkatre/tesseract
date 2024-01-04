const std = @import("std");
const Allocator = std.mem.Allocator;

pub fn LazyBuffer(comptime dtype: type) type {
    _ = dtype;
    return struct {
        const Self = @This();
        deinitFn: *const fn (self: *Self) void,
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
        lazy_buffer: LazyBuffer(dtype),
        pub fn init(size: usize, allocator: Allocator) !*Self {
            const Impl = struct {
                pub fn deinit(ptr: *LazyBuffer) void {
                    const self = @fieldParentPtr(Self, "graph_buffer", ptr);
                    self.deinit();
                }
            };
            const data = try allocator.alloc(dtype, size);
            const zigBuffer = try allocator.create(Self);
            zigBuffer.* = .{
                .data = data,
                .allocator = allocator,
                .graph_buffer = .{
                    .deinitFn = Impl.deinit,
                },
            };
            return zigBuffer;
        }
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
        }
    };
}
