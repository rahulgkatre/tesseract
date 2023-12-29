const std = @import("std");
const Allocator = std.mem.Allocator;
// Wrapper class for a reserved section of memory where the size is part of the type
// This prevents the memory from being used with tensors with an incompatible shape.
// TODO: Make this a normal struct (not a generic) so it can be stored in a GraphTensor.
pub fn TensorStorage(comptime dtype: type, comptime size: usize) type {
    return struct {
        const Self = @This();
        data: []dtype,
        // This allocator is a placeholder for a Device struct that provides an allocator object
        // so that the tensor storage can be allocated on device memory
        allocator: Allocator,
        pub fn init(allocator: Allocator) !*Self {
            const storage = try allocator.create(Self);
            storage.* = .{
                .data = try allocator.alloc(dtype, size),
                .allocator = allocator,
            };
            return storage;
        }
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
        }
    };
}
