const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("src/tensor.zig").Tensor;
const Range = @import("src/tensor.zig").Range;

const Backend = @import("src/backend.zig").Backend;

const TestBackend = &Backend{ .Zig = .{} };

pub fn main() !void {
    // To take advantage of comptime features, all tensor code should be in comptime
    const out = comptime blk: {
        const x1 = Range(TestBackend, i32, 0, 32).view(.{ 1, 4, 8 });
        const x2 = Range(TestBackend, i32, 32, 64).view(.{ 8, 4, 1 });
        _ = x2;
        const x3 = x1.neg().sum(null);
        break :blk x3;
    };

    TestBackend.init(.{});
    defer TestBackend.deinit();

    out.graph();
    std.debug.print("\n{any}\n", .{out.eval().storage.?.Zig});
}
