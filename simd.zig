const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("src/tensor.zig").Tensor;
const Range = @import("src/tensor.zig").Range;

const Backend = @import("src/backend.zig").Backend;

const TestBackend = &Backend{ .Zig = .{} };

pub fn main() !void {
    // To take advantage of comptime features, all tensor code should be in comptime
    const out = comptime blk: {
        const x1 = Range(TestBackend, f32, 0, 64).view(.{ 4, 4, 4 });
        const x2 = Range(TestBackend, f32, 64, 128).view(.{ 4, 4, 4 });
        const x3 = x1.mul(x2).sum(null);
        break :blk x3;
    };

    TestBackend.init(.{});
    defer TestBackend.deinit();

    out.graph();
    std.debug.print("\n{any}\n", .{out.eval().storage.?.Zig});
}
