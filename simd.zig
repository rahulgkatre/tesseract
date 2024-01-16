const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("src/tensor.zig").Tensor;
const Range = @import("src/tensor.zig").Range;

const Backend = @import("src/backend.zig").Backend;

const TestBackend = &Backend{ .Zig = .{} };

pub fn main() !void {
    // To take advantage of comptime features, all tensor code should be in comptime
    const out = comptime blk: {
        const x1 = Range(TestBackend, i32, 0, 64).view(.{ 8, 1, 4, 1, 2 }).view(.{ 8, 1, 4, 1, 2 });
        const x2 = Range(TestBackend, i32, 64, 128).view(.{ 8, 2, 1, 4, 1 }).view(.{ 8, 4, 1, 2, 1 });
        const x3 = x1.add(x2);
        break :blk x3;
    };

    TestBackend.init(.{});
    defer TestBackend.deinit();

    out.graph();

    const out_eval = out.eval();
    std.debug.print("\n{any}\n", .{out_eval.storage.?.Zig});
}
