const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("src/tensor.zig");
const Backend = @import("src/backend.zig").Backend;

const TestBackend = &Backend{ .Zig = .{} };

pub fn main() !void {
    const out = comptime blk: {
        const dim1 = 256;
        const dim2 = 2 * dim1;
        const x1 = Tensor.range(TestBackend, f32, 0, dim1 * dim2);
        const x2 = x1.view(.{ 1, dim1, dim2 });
        const x3 = x2.neg();
        const x4 = Tensor.range(TestBackend, f32, dim1 * dim2, 2 * dim1 * dim2);
        const x5 = x4.view(.{ dim1, 1, dim2 });
        const x6 = x5.neg();
        const x7 = x3.mul(x6);
        const x8 = x7.sum(null);
        break :blk x8;
    };

    TestBackend.init(.{});
    out.graph();
    const res = out.eval();
    std.debug.print("{any}\n", .{res.storage.?.Zig.data});
}
