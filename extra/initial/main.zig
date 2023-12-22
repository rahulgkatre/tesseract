const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

pub fn main() !void {
    // // Prints to stderr (it's a shortcut based on `std.io.getStdErr()`)

    // std.debug.print("All your {s} are belong to us.\n", .{"codebase"});

    // // stdout is for the actual output of your application, for example if you
    // // are implementing gzip, then only the compressed bytes should be sent to
    // // stdout, not any debugging messages.
    // const stdout_file = std.io.getStdOut().writer();
    // var bw = std.io.bufferedWriter(stdout_file);
    // const stdout = bw.writer();

    // try stdout.print("Run `zig build test` to run the tests.\n", .{});

    // try bw.flush(); // don't forget to flush!
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    // const shape: []const usize = &[_]usize;
    var tensor = try Tensor(f32, &.{ 2, 3, 4 }).rand(allocator);
    const tensor2 = try tensor.permute(&.{ 0, 1, 2 });
    _ = tensor2;
    // _ = tensor;
    std.debug.print("{any}\n", .{tensor});
    for (0..2) |i| {
        for (0..2) |j| {
            for (0..2) |k| {
                const pos = try tensor.buffer.idx2pos(&.{ i, j, k });
                std.debug.print("{any}: {any} = {any} \n", .{ .{ i, j, k }, pos, tensor.buffer.data[pos] });
            }
        }
    }
}
