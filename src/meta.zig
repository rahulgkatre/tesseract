const ops = @import("ops.zig");
const std = @import("std");
const utils = @import("utils.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;
const tensor = @import("tensor.zig");
const graph = @import("graph.zig");

/// Metadata for tensors, shared between the shape-typed Tensor and AnyTensor
pub const Metadata = struct {
    instr: ops.Instruction,
    forward: *const fn (*const AnyTensor, *graph.Graph) anyerror!void,
    backward: ?*const fn (*const AnyTensor, *graph.Graph) anyerror!void = null,
    constant: bool,
    label: ?[]const u8,

    pub fn init(comptime instr: ops.Instruction, constant: bool, label: ?[]const u8) Metadata {
        return .{
            .instr = instr,
            .forward = struct {
                fn forwardImpl(out: *const AnyTensor, g: *graph.Graph) !void {
                    std.debug.print("tracing {d}\n", .{@intFromPtr(out)});
                    switch (instr) {
                        inline else => |instr_| {
                            for (instr_.src) |src| {
                                try g.trace(src);
                            }
                        },
                    }
                    std.debug.print("finished tracing {d}\n", .{@intFromPtr(out)});
                    try g.compute(out, instr);
                }
            }.forwardImpl,
            .constant = constant,
            .label = label,
        };
    }
};
