const std = @import("std");
const ops = @import("ops.zig");
const comptimePrint = @import("std").fmt.comptimePrint;
const Graph = @import("Graph.zig");
const Program = @import("Program.zig");

// All C-like programming languages support the same arithmetic expressions (+ and *) and formatting
// So these are shared codegen functions for

/// Based on implementation of unravel_index from numpy
/// https://chromium.googlesource.com/external/github.com/numpy/numpy/+/maintenance/1.3.x/numpy/lib/index_tricks.py
/// This is actually inlineable as opposed to the modulo method
pub fn unravelCode(allocator: std.mem.Allocator, node: *const Graph.TensorNode) std.mem.Allocator.Error![]const u8 {
    const ndims = node.tensor.ndims;
    const strides = node.tensor.strides;

    var count = std.fmt.count("{d}", .{node.tensor.strides[ndims]});
    for (0..ndims) |d| {
        if (node.tensor.strides[ndims - 1 - d] != 0) {
            if (node.tensor.strides[ndims - 1 - d] == 1) {
                count += std.fmt.count("+" ++ loop_var_fmt, .{ndims - 1 - d});
            } else {
                count += std.fmt.count("+" ++ loop_var_fmt ++ "*{d}", .{ ndims - 1 - d, node.tensor.strides[ndims - 1 - d] });
            }
        }
    }
    const buf: []u8 = try allocator.alloc(u8, count);

    var printed = std.fmt.bufPrint(buf, "{d}", .{strides[ndims]}) catch unreachable;
    var offset = printed.len;
    for (0..ndims) |d| {
        const rev_d = ndims - 1 - d;
        if (strides[rev_d] != 0) {
            if (strides[rev_d] == 1) {
                printed = std.fmt.bufPrint(buf[offset..], "+" ++ loop_var_fmt, .{rev_d}) catch unreachable;
            } else {
                printed = std.fmt.bufPrint(buf[offset..], "+" ++ loop_var_fmt ++ "*{d}", .{ rev_d, node.tensor.strides[rev_d] }) catch unreachable;
            }
            offset += printed.len;
        }
    }

    return buf;
}

// Similar to above but with added logic for broadcasting the position between two tensors
pub fn broadcastedUnravelCode(
    allocator: std.mem.Allocator,
    node: *const Graph.TensorNode,
    bc_node: *const Graph.TensorNode,
) ![]const u8 {
    const strides = node.tensor.strides;
    const shape = node.tensor.shape;
    const ndims = strides.len - 1;

    const bc_strides = bc_node.tensor.strides;
    const bc_shape = bc_node.tensor.shape;
    const bc_ndims = bc_strides.len - 1;

    var count = std.fmt.count("{d}", .{strides[ndims]});
    for (0..bc_ndims) |d| {
        const dim = if (d >= ndims) 1 else shape[ndims - 1 - d];
        const stride = if (d >= ndims) 0 else strides[ndims - 1 - d];
        if (dim == bc_shape[bc_ndims - 1 - d]) {
            if (stride != 0) {
                if (stride == 1) {
                    count += std.fmt.count("+" ++ loop_var_fmt, .{bc_ndims - 1 - d});
                } else {
                    count += std.fmt.count("+" ++ loop_var_fmt ++ "*{d}", .{ bc_ndims - 1 - d, stride });
                }
            }
        }
    }

    const buf: []u8 = try allocator.alloc(u8, count);
    var printed = std.fmt.bufPrint(buf, "{d}", .{strides[ndims]}) catch unreachable;
    var offset = printed.len;
    for (0..bc_ndims) |d| {
        const dim = if (d >= ndims) 1 else shape[ndims - 1 - d];
        const stride = if (d >= ndims) 0 else strides[ndims - 1 - d];
        if (dim == bc_shape[bc_ndims - 1 - d]) {
            if (stride == 1) {
                printed = std.fmt.bufPrint(buf[offset..], "+" ++ loop_var_fmt, .{bc_ndims - 1 - d}) catch unreachable;
            } else {
                printed = std.fmt.bufPrint(buf[offset..], "+" ++ loop_var_fmt ++ "*{d}", .{ bc_ndims - 1 - d, stride }) catch unreachable;
            }
            offset += printed.len;
        }
    }

    return buf;
}

const loop_var_fmt = "d{d}";
pub fn loopVarCode(allocator: std.mem.Allocator, loop: *const Program.Loop) ![]const u8 {
    return try std.fmt.allocPrint(allocator, loop_var_fmt, .{loop.dim});
}
