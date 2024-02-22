const std = @import("std");
const ZigCodegen = @import("codegen/ZigCodegen.zig");
const ops = @import("ops.zig");
const comptimePrint = @import("std").fmt.comptimePrint;
const Graph = @import("Graph.zig");
const Program = @import("Program.zig");

pub const CodegenTypes = enum {
    Zig,
};

pub const Codegen = union(CodegenTypes) {
    Zig: ZigCodegen,
    const Self = @This();
    pub fn header(gen: *const Codegen, writer: anytype) void {
        switch (gen.*) {
            inline else => |*cg| cg.header(writer),
        }
    }
    pub fn footer(gen: *const Codegen, writer: anytype) void {
        switch (gen.*) {
            inline else => |*cg| cg.footer(writer),
        }
    }
    pub fn storage(gen: *const Codegen, writer: anytype, id: usize, comptime dtype: type, size: usize) void {
        switch (gen.*) {
            inline else => |*cg| cg.alloc(writer, id, dtype, size),
        }
    }
};

// All C-like programming languages support the same arithmetic expressions (+ and *) and formatting
// So these are shared codegen functions for

/// Based on implementation of unravel_index from numpy
/// https://chromium.googlesource.com/external/github.com/numpy/numpy/+/maintenance/1.3.x/numpy/lib/index_tricks.py
/// This is actually inlineable as opposed to the modulo method
pub fn unravelCodegenCount(node: *Graph.Vertex) u64 {
    const ndims = node.tensor.ndims;
    var count = std.fmt.count("{d}", .{node.tensor.strides[ndims]});
    for (0..ndims) |d| {
        if (node.tensor.strides[ndims - 1 - d] != 0) {
            if (node.tensor.strides[ndims - 1 - d] == 1) {
                count += std.fmt.count("+" ++ loop_var_fmt, .{ node.id, ndims - 1 - d });
            } else {
                count += std.fmt.count("+" ++ loop_var_fmt ++ "*{d}", .{ node.id, ndims - 1 - d, node.tensor.strides[ndims - 1 - d] });
            }
        }
    }
    return count;
}
pub fn unravelCodegen(allocator: std.mem.Allocator, node: *Graph.Vertex) ![]const u8 {
    const ndims = node.tensor.ndims;
    const strides = node.tensor.strides;
    const buf: []u8 = try allocator.alloc(u8, unravelCodegenCount(node));
    var printed = try std.fmt.bufPrint(buf, "{d}", .{strides[ndims]});
    var offset = printed.len;
    for (0..ndims) |d| {
        const rev_d = ndims - 1 - d;
        if (strides[rev_d] != 0) {
            if (strides[rev_d] == 1) {
                printed = try std.fmt.bufPrint(buf[offset..], "+" ++ loop_var_fmt, .{ node.id, rev_d });
            } else {
                printed = try std.fmt.bufPrint(buf[offset..], "+" ++ loop_var_fmt ++ "*{d}", .{ node.id, rev_d, node.tensor.strides[rev_d] });
            }
            offset += printed.len;
        }
    }

    return buf;
}

// Similar to above but with added logic for broadcasting the position between two tensors
pub fn broadcastedUnravelCodegenCount(
    node: *Graph.Vertex,
    bc_node: *Graph.Vertex,
) u64 {
    const strides = node.tensor.strides;
    const shape = node.tensor.shape;
    const ndims = node.tensor.ndims;

    const bc_shape = bc_node.tensor.shape;
    const bc_ndims = bc_node.tensor.ndims;
    var count = std.fmt.count("{d}", .{strides[ndims]});
    for (0..bc_ndims) |d| {
        const dim = if (d >= ndims) 1 else shape[ndims - 1 - d];
        const stride = if (d >= ndims) 0 else strides[ndims - 1 - d];
        if (dim == bc_shape[bc_ndims - 1 - d]) {
            if (stride != 0) {
                if (stride == 1) {
                    count += std.fmt.count("+" ++ loop_var_fmt, .{ bc_node.id, bc_ndims - 1 - d });
                } else {
                    count += std.fmt.count("+" ++ loop_var_fmt ++ "*{d}", .{ bc_node.id, bc_ndims - 1 - d, stride });
                }
            }
        }
    }
    return count;
}
pub fn broadcastedUnravelCodegen(
    allocator: std.mem.Allocator,
    node: *Graph.Vertex,
    bc_node: *Graph.Vertex,
) ![]const u8 {
    const strides = node.tensor.strides;
    const shape = node.tensor.shape;
    const ndims = strides.len - 1;

    const bc_strides = bc_node.tensor.strides;
    const bc_shape = bc_node.tensor.shape;
    const bc_ndims = bc_strides.len - 1;

    const buf: []u8 = try allocator.alloc(u8, broadcastedUnravelCodegenCount(node, bc_node));
    var printed = try std.fmt.bufPrint(buf, "{d}", .{strides[ndims]});
    var offset = printed.len;
    for (0..bc_ndims) |d| {
        const dim = if (d >= ndims) 1 else shape[ndims - 1 - d];
        const stride = if (d >= ndims) 0 else strides[ndims - 1 - d];
        if (dim == bc_shape[bc_ndims - 1 - d]) {
            if (stride == 1) {
                printed = try std.fmt.bufPrint(buf[offset..], "+" ++ loop_var_fmt, .{ bc_node.id, bc_ndims - 1 - d });
            } else {
                printed = try std.fmt.bufPrint(buf[offset..], "+" ++ loop_var_fmt ++ "*{d}", .{ bc_node.id, bc_ndims - 1 - d, stride });
            }
            offset += printed.len;
        }
    }

    return buf;
}

const loop_var_fmt = "i{d}_d{d}";
pub fn loopVarCodegenCount(loop: *Program.Loop) u64 {
    return std.fmt.count(loop_var_fmt, .{ loop.node.id, loop.dim });
}
pub fn loopVarCodegen(allocator: std.mem.Allocator, loop: *Program.Loop) []const u8 {
    return std.fmt.allocPrint(allocator, loop_var_fmt, .{ loop.node.id, loop.dim }) catch unreachable;
}
