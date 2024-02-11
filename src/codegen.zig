const ZigCodegen = @import("codegen/ZigCodegen.zig");
const ops = @import("ops.zig");
const comptimePrint = @import("std").fmt.comptimePrint;
const Graph = @import("Graph.zig");

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

// TODO: Need to get rid of comptime here
// All programming languages we would target support the same arithmetic expressions (+ and *) and formatting
// Based on implementation of unravel_index from numpy
// https://chromium.googlesource.com/external/github.com/numpy/numpy/+/maintenance/1.3.x/numpy/lib/index_tricks.py
// This is actually inlineable as opposed to the modulo method
pub fn unravelMultiIndex(comptime Tensor: type, comptime loop_var_prefix: []const u8) []const u8 {
    return comptime str: {
        var expr: []const u8 = comptimePrint("{d}", .{Tensor.strides[Tensor.ndims]});
        for (0..Tensor.ndims) |d| {
            const rev_d = Tensor.ndims - 1 - d;
            if (Tensor.strides[rev_d] != 0) {
                if (Tensor.strides[rev_d] == 1) {
                    expr = expr ++ comptimePrint("+{s}{d}", .{ loop_var_prefix, rev_d });
                } else {
                    expr = expr ++ comptimePrint("+{s}{d}*{d}", .{ loop_var_prefix, rev_d, Tensor.strides[rev_d] });
                }
            }
        }
        break :str expr;
    };
}

// Similar to above but with added logic for broadcasting the position
pub fn broadcastedUnravelMultiIndex(comptime TensorType: type, comptime BroadcastTensorType: type, comptime loop_var_prefix: []const u8) []const u8 {
    return comptime str: {
        var expr: []const u8 = comptimePrint("{d}", .{TensorType.strides[TensorType.ndims]});
        for (0..TensorType.ndims) |d| {
            if (TensorType.shape[TensorType.ndims - 1 - d] == BroadcastTensorType.shape[BroadcastTensorType.ndims - 1 - d]) {
                if (TensorType.strides[TensorType.ndims - 1 - d] == 1) {
                    expr = expr ++ comptimePrint("+{s}{d}", .{ loop_var_prefix, BroadcastTensorType.ndims - 1 - d });
                } else if (TensorType.strides[TensorType.ndims - 1 - d] != 0) {
                    expr = expr ++ comptimePrint("+{s}{d}*{d}", .{ loop_var_prefix, BroadcastTensorType.ndims - 1 - d, TensorType.strides[BroadcastTensorType.ndims - 1 - d] });
                }
            }
        }
        break :str expr;
    };
}
