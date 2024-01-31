const ZigCodegen = @import("codegen/ZigCodegen.zig");
const ops = @import("ops.zig");
const comptimePrint = @import("std").fmt.comptimePrint;

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
    pub fn alloc(gen: *const Codegen, writer: anytype, id: usize, comptime dtype: type, size: usize) void {
        switch (gen.*) {
            inline else => |*cg| cg.alloc(writer, id, dtype, size),
        }
    }
};

// All programming languages we would target support the same arithmetic expressions (+ and *) and formatting
pub fn idxToPos(comptime Tensor: type, comptime loop_var_prefix: []const u8) []const u8 {
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

pub fn broadcastIdxToPos(comptime Tensor: type, comptime BroadcastTensor: type, comptime loop_var_prefix: []const u8) []const u8 {
    return comptime str: {
        var expr: []const u8 = comptimePrint("{d}", .{Tensor.strides[Tensor.ndims]});
        for (0..Tensor.ndims) |d| {
            if (Tensor.shape[Tensor.ndims - 1 - d] == BroadcastTensor.shape[BroadcastTensor.ndims - 1 - d]) {
                if (Tensor.strides[Tensor.ndims - 1 - d] == 1) {
                    expr = expr ++ comptimePrint("+{s}{d}", .{ loop_var_prefix, BroadcastTensor.ndims - 1 - d });
                } else if (Tensor.strides[Tensor.ndims - 1 - d] != 0) {
                    expr = expr ++ comptimePrint("+{s}{d}*{d}", .{ loop_var_prefix, BroadcastTensor.ndims - 1 - d, Tensor.strides[BroadcastTensor.ndims - 1 - d] });
                }
            }
        }
        break :str expr;
    };
}
