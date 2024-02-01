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

// Similar to above but with added logic for broadcasting the position
pub fn broadcastIdxToPos(comptime TensorType: type, comptime BroadcastTensorType: type, comptime loop_var_prefix: []const u8) []const u8 {
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

// Abstractions for lowering Graph.Node into a Loop which can be codegened for a specific language
// Loop structs will be stored in a list where order is exact order of code
// Loops are defined as a grammar, ever loop has a header and a body
pub const Loop = struct {
    header: LoopHeader,
    body: LoopBody,
};

// Loop header defines the bounds of the loop and the loop variable
// Loop variable will almost always be i_X where X is a number so just store the id
const LoopHeader = struct {
    lower_bound: usize,
    upper_bound: usize,
    loop_var_id: usize,
};

// Loop body can either be another loop (normal or accumulating) or an expression
// Expression can just reuse Graph.Link as it has access to all needed information
const LoopBody = union(enum) {
    InnerLoop: Loop,
    InnerAccLoop: AccLoop,
    expr: []Graph.Link,
};

// Accumulating loop is special because it needs an accumulator (var acc_X)
// To accumulate over multiple dimensions it can also be nested with inner loops
const AccLoop = struct {
    header: LoopHeader,
    body: LoopBody,
    acc_var_id: usize,
};
