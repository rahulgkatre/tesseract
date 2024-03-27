const std = @import("std");
const ops = @import("ops.zig");
const utils = @import("utils.zig");
const dtypes = @import("dtypes.zig");

var gpa: std.heap.GeneralPurposeAllocator(.{}) = undefined;
var arena: std.heap.ArenaAllocator = undefined;
var uids: std.AutoArrayHashMap(usize, usize) = undefined;
var tensors: std.AutoArrayHashMap(usize, TensorNode) = undefined;

pub fn init() void {
    gpa = .{};
    arena = std.heap.ArenaAllocator.init(gpa.allocator());
    uids = std.AutoArrayHashMap(usize, usize).init(arena.allocator());
    tensors = std.AutoArrayHashMap(usize, TensorNode).init(arena.allocator());
}

pub fn deinit() void {
    arena.deinit();
    _ = gpa.deinit();
}

/// Strips away comptime / generic information to make it easier to work with pointers to tensors
pub const TensorNode = struct {
    ptr: *const anyopaque,
    dtype: dtypes.DType,
    ndims: u8,
    shape: []const u64,
    strides: []const u64,
    offset: u64,
    // size: u64,
    // contiguous: bool,
    op_node: OpNode,

    pub fn viz(self: *const TensorNode, writer: anytype, visited: []bool) !void {
        if (visited[@intCast(self.uid())]) {
            return;
        }
        visited[@intCast(self.uid())] = true;
        switch (self.op_node) {
            .TernaryOp => |ternary_op| {
                try ternary_op.a.tensor.viz(writer, visited);
                try ternary_op.b.tensor.viz(writer, visited);
                try ternary_op.c.tensor.viz(writer, visited);
                try writer.print(
                    \\    {[op_tag]s}{[out_uid]d}[label="{[op_tag]s}"];
                    \\    T{[a_uid]d}->{[op_tag]s}{[out_uid]d}[label="{[a_dtype]s}{[a_shape]any}"];
                    \\    T{[b_uid]d}->{[op_tag]s}{[out_uid]d}[label="{[b_dtype]s}{[b_shape]any}"];
                    \\    T{[c_uid]d}->{[op_tag]s}{[out_uid]d}[label="{[c_dtype]s}{[c_shape]any}"];
                    \\    T{[out_uid]d}[label="T{[out_uid]d}"shape=box];
                    \\    {[op_tag]s}{[out_uid]d}->T{[out_uid]d}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\    
                , .{
                    .op_tag = @tagName(ternary_op.op),
                    .out_uid = self.uid(),
                    .out_dtype = @tagName(self.dtype),
                    .out_shape = self.shape,
                    .a_uid = ternary_op.a.tensor.uid(),
                    .a_dtype = @tagName(ternary_op.a.tensor.dtype),
                    .a_shape = ternary_op.a.tensor.shape,
                    .b_uid = ternary_op.b.tensor.uid(),
                    .b_dtype = @tagName(ternary_op.b.tensor.dtype),
                    .b_shape = ternary_op.b.tensor.shape,
                    .c_uid = ternary_op.c.tensor.uid(),
                    .c_dtype = @tagName(ternary_op.c.tensor.dtype),
                    .c_shape = ternary_op.c.tensor.shape,
                });
            },
            .ZipOp => |binary_op| {
                try binary_op.a.tensor.viz(writer, visited);
                try binary_op.b.tensor.viz(writer, visited);
                try writer.print(
                    \\    {[op_tag]s}{[out_uid]d}[label="{[op_tag]s}"];
                    \\    T{[a_uid]d}->{[op_tag]s}{[out_uid]d}[label="{[a_dtype]s}{[a_shape]any}"];
                    \\    T{[b_uid]d}->{[op_tag]s}{[out_uid]d}[label="{[b_dtype]s}{[b_shape]any}"];
                    \\    T{[out_uid]d}[label="T{[out_uid]d}"shape=box];
                    \\    {[op_tag]s}{[out_uid]d}->T{[out_uid]d}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\
                , .{
                    .op_tag = @tagName(binary_op.op),
                    .out_uid = self.uid(),
                    .out_dtype = @tagName(self.dtype),
                    .out_shape = self.shape,
                    .a_uid = binary_op.a.tensor.uid(),
                    .a_dtype = @tagName(binary_op.a.tensor.dtype),
                    .a_shape = binary_op.a.tensor.shape,
                    .b_uid = binary_op.b.tensor.uid(),
                    .b_dtype = @tagName(binary_op.b.tensor.dtype),
                    .b_shape = binary_op.b.tensor.shape,
                });
            },
            .InitOp => |init_op| {
                try writer.print(
                    \\    {[op_tag]s}{[out_uid]d}[label="{[op_tag]s}"];
                    \\    T{[out_uid]d}[label="T{[out_uid]d}"shape=box];
                    \\    {[op_tag]s}{[out_uid]d}->T{[out_uid]d}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\
                , .{
                    .op_tag = @tagName(init_op.op),
                    .out_uid = self.uid(),
                    .out_dtype = @tagName(self.dtype),
                    .out_shape = self.shape,
                });
            },
            .TypeOp => |unary_op| {
                try unary_op.a.tensor.viz(writer, visited);
                try writer.print(
                    \\    {[op_tag]s}{[out_uid]d}[label="{[op_tag]s}{[out_extra]any}"];
                    \\    T{[a_uid]d}->{[op_tag]s}{[out_uid]d}[label="{[a_dtype]s}{[a_shape]any}"];
                    \\    T{[out_uid]d}[label="T{[out_uid]d}"shape=box];
                    \\    {[op_tag]s}{[out_uid]d}->T{[out_uid]d}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\
                , .{
                    .op_tag = @tagName(unary_op.op),
                    .out_uid = self.uid(),
                    .out_dtype = @tagName(self.dtype),
                    .out_shape = self.shape,
                    .out_extra = .{
                        self.shape,
                        self.strides,
                        self.offset,
                    },
                    .a_uid = unary_op.a.tensor.uid(),
                    .a_dtype = @tagName(unary_op.a.tensor.dtype),
                    .a_shape = unary_op.a.tensor.shape,
                });
            },
            inline else => |unary_op| {
                try unary_op.a.tensor.viz(writer, visited);
                try writer.print(
                    \\    {[op_tag]s}{[out_uid]d}[label="{[op_tag]s}"];
                    \\    T{[a_uid]d}->{[op_tag]s}{[out_uid]d}[label="{[a_dtype]s}{[a_shape]any}"];
                    \\    T{[out_uid]d}[label="T{[out_uid]d}"shape=box];
                    \\    {[op_tag]s}{[out_uid]d}->T{[out_uid]d}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\
                , .{
                    .op_tag = @tagName(unary_op.op),
                    .out_uid = self.uid(),
                    .out_dtype = @tagName(self.dtype),
                    .out_shape = self.shape,
                    .a_uid = unary_op.a.tensor.uid(),
                    .a_dtype = @tagName(unary_op.a.tensor.dtype),
                    .a_shape = unary_op.a.tensor.shape,
                });
            },
        }
    }

    pub fn trace(self: *const TensorNode) void {
        switch (self.op_node) {
            .TernaryOp => |ternary_op| {
                ternary_op.a.tensor.trace();
                ternary_op.b.tensor.trace();
                ternary_op.c.tensor.trace();
            },
            .ZipOp => |binary_op| {
                binary_op.a.tensor.trace();
                binary_op.b.tensor.trace();
            },
            .InitOp => {},
            inline else => |unary_op| {
                unary_op.a.tensor.trace();
            },
        }
        const key = @intFromPtr(self.ptr);
        if (!tensors.contains(key)) {
            uids.putNoClobber(key, uids.count()) catch unreachable;
            tensors.putNoClobber(key, self.*) catch unreachable;
        }
    }

    pub fn jsonStringify(self: *const TensorNode, write_stream: anytype) !void {
        try write_stream.write(.{
            .ptr = @intFromPtr(self.ptr),
            .dtype = self.dtype,
            .ndims = self.ndims,
            .shape = self.shape,
            .strides = self.strides,
            .offset = self.offset,
            // .size = self.size,
            // .contiguous = self.contiguous,
        });
    }

    pub fn uid(self: *const TensorNode) u64 {
        return uids.get(@intFromPtr(self.ptr)).?;
    }
};

pub const OpNode = union(ops.OpTypes) {
    const Input = struct {
        tensor: *const TensorNode,
        fused: bool = false,
    };

    const JsonFormat = union(ops.OpTypes) {
        MapOp: struct {
            op: ops.MapOp,
            a: usize,
            out: usize,
        },
        ZipOp: struct {
            op: ops.ZipOp,
            a: usize,
            b: usize,
            out: usize,
        },
        ReduceOp: struct {
            op: ops.ReduceOp,
            a: usize,
            dims: []const bool,
            out: usize,
        },
        TypeOp: struct {
            op: ops.TypeOp,
            a: usize,
            out: usize,
        },
        InitOp: struct {
            op: ops.InitOp,
            args: ops.InitOp.Args,
            out: usize,
        },
        TernaryOp: struct {
            op: ops.TernaryOp,
            a: usize,
            b: usize,
            c: usize,
            out: usize,
        },
    };

    MapOp: struct {
        op: ops.MapOp,
        a: Input,
    },
    ZipOp: struct {
        op: ops.ZipOp,
        a: Input,
        b: Input,
    },
    ReduceOp: struct {
        op: ops.ReduceOp,
        a: Input,
        dims: []const bool,
    },
    TypeOp: struct {
        op: ops.TypeOp,
        a: Input,
    },
    InitOp: struct {
        op: ops.InitOp,
        args: ops.InitOp.Args,
    },
    TernaryOp: struct {
        op: ops.TernaryOp,
        a: Input,
        b: Input,
        c: Input,
    },

    pub fn init(
        comptime tag: ops.OpTypes,
        op: @field(ops, @tagName(tag)),
        inputs: switch (tag) {
            .TernaryOp => [3]*const TensorNode,
            .ZipOp => [2]*const TensorNode,
            .MapOp, .TypeOp, .ReduceOp => [1]*const TensorNode,
            .InitOp => void,
        },
        args: switch (tag) {
            .ReduceOp => []const bool,
            .InitOp => ops.InitOp.Args,
            else => void,
        },
    ) OpNode {
        return @unionInit(OpNode, @tagName(tag), switch (tag) {
            .TernaryOp => .{
                .op = op,
                .a = .{ .tensor = inputs[0] },
                .b = .{ .tensor = inputs[1] },
                .c = .{ .tensor = inputs[2] },
            },
            .ZipOp => .{
                .op = op,
                .a = .{ .tensor = inputs[0] },
                .b = .{ .tensor = inputs[1] },
            },
            .ReduceOp => .{
                .op = op,
                .a = .{ .tensor = inputs[0] },
                .dims = args,
            },
            .MapOp, .TypeOp => .{
                .op = op,
                .a = .{ .tensor = inputs[0] },
            },
            .InitOp => .{
                .op = op,
                .args = args,
            },
        });
    }

    fn toJsonFormat(self: OpNode, out: TensorNode) JsonFormat {
        return switch (self) {
            .MapOp => |op_node| .{ .MapOp = .{
                .op = op_node.op,
                .a = @intFromPtr(op_node.a),
                .out = @intFromPtr(out.ptr),
            } },
            .ZipOp => |op_node| .{ .ZipOp = .{
                .op = op_node.op,
                .a = @intFromPtr(op_node.a.tensor),
                .b = @intFromPtr(op_node.b.tensor),
                .out = @intFromPtr(out.ptr),
            } },
            .ReduceOp => |op_node| .{ .ReduceOp = .{
                .op = op_node.op,
                .a = @intFromPtr(op_node.a.tensor),
                .dims = op_node.dims,
                .out = @intFromPtr(out.ptr),
            } },
            .TypeOp => |op_node| .{ .TypeOp = .{
                .op = op_node.op,
                .a = @intFromPtr(op_node.a.tensor),
                .out = @intFromPtr(out.ptr),
            } },
            .InitOp => |op_node| .{ .InitOp = .{
                .op = op_node.op,
                .args = op_node.args,
                .out = @intFromPtr(out.ptr),
            } },
            .TernaryOp => |op_node| .{ .TernaryOp = .{
                .op = op_node.op,
                .a = @intFromPtr(op_node.a.tensor),
                .b = @intFromPtr(op_node.b.tensor),
                .c = @intFromPtr(op_node.c.tensor),
                .out = @intFromPtr(out.ptr),
            } },
        };
    }
};

pub fn jsonStringify(_: @This(), write_stream: anytype) !void {
    const operations: []OpNode.JsonFormat = gpa.allocator().alloc(OpNode.JsonFormat, tensors.count()) catch unreachable;
    defer gpa.allocator().free(operations);
    for (tensors.values(), operations) |tensor, *operation| {
        operation.* = tensor.op_node.toJsonFormat(tensor);
    }
    try write_stream.write(.{
        .tensors = tensors.values(),
        .operations = operations,
    });
}

pub fn viz(writer: anytype) !void {
    const visited = arena.allocator().alloc(bool, tensors.count()) catch unreachable;
    defer arena.allocator().free(visited);
    try writer.print(
        \\digraph G {{
        \\    compound=true;
        \\
    , .{});
    // TODO: Support for multiple entrypoints in the case of a DAG with multiple sinks
    for (tensors.values()) |tensor| {
        try tensor.viz(writer, visited);
    }
    try writer.print(
        \\}}
        \\
    , .{});
}
