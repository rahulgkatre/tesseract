const dtypes = @import("dtypes.zig");
const record = @import("record.zig");
const Graph = @import("Graph.zig");
const tensor = @import("tensor.zig");
const anytensor = @This();

/// Strips away comptime generic information to make it easier to work with pointers to tensors
/// The fields are the same in order to be able to @ptrCast() a generic tensor to anytensor
dtype: dtypes.DType,
ndims: u8,
shape: []const u64,
strides: []const u64,
offset: u64,
record: record.Record,

pub fn TensorType(comptime self: anytensor) type {
    return tensor.Tensor(self.dtype, self.ndims, self.shape);
}

pub fn infer(comptime self: anytensor) TensorType(self) {
    return @as(*TensorType(self), @ptrCast(&self)).*;
}

pub fn viz(self: *const anytensor, writer: anytype) !void {
    switch (self.record) {
        .TernaryOp => |ternary_op| {
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
                .out_uid = self.ordinal(),
                .out_dtype = @tagName(self.dtype),
                .out_shape = self.shape,
                .a_uid = ternary_op.a.tensor.ordinal(),
                .a_dtype = @tagName(ternary_op.a.tensor.dtype),
                .a_shape = ternary_op.a.tensor.shape,
                .b_uid = ternary_op.b.tensor.ordinal(),
                .b_dtype = @tagName(ternary_op.b.tensor.dtype),
                .b_shape = ternary_op.b.tensor.shape,
                .c_uid = ternary_op.c.tensor.ordinal(),
                .c_dtype = @tagName(ternary_op.c.tensor.dtype),
                .c_shape = ternary_op.c.tensor.shape,
            });
        },
        .ZipOp => |binary_op| {
            try writer.print(
                \\    {[op_tag]s}{[out_uid]d}[label="{[op_tag]s}"];
                \\    T{[a_uid]d}->{[op_tag]s}{[out_uid]d}[label="{[a_dtype]s}{[a_shape]any}"];
                \\    T{[b_uid]d}->{[op_tag]s}{[out_uid]d}[label="{[b_dtype]s}{[b_shape]any}"];
                \\    T{[out_uid]d}[label="T{[out_uid]d}"shape=box];
                \\    {[op_tag]s}{[out_uid]d}->T{[out_uid]d}[label="{[out_dtype]s}{[out_shape]any}"];
                \\
            , .{
                .op_tag = @tagName(binary_op.op),
                .out_uid = self.ordinal(),
                .out_dtype = @tagName(self.dtype),
                .out_shape = self.shape,
                .a_uid = binary_op.a.tensor.ordinal(),
                .a_dtype = @tagName(binary_op.a.tensor.dtype),
                .a_shape = binary_op.a.tensor.shape,
                .b_uid = binary_op.b.tensor.ordinal(),
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
                .out_uid = self.ordinal(),
                .out_dtype = @tagName(self.dtype),
                .out_shape = self.shape,
            });
        },
        .TypeOp => |unary_op| {
            try writer.print(
                \\    {[op_tag]s}{[out_uid]d}[label="{[op_tag]s}{[out_extra]any}"];
                \\    T{[a_uid]d}->{[op_tag]s}{[out_uid]d}[label="{[a_dtype]s}{[a_shape]any}"];
                \\    T{[out_uid]d}[label="T{[out_uid]d}"shape=box];
                \\    {[op_tag]s}{[out_uid]d}->T{[out_uid]d}[label="{[out_dtype]s}{[out_shape]any}"];
                \\
            , .{
                .op_tag = @tagName(unary_op.op),
                .out_uid = self.ordinal(),
                .out_dtype = @tagName(self.dtype),
                .out_shape = self.shape,
                .out_extra = .{
                    self.shape,
                    self.strides,
                    self.offset,
                },
                .a_uid = unary_op.a.tensor.ordinal(),
                .a_dtype = @tagName(unary_op.a.tensor.dtype),
                .a_shape = unary_op.a.tensor.shape,
            });
        },
        inline else => |unary_op| {
            try writer.print(
                \\    {[op_tag]s}{[out_uid]d}[label="{[op_tag]s}"];
                \\    T{[a_uid]d}->{[op_tag]s}{[out_uid]d}[label="{[a_dtype]s}{[a_shape]any}"];
                \\    T{[out_uid]d}[label="T{[out_uid]d}"shape=box];
                \\    {[op_tag]s}{[out_uid]d}->T{[out_uid]d}[label="{[out_dtype]s}{[out_shape]any}"];
                \\
            , .{
                .op_tag = @tagName(unary_op.op),
                .out_uid = self.ordinal(),
                .out_dtype = @tagName(self.dtype),
                .out_shape = self.shape,
                .a_uid = unary_op.a.tensor.ordinal(),
                .a_dtype = @tagName(unary_op.a.tensor.dtype),
                .a_shape = unary_op.a.tensor.shape,
            });
        },
    }
}

pub fn trace(self: *const anytensor) void {
    switch (self.record) {
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
    const key = @intFromPtr(self);
    if (!Graph.tensors.contains(key)) {
        Graph.ordinals.putNoClobber(key, Graph.ordinals.count()) catch unreachable;
        Graph.tensors.putNoClobber(key, self.*) catch unreachable;
    }
}

pub const JsonFormat = struct {
    uid: usize,
    dtype: dtypes.DType,
    ndims: u8,
    shape: []const u64,
    strides: []const u64,
    offset: u64,
};

pub fn toJsonFormat(self: *const anytensor) JsonFormat {
    return .{
        .uid = @intFromPtr(self),
        .dtype = self.dtype,
        .ndims = self.ndims,
        .shape = self.shape,
        .strides = self.strides,
        .offset = self.offset,
    };
}

pub fn ordinal(self: *const anytensor) u64 {
    // const std = @import("std");
    // std.debug.print("{}\n", .{@intFromPtr(self)});

    return Graph.ordinals.get(@intFromPtr(self)).?;
}
