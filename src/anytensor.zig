const dtypes = @import("dtypes.zig");
const record = @import("record.zig");
const Graph = @import("Graph.zig");
const tensor = @import("tensor.zig");

/// Strips away comptime generic information to make it easier to work with pointers to tensors
/// The fields are the same in order to be able to @ptrCast() a generic tensor to anytensor
pub const anytensor = extern struct {
    dtype: dtypes.DType,
    ndims: u8,
    shape: [*]const u64,
    strides: [*]const u64,
    offset: u64,
    record: *const record.Record,

    pub fn TensorType(comptime self: anytensor) type {
        return tensor.Tensor(self.dtype, self.ndims, self.shape[0..self.ndims][0..].*);
    }

    pub fn infer(comptime self: anytensor) TensorType(self) {
        return @as(*const TensorType(self), @ptrCast(&self)).*;
    }

    pub fn viz(self: *const anytensor, writer: anytype) !void {
        switch (self.record.*) {
            .TernaryOp => |rec| {
                try writer.print(
                    \\    {[op_tag]s}{[out_uid]d}[label="{[op_tag]s}"];
                    \\    T{[a_uid]d}->{[op_tag]s}{[out_uid]d}[label="{[a_dtype]s}{[a_shape]any}"];
                    \\    T{[b_uid]d}->{[op_tag]s}{[out_uid]d}[label="{[b_dtype]s}{[b_shape]any}"];
                    \\    T{[c_uid]d}->{[op_tag]s}{[out_uid]d}[label="{[c_dtype]s}{[c_shape]any}"];
                    \\    T{[out_uid]d}[label="T{[out_uid]d}"shape=box];
                    \\    {[op_tag]s}{[out_uid]d}->T{[out_uid]d}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\    
                , .{
                    .op_tag = @tagName(rec.op),
                    .out_uid = self.ordinal(),
                    .out_dtype = @tagName(self.dtype),
                    .out_shape = self.shape[0..self.ndims],
                    .a_uid = rec.a.tensor.ordinal(),
                    .a_dtype = @tagName(rec.a.tensor.dtype),
                    .a_shape = rec.a.tensor.shape[0..rec.a.tensor.ndims],
                    .b_uid = rec.b.tensor.ordinal(),
                    .b_dtype = @tagName(rec.b.tensor.dtype),
                    .b_shape = rec.b.tensor.shape[0..rec.b.tensor.ndims],
                    .c_uid = rec.c.tensor.ordinal(),
                    .c_dtype = @tagName(rec.c.tensor.dtype),
                    .c_shape = rec.c.tensor.shape[0..rec.c.tensor.ndims],
                });
            },
            .BinaryOp => |rec| {
                try writer.print(
                    \\    {[op_tag]s}{[out_uid]d}[label="{[op_tag]s}"];
                    \\    T{[a_uid]d}->{[op_tag]s}{[out_uid]d}[label="{[a_dtype]s}{[a_shape]any}"];
                    \\    T{[b_uid]d}->{[op_tag]s}{[out_uid]d}[label="{[b_dtype]s}{[b_shape]any}"];
                    \\    T{[out_uid]d}[label="T{[out_uid]d}"shape=box];
                    \\    {[op_tag]s}{[out_uid]d}->T{[out_uid]d}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\
                , .{
                    .op_tag = @tagName(rec.op),
                    .out_uid = self.ordinal(),
                    .out_dtype = @tagName(self.dtype),
                    .out_shape = self.shape[0..self.ndims],
                    .a_uid = rec.a.tensor.ordinal(),
                    .a_dtype = @tagName(rec.a.tensor.dtype),
                    .a_shape = rec.a.tensor.shape[0..rec.a.tensor.ndims],
                    .b_uid = rec.b.tensor.ordinal(),
                    .b_dtype = @tagName(rec.b.tensor.dtype),
                    .b_shape = rec.b.tensor.shape[0..rec.b.tensor.ndims],
                });
            },
            .InitOp => |rec| {
                try writer.print(
                    \\    {[op_tag]s}{[out_uid]d}[label="{[op_tag]s}"];
                    \\    T{[out_uid]d}[label="T{[out_uid]d}"shape=box];
                    \\    {[op_tag]s}{[out_uid]d}->T{[out_uid]d}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\
                , .{
                    .op_tag = @tagName(rec.op),
                    .out_uid = self.ordinal(),
                    .out_dtype = @tagName(self.dtype),
                    .out_shape = self.shape[0..self.ndims],
                });
            },
            .DataOp => |rec| {
                try writer.print(
                    \\    {[op_tag]s}{[out_uid]d}[label="{[op_tag]s}{[out_extra]any}"];
                    \\    T{[a_uid]d}->{[op_tag]s}{[out_uid]d}[label="{[a_dtype]s}{[a_shape]any}"];
                    \\    T{[out_uid]d}[label="T{[out_uid]d}"shape=box];
                    \\    {[op_tag]s}{[out_uid]d}->T{[out_uid]d}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\
                , .{
                    .op_tag = @tagName(rec.op),
                    .out_uid = self.ordinal(),
                    .out_dtype = @tagName(self.dtype),
                    .out_shape = self.shape[0..self.ndims],
                    .out_extra = .{
                        self.shape[0..self.ndims],
                        self.strides[0..self.ndims],
                        self.offset,
                    },
                    .a_uid = rec.a.tensor.ordinal(),
                    .a_dtype = @tagName(rec.a.tensor.dtype),
                    .a_shape = rec.a.tensor.shape[0..rec.a.tensor.ndims],
                });
            },
            inline else => |rec| {
                try writer.print(
                    \\    {[op_tag]s}{[out_uid]d}[label="{[op_tag]s}"];
                    \\    T{[a_uid]d}->{[op_tag]s}{[out_uid]d}[label="{[a_dtype]s}{[a_shape]any}"];
                    \\    T{[out_uid]d}[label="T{[out_uid]d}"shape=box];
                    \\    {[op_tag]s}{[out_uid]d}->T{[out_uid]d}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\
                , .{
                    .op_tag = @tagName(rec.op),
                    .out_uid = self.ordinal(),
                    .out_dtype = @tagName(self.dtype),
                    .out_shape = self.shape[0..self.ndims],
                    .a_uid = rec.a.tensor.ordinal(),
                    .a_dtype = @tagName(rec.a.tensor.dtype),
                    .a_shape = rec.a.tensor.shape[0..rec.a.tensor.ndims],
                });
            },
        }
    }

    pub fn trace(self: *const anytensor) void {
        switch (self.record.*) {
            .TernaryOp => |ternary_op| {
                ternary_op.a.tensor.trace();
                ternary_op.b.tensor.trace();
                ternary_op.c.tensor.trace();
            },
            .BinaryOp => |binary_op| {
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
            .shape = self.shape[0..self.ndims],
            .strides = self.strides[0..self.ndims],
            .offset = self.offset,
        };
    }

    pub fn ordinal(self: *const anytensor) u64 {
        return Graph.ordinals.get(@intFromPtr(self)).?;
    }
};
