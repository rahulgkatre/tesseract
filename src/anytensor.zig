const dtypes = @import("dtypes.zig");
const record = @import("record.zig");
const tensor = @import("tensor.zig");

/// Strips away generic information to make it easier to work with pointers to tensors
/// with different shapes, dtypes, etc.
/// By making anytensor and generic tensor extern structs, they are guaranteed to have
/// the same layout.
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
        const Viz = struct {
            fn vizOp(t: *const anytensor, w: anytype) !void {
                switch (t.record.*) {
                    .DataOp => |rec| {
                        try switch (rec.op) {
                            .AsType => w.print(
                                \\    {[op]s}_{[out]x}[label="{[op]s}({[data]s})"];
                                \\
                            , .{
                                .op = @tagName(rec.op),
                                .out = @intFromPtr(t),
                                .data = @tagName(t.dtype),
                            }),
                            .AsStrided => w.print(
                                \\    {[op]s}_{[out]x}[label="{[op]s}{[data]any}"];
                                \\
                            , .{
                                .op = @tagName(rec.op),
                                .out = @intFromPtr(t),
                                .data = .{
                                    t.shape[0..t.ndims],
                                    t.strides[0..t.ndims],
                                    t.offset,
                                },
                            }),
                        };
                    },
                    inline else => |rec| {
                        try w.print(
                            \\    {[op]s}_{[out]x}[label="{[op]s}"];
                            \\
                        , .{
                            .op = @tagName(rec.op),
                            .out = @intFromPtr(t),
                        });
                    },
                }
            }
        };

        try Viz.vizOp(self, writer);

        switch (self.record.*) {
            .TernaryOp => |rec| {
                try writer.print(
                    \\    T_{[a]x}->{[op]s}_{[out]x}[label="{[a_dtype]s}{[a_shape]any}"];
                    \\    T_{[b]x}->{[op]s}_{[out]x}[label="{[b_dtype]s}{[b_shape]any}"];
                    \\    T_{[c]x}->{[op]s}_{[out]x}[label="{[c_dtype]s}{[c_shape]any}"];
                    \\    T_{[out]x}[label="T_{[out]x}"shape=box];
                    \\    {[op]s}_{[out]x}->T_{[out]x}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\    
                , .{
                    .op = @tagName(rec.op),
                    .out = @intFromPtr(self),
                    .out_dtype = @tagName(self.dtype),
                    .out_shape = self.shape[0..self.ndims],
                    .a = @intFromPtr(rec.a.tensor),
                    .a_dtype = @tagName(rec.a.tensor.dtype),
                    .a_shape = rec.a.tensor.shape[0..rec.a.tensor.ndims],
                    .b = @intFromPtr(rec.b.tensor),
                    .b_dtype = @tagName(rec.b.tensor.dtype),
                    .b_shape = rec.b.tensor.shape[0..rec.b.tensor.ndims],
                    .c = @intFromPtr(rec.c.tensor),
                    .c_dtype = @tagName(rec.c.tensor.dtype),
                    .c_shape = rec.c.tensor.shape[0..rec.c.tensor.ndims],
                });
            },
            .BinaryOp => |rec| {
                try writer.print(
                    \\    T_{[a]x}->{[op]s}_{[out]x}[label="{[a_dtype]s}{[a_shape]any}"];
                    \\    T_{[b]x}->{[op]s}_{[out]x}[label="{[b_dtype]s}{[b_shape]any}"];
                    \\    T_{[out]x}[label="T_{[out]x}"shape=box];
                    \\    {[op]s}_{[out]x}->T_{[out]x}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\
                , .{
                    .op = @tagName(rec.op),
                    .out = @intFromPtr(self),
                    .out_dtype = @tagName(self.dtype),
                    .out_shape = self.shape[0..self.ndims],
                    .a = @intFromPtr(rec.a.tensor),
                    .a_dtype = @tagName(rec.a.tensor.dtype),
                    .a_shape = rec.a.tensor.shape[0..rec.a.tensor.ndims],
                    .b = @intFromPtr(rec.b.tensor),
                    .b_dtype = @tagName(rec.b.tensor.dtype),
                    .b_shape = rec.b.tensor.shape[0..rec.b.tensor.ndims],
                });
            },
            .InitOp => |rec| {
                try writer.print(
                    \\    T_{[out]x}[label="T_{[out]x}"shape=box];
                    \\    {[op]s}_{[out]x}->T_{[out]x}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\
                , .{
                    .op = @tagName(rec.op),
                    .out = @intFromPtr(self),
                    .out_dtype = @tagName(self.dtype),
                    .out_shape = self.shape[0..self.ndims],
                });
            },
            inline else => |rec| {
                try writer.print(
                    \\    T_{[a]x}->{[op]s}_{[out]x}[label="{[a_dtype]s}{[a_shape]any}"];
                    \\    T_{[out]x}[label="T_{[out]x}"shape=box];
                    \\    {[op]s}_{[out]x}->T_{[out]x}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\
                , .{
                    .op = @tagName(rec.op),
                    .out = @intFromPtr(self),
                    .out_dtype = @tagName(self.dtype),
                    .out_shape = self.shape[0..self.ndims],
                    .a = @intFromPtr(rec.a.tensor),
                    .a_dtype = @tagName(rec.a.tensor.dtype),
                    .a_shape = rec.a.tensor.shape[0..rec.a.tensor.ndims],
                });
            },
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
};
