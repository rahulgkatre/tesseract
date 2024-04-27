const ops = @import("ops.zig");
const anytensor = @import("anytensor.zig").anytensor;

pub const Record = union(ops.OpTypes) {
    const Input = struct {
        tensor: *const anytensor,

        pub fn jsonStringify(input: @This(), write_stream: anytype) !void {
            try write_stream.write(@intFromPtr(input.tensor));
        }
    };

    pub const JsonFormat = union(ops.OpTypes) {
        UnaryOp: struct {
            op: ops.UnaryOp,
            a: Input,
            out: usize,
        },
        BinaryOp: struct {
            op: ops.BinaryOp,
            a: Input,
            b: Input,
            out: usize,
        },
        ReduceOp: struct {
            op: ops.ReduceOp,
            a: Input,
            dims: []const bool,
            out: usize,
        },
        DataOp: struct {
            op: ops.DataOp,
            a: Input,
            out: usize,
        },
        InitOp: struct {
            op: ops.InitOp,
            args: ops.InitOp.Args,
            out: usize,
        },
        TernaryOp: struct {
            op: ops.TernaryOp,
            a: Input,
            b: Input,
            c: Input,
            out: usize,
        },
    };

    UnaryOp: struct {
        op: ops.UnaryOp,
        a: Input,
    },
    BinaryOp: struct {
        op: ops.BinaryOp,
        a: Input,
        b: Input,
    },
    ReduceOp: struct {
        op: ops.ReduceOp,
        a: Input,
        dims: []const bool,
    },
    DataOp: struct {
        op: ops.DataOp,
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
            .TernaryOp => [3]*const anytensor,
            .BinaryOp => [2]*const anytensor,
            .UnaryOp, .DataOp, .ReduceOp => [1]*const anytensor,
            .InitOp => void,
        },
        args: switch (tag) {
            .ReduceOp => []const bool,
            .InitOp => ops.InitOp.Args,
            else => void,
        },
    ) Record {
        return @unionInit(Record, @tagName(tag), switch (tag) {
            .TernaryOp => .{
                .op = op,
                .a = .{ .tensor = inputs[0] },
                .b = .{ .tensor = inputs[1] },
                .c = .{ .tensor = inputs[2] },
            },
            .BinaryOp => .{
                .op = op,
                .a = .{ .tensor = inputs[0] },
                .b = .{ .tensor = inputs[1] },
            },
            .ReduceOp => .{
                .op = op,
                .a = .{ .tensor = inputs[0] },
                .dims = args,
            },
            .UnaryOp, .DataOp => .{
                .op = op,
                .a = .{ .tensor = inputs[0] },
            },
            .InitOp => .{
                .op = op,
                .args = args,
            },
        });
    }

    pub fn toJsonFormat(self: *const Record, out: *const anytensor) JsonFormat {
        return switch (self.*) {
            .UnaryOp => |record| .{ .UnaryOp = .{
                .op = record.op,
                .a = record.a,
                .out = @intFromPtr(out),
            } },
            .BinaryOp => |record| .{ .BinaryOp = .{
                .op = record.op,
                .a = record.a,
                .b = record.a,
                .out = @intFromPtr(out),
            } },
            .ReduceOp => |record| .{ .ReduceOp = .{
                .op = record.op,
                .a = record.a,
                .dims = record.dims,
                .out = @intFromPtr(out),
            } },
            .DataOp => |record| .{ .DataOp = .{
                .op = record.op,
                .a = record.a,
                .out = @intFromPtr(out),
            } },
            .InitOp => |record| .{ .InitOp = .{
                .op = record.op,
                .args = record.args,
                .out = @intFromPtr(out),
            } },
            .TernaryOp => |record| .{ .TernaryOp = .{
                .op = record.op,
                .a = record.a,
                .b = record.a,
                .c = record.a,
                .out = @intFromPtr(out),
            } },
        };
    }
};
