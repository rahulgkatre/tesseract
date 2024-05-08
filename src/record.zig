const ops = @import("ops.zig");
const anytensor = @import("anytensor.zig").anytensor;

pub const Record = union(ops.OpTypes) {
    UnaryOp: struct {
        op: ops.UnaryOp,
        a: *const anytensor,
        ctx: ?[]const u8 = null,

        pub fn jsonStringify(self: @This(), write_stream: anytype) !void {
            try write_stream.write(.{ .op = self.op, .a = @intFromPtr(self.a) });
        }
    },
    BinaryOp: struct {
        op: ops.BinaryOp,
        a: *const anytensor,
        b: *const anytensor,
        ctx: ?[]const u8 = null,

        pub fn jsonStringify(self: @This(), write_stream: anytype) !void {
            try write_stream.write(.{ .op = self.op, .a = @intFromPtr(self.a), .b = @intFromPtr(self.b) });
        }
    },
    ReduceOp: struct {
        op: ops.ReduceOp,
        a: *const anytensor,
        dims: []const bool,
        ctx: ?[]const u8 = null,

        pub fn jsonStringify(self: @This(), write_stream: anytype) !void {
            try write_stream.write(.{ .op = self.op, .a = @intFromPtr(self.a), .dims = self.dims });
        }
    },
    ArrayOp: struct {
        op: ops.ArrayOp,
        a: *const anytensor,
        ctx: ?[]const u8 = null,

        pub fn jsonStringify(self: @This(), write_stream: anytype) !void {
            try write_stream.write(.{ .op = self.op, .a = @intFromPtr(self.a) });
        }
    },
    InitOp: struct {
        op: ops.InitOp,
        args: ops.InitOp.Args,
        ctx: ?[]const u8 = null,

        pub fn jsonStringify(self: @This(), write_stream: anytype) !void {
            try write_stream.write(.{ .op = self.op, .args = @intFromPtr(self.args) });
        }
    },
    TernaryOp: struct {
        op: ops.TernaryOp,
        a: *const anytensor,
        b: *const anytensor,
        c: *const anytensor,
        ctx: ?[]const u8 = null,

        pub fn jsonStringify(self: @This(), write_stream: anytype) !void {
            try write_stream.write(.{ .op = self.op, .a = @intFromPtr(self.a), .b = @intFromPtr(self.b), .c = @intFromPtr(self.c) });
        }
    },
    CustomOp: struct {
        op_name: []const u8,
        inputs: []const *const anytensor,

        pub fn jsonStringify(self: @This(), write_stream: anytype) !void {
            try write_stream.write(.{
                .op_type = self.op_type,
                .op_name = self.op_name,
                .inputs = self.inputs,
            });
        }
    },

    pub fn init(
        comptime tag: ops.OpTypes,
        comptime op: @field(ops, @tagName(tag)),
        inputs: switch (tag) {
            .TernaryOp => [3]*const anytensor,
            .BinaryOp => [2]*const anytensor,
            .UnaryOp, .ArrayOp, .ReduceOp => [1]*const anytensor,
            .InitOp => void,
            .CustomOp => []const *const anytensor,
        },
        args: switch (tag) {
            .ReduceOp => []const bool,
            .InitOp => ops.InitOp.Args,
            .CustomOp => []const u8,
            else => void,
        },
    ) Record {
        return @unionInit(Record, @tagName(tag), switch (tag) {
            .TernaryOp => .{
                .op = op,
                .a = inputs[0],
                .b = inputs[1],
                .c = inputs[2],
            },
            .BinaryOp => .{
                .op = op,
                .a = inputs[0],
                .b = inputs[1],
            },
            .ReduceOp => .{
                .op = op,
                .a = inputs[0],
                .dims = args,
            },
            .UnaryOp, .ArrayOp => .{
                .op = op,
                .a = inputs[0],
            },
            .InitOp => .{
                .op = op,
                .args = args,
            },
            .CustomOp => .{
                .op_name = args,
                .inputs = inputs,
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
                .b = record.b,
                .out = @intFromPtr(out),
            } },
            .ReduceOp => |record| .{ .ReduceOp = .{
                .op = record.op,
                .a = record.a,
                .dims = record.dims,
                .out = @intFromPtr(out),
            } },
            .ArrayOp => |record| .{ .ArrayOp = .{
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
                .b = record.b,
                .c = record.c,
                .out = @intFromPtr(out),
            } },
            .CustomOp => |record| .{ .CustomOp = .{
                .op_name = record.op_name,
                .inputs = record.inputs,
            } },
        };
    }

    pub const JsonFormat = union(ops.OpTypes) {
        UnaryOp: struct {
            op: ops.UnaryOp,
            a: *const anytensor,
            out: usize,
        },
        BinaryOp: struct {
            op: ops.BinaryOp,
            a: *const anytensor,
            b: *const anytensor,
            out: usize,
        },
        ReduceOp: struct {
            op: ops.ReduceOp,
            a: *const anytensor,
            dims: []const bool,
            out: usize,
        },
        ArrayOp: struct {
            op: ops.ArrayOp,
            a: *const anytensor,
            out: usize,
        },
        InitOp: struct {
            op: ops.InitOp,
            args: ops.InitOp.Args,
            out: usize,
        },
        TernaryOp: struct {
            op: ops.TernaryOp,
            a: *const anytensor,
            b: *const anytensor,
            c: *const anytensor,
            out: usize,
        },
        CustomOp: struct {
            op_name: []const u8,
            inputs: []const *const anytensor,
            out: usize,
        },
    };
};
