const ops = @import("ops.zig");
const anytensor = @import("anytensor.zig");

pub const Record = union(ops.OpTypes) {
    const Input = struct {
        tensor: *const anytensor,
        fused: bool = false,

        pub fn jsonStringify(input: @This(), write_stream: anytype) !void {
            try write_stream.write(input.tensor.ordinal());
        }
    };

    pub const JsonFormat = union(ops.OpTypes) {
        MapOp: struct {
            op: ops.MapOp,
            a: Input,
            out: usize,
        },
        ZipOp: struct {
            op: ops.ZipOp,
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
        TypeOp: struct {
            op: ops.TypeOp,
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
            .TernaryOp => [3]*const anytensor,
            .ZipOp => [2]*const anytensor,
            .MapOp, .TypeOp, .ReduceOp => [1]*const anytensor,
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

    pub fn toJsonFormat(self: *const Record) JsonFormat {
        const out: *const anytensor = @fieldParentPtr(anytensor, "record", self);
        return switch (self.*) {
            .MapOp => |record| .{ .MapOp = .{
                .op = record.op,
                .a = record.a,
                .out = @intFromPtr(out),
            } },
            .ZipOp => |record| .{ .ZipOp = .{
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
            .TypeOp => |record| .{ .TypeOp = .{
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
