const AnyTensor = @import("anytensor.zig").AnyTensor;
const std = @import("std");
const utils = @import("utils.zig");
const dtypes = @import("dtypes.zig");

// Arithmetic operations for unary functions, binary functions,
// and reducing a dimension of a tensor to a single value by applying some binary function
pub const UnaryOp = enum {
    pub const Instr = struct {
        op: UnaryOp,
        in: [1]*const AnyTensor,
        args: Args = {},
    };
    pub const Args = void;
    pub const Json = struct {
        op: UnaryOp,
        in: [1]usize,
        out: usize,
    };

    neg,
    log2,
    exp2,
    sqrt,
    recip,
    sin,
};
// Lt, Eq, Xor will produce a bool tensor which can be used in mask based operations later on
pub const BinaryOp = enum {
    pub const Instr = struct {
        op: BinaryOp,
        in: [2]*const AnyTensor,
        args: Args = {},
    };
    pub const Args = void;
    pub const Json = struct {
        op: BinaryOp,
        in: [2]usize,
        out: usize,
    };

    add,
    mul,
    max,
    mod,
    less_than,
    equals,
    xor,
};
// Ternary ops take in 3 arguments which can have different purposes
pub const TernaryOp = enum {
    pub const Instr = struct {
        op: TernaryOp,
        in: [3]*const AnyTensor,
        args: Args = {},
    };
    pub const Args = void;
    pub const Json = struct {
        op: TernaryOp,
        in: [3]usize,
        out: usize,
    };
    where,
};
// ReduceOps are just recurrently applied binary ops
pub const ReduceOp = enum {
    pub const Instr = struct {
        op: ReduceOp,
        in: [1]*const AnyTensor,
        args: Args,
    };
    pub const Args = struct {
        dims: []const u16,
        mask: []const bool,

        pub fn format(
            self: Args,
            comptime _: []const u8,
            _: std.fmt.FormatOptions,
            writer: anytype,
        ) anyerror!void {
            try std.fmt.format(writer, ".dims = {any} .mask = {any}\t", .{ self.dims, self.mask });
        }
    };
    pub const Json = struct {
        op: ReduceOp,
        in: [1]usize,
        args: Args,
        out: usize,
    };

    add,
    mul,
    max,
    xor,

    pub fn binaryOp(reduceOp: ReduceOp) BinaryOp {
        return @field(BinaryOp, @tagName(reduceOp));
    }
};
// TypeOps mutate the type of the tensor, in Tesseract's case this not only changes
// the dtype but also the shape, so any shape affecting ops are TypeOps

pub const DataOp = enum {
    pub const Instr = struct {
        op: DataOp,
        in: [1]*const AnyTensor,
        args: Args,
    };
    pub const Args = union(DataOp) {
        pub const View = struct {
            shape: []const u64,
            strides: []const u64,
            offset: u64,
        };
        pub const Pad = struct {
            pub const Mode = enum {
                constant,
                reflect,
                replicate,
                circular,
            };

            padding: []const [2]u64,
            mode: union(Mode) {
                constant: []const u8,
                reflect: void,
                replicate: void,
                circular: void,
            },
        };
        view: View,
        cast: dtypes.DType,
        pad: Pad,
        contiguous: View,

        pub fn format(
            self: Args,
            comptime _: []const u8,
            _: std.fmt.FormatOptions,
            writer: anytype,
        ) anyerror!void {
            switch (self) {
                .pad => |pad| try std.fmt.format(writer, "{}\t", .{pad}),
                .cast => |cast| try std.fmt.format(writer, "{s}\t", .{utils.rawTagName(cast)}),
                .view, .contiguous => |view| {
                    try std.fmt.format(writer, ".shape = {[shape]any} .strides = {[strides]any} .offset = {[offset]d}", view);
                },
            }
        }
    };
    pub const Json = struct {
        op: DataOp,
        in: [1]usize,
        out: usize,
    };

    view,
    cast,
    pad,
    contiguous,
};
pub const InitOp = enum {
    pub const Instr = struct {
        op: InitOp,
        in: [0]*const AnyTensor = .{},
        args: InitOp.Args,
    };
    pub const Args = union(InitOp) {
        pub const Full = []const u8;
        pub const Range = struct {
            start: []const u8,
            stop: []const u8,
        };

        empty: void,
        input: void,
        parameter: void,
        full: Full,
        random: void,
        range: Range,

        pub fn format(
            self: Args,
            comptime _: []const u8,
            _: std.fmt.FormatOptions,
            writer: anytype,
        ) anyerror!void {
            switch (self) {
                .input, .parameter, .empty, .random => {},
                .full => |full| try std.fmt.format(writer, ".value = {s}", .{full}),
                .range => |range| try std.fmt.format(writer, ".start = {[start]s} .stop = {[stop]s}", range),
            }
        }
    };
    pub const Json = struct {
        op: InitOp,
        args: Args,
        out: usize,
    };
    empty,
    input,
    parameter,
    full,
    random,
    range,
};
pub const OpTypes = enum {
    InitOp,
    UnaryOp,
    BinaryOp,
    TernaryOp,
    ReduceOp,
    DataOp,
};

pub const Instruction = union(OpTypes) {
    InitOp: InitOp.Instr,
    UnaryOp: UnaryOp.Instr,
    BinaryOp: BinaryOp.Instr,
    TernaryOp: TernaryOp.Instr,
    ReduceOp: ReduceOp.Instr,
    DataOp: DataOp.Instr,

    pub const Json = union(OpTypes) {
        InitOp: InitOp.Json,
        UnaryOp: UnaryOp.Json,
        BinaryOp: BinaryOp.Json,
        TernaryOp: TernaryOp.Json,
        ReduceOp: ReduceOp.Json,
        DataOp: DataOp.Json,
    };

    pub fn format(
        self: Instruction,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) anyerror!void {
        switch (self) {
            inline else => |instr| {
                std.debug.assert(fmt.len == 0 or std.mem.eql(u8, fmt, "any"));
                try std.fmt.format(writer, "{s: <10}\t", .{utils.rawTagName(instr.op)});
                for (instr.in, 0..) |in, i| {
                    try writer.writeAll(".in[");
                    try std.fmt.formatInt(i, 10, .lower, options, writer);
                    try writer.writeAll("] = %");
                    try std.fmt.formatInt(@intFromPtr(in), 16, .lower, options, writer);
                    try writer.writeAll(" ");
                }
                if (@TypeOf(instr.args) != void) {
                    try std.fmt.format(writer, "{}", .{instr.args});
                }
            },
        }
    }
};
