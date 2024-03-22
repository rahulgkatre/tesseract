const std = @import("std");
const ops = @import("ops.zig");
const utils = @import("utils.zig");
const Graph = @This();
const dtypes = @import("dtypes.zig");

var gpa: std.heap.GeneralPurposeAllocator(.{}) = undefined;
var arena: std.heap.ArenaAllocator = undefined;
var tensors: std.AutoArrayHashMap(usize, AnyTensor) = undefined;

pub fn init() void {
    gpa = .{};
    arena = std.heap.ArenaAllocator.init(gpa.allocator());
    tensors = std.AutoArrayHashMap(usize, AnyTensor).init(arena.allocator());
}

pub fn deinit() void {
    arena.deinit();
    _ = gpa.deinit();
}

/// Strips away comptime / generic information to make it easier to work with pointers to tensors
pub const AnyTensor = struct {
    ptr: *const anyopaque,
    dtype: dtypes.DType,
    ndims: u8,
    shape: []const u64,
    strides: []const u64,
    offset: u64,
    size: u64,
    contiguous: bool,
    last_op: AnyOp,

    pub fn trace(self: *const AnyTensor) void {
        switch (self.last_op) {
            .ZipOp => |binary_op| {
                binary_op.a.trace();
                binary_op.b.trace();
            },
            .InitOp => {},
            .TernaryOp => |ternary_op| {
                ternary_op.a.trace();
                ternary_op.b.trace();
                ternary_op.c.trace();
            },
            inline else => |unary_op| unary_op.a.trace(),
        }
        const key = @intFromPtr(self);
        if (!tensors.contains(key)) {
            tensors.put(key, self.*) catch unreachable;
        }
    }

    pub fn jsonStringify(self: *const AnyTensor, write_stream: anytype) !void {
        try write_stream.write(.{
            .ptr = @intFromPtr(self.ptr),
            .dtype = self.dtype,
            .ndims = self.ndims,
            .shape = self.shape,
            .strides = self.strides,
            .offset = self.offset,
            .size = self.size,
            .contiguous = self.contiguous,
        });
    }
};

pub const AnyOp = union(ops.OpTypes) {
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

        pub fn jsonStringify(self: JsonFormat, write_stream: anytype) !void {
            switch (self) {
                inline else => |data| try write_stream.write(data),
            }
        }
    };

    MapOp: struct {
        op: ops.MapOp,
        a: *const AnyTensor,
    },
    ZipOp: struct {
        op: ops.ZipOp,
        a: *const AnyTensor,
        b: *const AnyTensor,
    },
    ReduceOp: struct {
        op: ops.ReduceOp,
        a: *const AnyTensor,
        dims: []const bool,
    },
    TypeOp: struct {
        op: ops.TypeOp,
        a: *const AnyTensor,
    },
    InitOp: struct {
        op: ops.InitOp,
        args: ops.InitOp.Args,
    },
    TernaryOp: struct {
        op: ops.TernaryOp,
        a: *const AnyTensor,
        b: *const AnyTensor,
        c: *const AnyTensor,
    },

    fn toJsonFormat(self: AnyOp, out: *const AnyTensor) JsonFormat {
        return switch (self) {
            .MapOp => |aop| .{ .MapOp = .{
                .op = aop.op,
                .a = @intFromPtr(aop.a),
                .out = @intFromPtr(out.ptr),
            } },
            .ZipOp => |aop| .{ .ZipOp = .{
                .op = aop.op,
                .a = @intFromPtr(aop.a),
                .b = @intFromPtr(aop.b),
                .out = @intFromPtr(out.ptr),
            } },
            .ReduceOp => |aop| .{ .ReduceOp = .{
                .op = aop.op,
                .a = @intFromPtr(aop.a),
                .dims = aop.dims,
                .out = @intFromPtr(out.ptr),
            } },
            .TypeOp => |aop| .{ .TypeOp = .{
                .op = aop.op,
                .a = @intFromPtr(aop.a),
                .out = @intFromPtr(out.ptr),
            } },
            .InitOp => |aop| .{ .InitOp = .{
                .op = aop.op,
                .args = aop.args,
                .out = @intFromPtr(out.ptr),
            } },
            .TernaryOp => |aop| .{ .TernaryOp = .{
                .op = aop.op,
                .a = @intFromPtr(aop.a),
                .b = @intFromPtr(aop.b),
                .c = @intFromPtr(aop.c),
                .out = @intFromPtr(out.ptr),
            } },
        };
    }
};

pub fn jsonStringify(_: Graph, write_stream: anytype) !void {
    const operations: []AnyOp.JsonFormat = gpa.allocator().alloc(AnyOp.JsonFormat, tensors.count()) catch unreachable;
    defer gpa.allocator().free(operations);
    for (tensors.values(), operations) |t, *operation| {
        operation.* = t.last_op.toJsonFormat(&t);
    }
    try write_stream.write(.{
        .tensors = tensors.values(),
        .operations = operations,
    });
}
