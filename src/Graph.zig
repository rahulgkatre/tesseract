const std = @import("std");
const ops = @import("ops.zig");
const utils = @import("utils.zig");
const Graph = @This();
const dtypes = @import("dtypes.zig");

var gpa: std.heap.GeneralPurposeAllocator(.{}) = undefined;
var arena: std.heap.ArenaAllocator = undefined;
var tensors: std.AutoArrayHashMap(usize, TensorNode) = undefined;

pub fn init() void {
    gpa = .{};
    arena = std.heap.ArenaAllocator.init(gpa.allocator());
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
    size: u64,
    contiguous: bool,
    op_node: OpNode,

    pub fn trace(self: *const TensorNode) void {
        switch (self.op_node) {
            .TernaryOp => |ternary_op| {
                ternary_op.a.trace();
                ternary_op.b.trace();
                ternary_op.c.trace();
            },
            .ZipOp => |binary_op| {
                binary_op.a.trace();
                binary_op.b.trace();
            },
            .InitOp => {},
            inline else => |unary_op| {
                unary_op.a.trace();
            },
        }
        const key = @intFromPtr(self);
        if (!tensors.contains(key)) {
            tensors.put(key, self.*) catch unreachable;
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
            .size = self.size,
            .contiguous = self.contiguous,
        });
    }
};

pub const OpNode = union(ops.OpTypes) {
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
        a: *const TensorNode,
    },
    ZipOp: struct {
        op: ops.ZipOp,
        a: *const TensorNode,
        b: *const TensorNode,
    },
    ReduceOp: struct {
        op: ops.ReduceOp,
        a: *const TensorNode,
        dims: []const bool,
    },
    TypeOp: struct {
        op: ops.TypeOp,
        a: *const TensorNode,
    },
    InitOp: struct {
        op: ops.InitOp,
        args: ops.InitOp.Args,
    },
    TernaryOp: struct {
        op: ops.TernaryOp,
        a: *const TensorNode,
        b: *const TensorNode,
        c: *const TensorNode,
    },

    fn toJsonFormat(self: OpNode, out: *const TensorNode) JsonFormat {
        return switch (self) {
            .MapOp => |op_node| .{ .MapOp = .{
                .op = op_node.op,
                .a = @intFromPtr(op_node.a),
                .out = @intFromPtr(out.ptr),
            } },
            .ZipOp => |op_node| .{ .ZipOp = .{
                .op = op_node.op,
                .a = @intFromPtr(op_node.a),
                .b = @intFromPtr(op_node.b),
                .out = @intFromPtr(out.ptr),
            } },
            .ReduceOp => |op_node| .{ .ReduceOp = .{
                .op = op_node.op,
                .a = @intFromPtr(op_node.a),
                .dims = op_node.dims,
                .out = @intFromPtr(out.ptr),
            } },
            .TypeOp => |op_node| .{ .TypeOp = .{
                .op = op_node.op,
                .a = @intFromPtr(op_node.a),
                .out = @intFromPtr(out.ptr),
            } },
            .InitOp => |op_node| .{ .InitOp = .{
                .op = op_node.op,
                .args = op_node.args,
                .out = @intFromPtr(out.ptr),
            } },
            .TernaryOp => |op_node| .{ .TernaryOp = .{
                .op = op_node.op,
                .a = @intFromPtr(op_node.a),
                .b = @intFromPtr(op_node.b),
                .c = @intFromPtr(op_node.c),
                .out = @intFromPtr(out.ptr),
            } },
        };
    }
};

pub fn jsonStringify(_: Graph, write_stream: anytype) !void {
    const operations: []OpNode.JsonFormat = gpa.allocator().alloc(OpNode.JsonFormat, tensors.count()) catch unreachable;
    defer gpa.allocator().free(operations);
    for (tensors.values(), operations) |t, *operation| {
        operation.* = t.op_node.toJsonFormat(&t);
    }
    try write_stream.write(.{
        .tensors = tensors.values(),
        .operations = operations,
    });
}
