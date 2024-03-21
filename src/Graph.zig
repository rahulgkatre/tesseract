const std = @import("std");
const ops = @import("ops.zig");
const utils = @import("utils.zig");
const Graph = @This();
const dtypes = @import("dtypes.zig");

var gpa: std.heap.GeneralPurposeAllocator(.{}) = undefined;
var arena: std.heap.ArenaAllocator = undefined;
var tensors: std.AutoArrayHashMap(usize, *AnyTensor) = undefined;

pub fn init() void {
    gpa = .{};
    arena = std.heap.ArenaAllocator.init(gpa.allocator());
    tensors = std.AutoArrayHashMap(usize, *AnyTensor).init(arena.allocator());
}

pub fn deinit() void {
    arena.deinit();
    _ = gpa.deinit();
}

/// Strips away comptime / generic information to make it easier to work with pointers to tensors
pub const AnyTensor = struct {
    ptr: *const anyopaque = undefined,
    dtype: dtypes.DType,
    ndims: u8,
    shape: []const u64,
    strides: []const u64,
    offset: u64,
    // size: u64,
    // contiguous: bool,
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
            inline else => |unary_op| unary_op.x.trace(),
        }
        var out: *AnyTensor = @constCast(self);
        out.ptr = out;
        const key = @intFromPtr(self);
        switch (out.last_op) {
            inline else => |*last_op| last_op.out = out,
        }
        if (!tensors.contains(key)) {
            tensors.put(key, out) catch unreachable;
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
            // .size = self.size,
            // .contiguous = self.contiguous,
        });
    }

    pub fn get(tensor: anytype) *const AnyTensor {
        return tensors.get(@intFromPtr(&tensor.any_tensor)).?;
    }
};

pub const AnyOp = union(ops.OpTypes) {
    MapOp: struct {
        op: ops.MapOp,
        x: *const AnyTensor,
        out: *const AnyTensor = undefined,
    },
    ZipOp: struct {
        op: ops.ZipOp,
        a: *const AnyTensor,
        b: *const AnyTensor,
        out: *const AnyTensor = undefined,
    },
    ReduceOp: struct {
        op: ops.ReduceOp,
        x: *const AnyTensor,
        dims: []const bool,
        out: *const AnyTensor = undefined,
    },
    TypeOp: struct {
        op: ops.TypeOp,
        x: *const AnyTensor,
        out: *const AnyTensor = undefined,
    },
    InitOp: struct {
        op: ops.InitOp,
        args: ops.InitOp.Args,
        out: *const AnyTensor = undefined,
    },
    TernaryOp: struct {
        op: ops.TernaryOp,
        a: *const AnyTensor,
        b: *const AnyTensor,
        c: *const AnyTensor,
        out: *const AnyTensor = undefined,
    },

    pub fn jsonStringify(self: AnyOp, write_stream: anytype) !void {
        switch (self) {
            .ZipOp => |binary| try write_stream.write(.{
                .op = binary.op,
                .a = @intFromPtr(binary.a),
                .b = @intFromPtr(binary.b),
                .out = @intFromPtr(binary.out),
            }),
            .TernaryOp => |ternary| try write_stream.write(.{
                .op = ternary.op,
                .a = @intFromPtr(ternary.a),
                .b = @intFromPtr(ternary.b),
                .c = @intFromPtr(ternary.c),
                .out = @intFromPtr(ternary.out),
            }),
            .InitOp => |_init| try write_stream.write(.{
                .op = _init.op,
                .args = _init.args,
                .out = @intFromPtr(_init.out),
            }),
            inline else => |unary| try write_stream.write(.{
                .op = unary.op,
                .x = @intFromPtr(unary.x),
                .out = @intFromPtr(unary.out),
            }),
        }
    }
};

pub fn jsonStringify(_: Graph, write_stream: anytype) !void {
    const operations: []AnyOp = gpa.allocator().alloc(AnyOp, tensors.count()) catch unreachable;
    defer gpa.allocator().free(operations);
    for (tensors.values(), operations) |t, *operation| {
        operation.* = t.last_op;
    }
    try write_stream.write(.{
        .tensors = tensors.values(),
        .operations = operations,
    });
}
