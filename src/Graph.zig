const std = @import("std");
const ops = @import("ops.zig");
const tensor = @import("tensor.zig");
const utils = @import("utils.zig");
const Graph = @This();
const dtypes = @import("dtypes.zig");

var arena: std.heap.ArenaAllocator = undefined;
var allocator: std.mem.Allocator = undefined;
var cache: std.AutoHashMap(usize, usize) = undefined;
var ids: std.AutoHashMap(usize, usize) = undefined;
var nodes: std.AutoHashMap(usize, *Vertex) = undefined;
var entrypoint: ?*Vertex = null;

pub fn init() void {
    arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    allocator = arena.allocator();
    cache = std.AutoHashMap(usize, usize).init(allocator);
    ids = std.AutoHashMap(usize, usize).init(allocator);
    nodes = std.AutoHashMap(usize, *Vertex).init(allocator);
}

pub fn deinit() void {
    arena.deinit();
}

pub const Edge = union(ops.OpTypes) {
    MapOp: struct {
        op: ops.MapOp,
        x: *Vertex,
    },
    ZipOp: struct {
        op: ops.ZipOp,
        a: *Vertex,
        b: *Vertex,
    },
    ReduceOp: struct {
        op: ops.ReduceOp,
        x: *Vertex,
        dims: []const u8,
    },
    TypeOp: struct {
        op: ops.TypeOp,
        x: *Vertex,
    },
    InitOp: struct {
        op: ops.InitOp,
    },
};

pub const Vertex = struct {
    id: usize,
    edge: Edge,
    visited: bool = false,

    // Tensor metadata which will be used for lowering and optimization
    ndims: u8,
    dtype: dtypes.DType,
    shape: []const usize,
    strides: []const usize,

    fn register(ptr: anytype, edge: Edge, comptime TensorType: type) !void {
        const key = @intFromPtr(ptr);
        if (!ids.contains(key)) {
            const id = ids.count();
            try ids.put(key, id);
            const vertex = try allocator.create(Vertex);
            vertex.* = .{
                .id = id,
                .edge = edge,
                .ndims = TensorType.ndims,
                .dtype = TensorType.dtype,
                .shape = TensorType.shape[0..],
                .strides = TensorType.strides[0..],
            };
            entrypoint = vertex;
            try nodes.put(id, vertex);
        }
    }

    pub fn new(ptr: anytype, edge: Edge, comptime TensorType: type) void {
        register(ptr, edge, TensorType) catch @panic("Out of memory");
    }

    pub fn get(ptr: anytype) *Vertex {
        return nodes.get(ids.get(@intFromPtr(ptr)).?).?;
    }

    pub fn viz(v: *Vertex) void {
        if (v.visited) {
            return;
        }
        v.visited = true;
        switch (v.edge) {
            .InitOp => |e| {
                std.debug.print("t{d} : ({s}, {any}) = {s}({s}, {any})\n", .{ v.id, @tagName(v.dtype), v.shape, @tagName(e.op), @tagName(v.dtype), v.shape });
            },
            .MapOp => |e| {
                e.x.viz();
                std.debug.print("t{d} : ({s}, {any}) = {s}(t{d})\n", .{ v.id, @tagName(v.dtype), v.shape, @tagName(e.op), e.x.id });
            },
            .ZipOp => |e| {
                e.a.viz();
                e.b.viz();
                std.debug.print("t{d} : ({s}, {any}) = {s}(t{d}, t{d})\n", .{ v.id, @tagName(v.dtype), v.shape, @tagName(e.op), e.a.id, e.b.id });
            },
            .ReduceOp => |e| {
                e.x.viz();
                std.debug.print("t{d} : ({s}, {any}) = {s}(t{d}, {any})\n", .{ v.id, @tagName(v.dtype), v.shape, @tagName(e.op), e.x.id, e.dims });
            },
            .TypeOp => |e| {
                e.x.viz();
                std.debug.print("t{d} : ({s}, {any}) = {s}({s}, {any}, {any})\n", .{ v.id, @tagName(v.dtype), v.shape, @tagName(e.op), @tagName(v.dtype), v.shape, v.strides });
            },
        }
    }
};

pub fn viz() void {
    std.debug.print("\n", .{});
    entrypoint.?.viz();
}
