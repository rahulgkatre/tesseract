const std = @import("std");
const ops = @import("ops.zig");
const tensor = @import("tensor.zig");
const utils = @import("utils.zig");
const Graph = @This();

var arena: std.heap.ArenaAllocator = undefined;
var allocator: std.mem.Allocator = undefined;
var cache: std.AutoHashMap(usize, usize) = undefined;
var ids: std.AutoHashMap(usize, usize) = undefined;
var nodes: std.AutoHashMap(usize, *Node) = undefined;
var last_node: ?*Node = null;

pub fn init() void {
    arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    allocator = arena.allocator();
    cache = std.AutoHashMap(usize, usize).init(allocator);
    ids = std.AutoHashMap(usize, usize).init(allocator);
    nodes = std.AutoHashMap(usize, *Node).init(allocator);
}

pub fn deinit() void {
    arena.deinit();
}

const Link = union(ops.OpTypes) {
    MapOp: struct {
        op: ops.MapOp,
        x: *Node,
    },
    ZipOp: struct {
        op: ops.ZipOp,
        a: *Node,
        b: *Node,
    },
    ReduceOp: struct {
        op: ops.ReduceOp,
        x: *Node,
        dim: ?u8,
    },
    TypeOp: struct {
        op: ops.TypeOp,
        x: *Node,
    },
    InitOp: struct {
        op: ops.InitOp,
    },
};

pub const Node = struct {
    id: usize,
    link: Link,
    visited: bool = false,
    str: []const u8,

    // Tensor metadata which will be used for lowering and optimization
    ndims: u8,
    dtype: []const u8,
    shape: []const usize,
    strides: []const usize,

    pub fn new(ptr: anytype, link: Link, comptime TensorType: type) void {
        const key = @intFromPtr(ptr);
        if (!ids.contains(key)) {
            const id = ids.count();
            ids.put(key, id) catch @panic("Out of memory");
            const n = allocator.create(Node) catch @panic("Out of memory");
            n.* = .{
                .id = id,
                .link = link,
                .str = utils.tensorString(TensorType),
                .ndims = TensorType.ndims,
                .dtype = @typeName(TensorType.dtype),
                .shape = TensorType.shape[0..],
                .strides = TensorType.strides[0..],
            };
            last_node = n;
            nodes.put(id, n) catch @panic("Out of memory");
        }
    }

    pub fn get(ptr: anytype) *Node {
        const id: usize = ids.get(@intFromPtr(ptr)).?;
        return nodes.get(id).?;
    }

    pub fn show(node: *Node) void {
        switch (node.link) {
            .MapOp => |link| {
                if (node.visited) {
                    return;
                }
                node.visited = true;
                link.x.show();
                std.debug.print("%{d} = \"MapOp.{s}\" (%{d}) : ({s}) -> {s}\n", .{ node.id, @tagName(link.op), link.x.id, link.x.str, node.str });
            },
            .ZipOp => |link| {
                if (node.visited) {
                    return;
                }
                node.visited = true;
                link.a.show();
                link.b.show();
                std.debug.print("%{d} = \"ZipOp.{s}\" (%{d}, %{d}) : ({s}, {s}) -> {s} \n", .{ node.id, @tagName(link.op), link.a.id, link.b.id, link.a.str, link.b.str, node.str });
            },
            .ReduceOp => |link| {
                if (node.visited) {
                    return;
                }
                node.visited = true;
                link.x.show();
                std.debug.print("%{d} = \"ReduceOp.{s}\" (%{d}) {{ dim = {?} }} : ({s}) -> {s}\n", .{ node.id, @tagName(link.op), link.x.id, link.dim, link.x.str, node.str });
            },
            .TypeOp => |link| {
                if (node.visited) {
                    return;
                }
                node.visited = true;
                link.x.show();
                std.debug.print("%{d} = \"TypeOp.{s}\" (%{d}) : ({s}) -> {s}\n", .{ node.id, @tagName(link.op), link.x.id, link.x.str, node.str });
            },
            .InitOp => |link| {
                if (node.visited) {
                    return;
                }
                node.visited = true;
                std.debug.print("%{d} = \"InitOp.{s}\" () -> {s}\n", .{ node.id, @tagName(link.op), node.str });
            },
        }
    }
};

pub fn show() void {
    last_node.?.show();
}
