const std = @import("std");
const ops = @import("ops.zig");
const tensor = @import("tensor.zig");

const Graph = @This();

var arena: ?std.heap.ArenaAllocator = null;
var cache: std.AutoHashMap(usize, usize) = undefined;
var ids: std.AutoHashMap(usize, usize) = undefined;
var nodes: std.AutoHashMap(usize, *Node) = undefined;
var graph: *Graph = undefined;
var last_node: ?*Node = null;

pub fn init() void {
    arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    cache = std.AutoHashMap(usize, usize).init(arena.?.allocator());
    ids = std.AutoHashMap(usize, usize).init(arena.?.allocator());
    nodes = std.AutoHashMap(usize, *Node).init(arena.?.allocator());
    graph = arena.?.allocator().create(Graph) catch @panic("Out of memory");
}

pub fn deinit() void {
    arena.?.deinit();
    arena = null;
}

fn allocator() std.mem.Allocator {
    return arena.?.allocator();
}

pub const Link = union(ops.OpTypes) {
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

const Node = struct {
    id: usize,
    link: ?Link,
    visited: bool = false,

    ndims: u8,
    dtype: []const u8,
    shape: []const usize,
    strides: []const usize,

    pub fn print(node: *Node) void {
        if (node.link != null) {
            switch (node.link.?) {
                .MapOp => |link| {
                    if (node.visited) {
                        return;
                    }
                    node.visited = true;
                    link.x.print();
                    std.debug.print("tensor_{d} = {s} tensor_{d}\n", .{ node.id, @tagName(link.op), link.x.id });
                },
                .ZipOp => |link| {
                    if (node.visited) {
                        return;
                    }
                    node.visited = true;
                    link.a.print();
                    link.b.print();
                    std.debug.print("tensor_{d} = {s} tensor_{d} tensor_{d}\n", .{ node.id, @tagName(link.op), link.a.id, link.b.id });
                },
                .ReduceOp => |link| {
                    if (node.visited) {
                        return;
                    }
                    node.visited = true;
                    link.x.print();
                    std.debug.print("tensor_{d} = {s}({?}) tensor_{d}\n", .{ node.id, @tagName(link.op), link.dim, link.x.id });
                },
                .TypeOp => |link| {
                    if (node.visited) {
                        return;
                    }
                    node.visited = true;
                    link.x.print();
                    std.debug.print("tensor_{d} = {s}(Tensor{{{s}, {any}}}) tensor_{d}\n", .{ node.id, @tagName(link.op), node.dtype, node.shape, link.x.id });
                },
                .InitOp => |link| {
                    if (node.visited) {
                        return;
                    }
                    node.visited = true;
                    std.debug.print("tensor_{d} = {s} Tensor{{{s}, {any}}}\n", .{ node.id, @tagName(link.op), node.dtype, node.shape });
                },
            }
        }
    }
};

pub fn new_node(ptr: anytype, link: ?Link, comptime Tensor: type) void {
    const key = @intFromPtr(ptr);
    if (!ids.contains(key)) {
        const id = ids.count();
        ids.put(key, id) catch @panic("Out of memory");
        const n = allocator().create(Node) catch @panic("Out of memory");
        n.* = .{
            .id = id,
            .link = link,
            .ndims = Tensor.ndims,
            .dtype = @typeName(Tensor.dtype),
            .shape = Tensor.shape[0..],
            .strides = Tensor.strides[0..],
        };
        last_node = n;
        nodes.put(id, n) catch @panic("Out of memory");
    }
}

pub fn get_node(ptr: anytype) *Node {
    const id: usize = ids.get(@intFromPtr(ptr)).?;
    return nodes.get(id).?;
}

pub fn print() void {
    last_node.?.print();
}

pub fn cast(comptime new_dtype: type, x: anytype) @TypeOf(x.*).Cast(new_dtype) {
    const Out: type = @TypeOf(x.*).Cast(new_dtype);
    const impl = struct {
        fn trace(out: *const Out) void {
            x.trace();
            new_node(out, .{ .TypeOp = .{ .op = .Cast, .x = get_node(x) } }, Out);
        }
    };
    return Out.init(graph, impl.trace);
}

pub fn map(op: ops.MapOp, x: anytype) @TypeOf(x.*) {
    const Out: type = @TypeOf(x.*);
    const impl = struct {
        fn trace(out: *const Out) void {
            x.trace();
            new_node(out, .{ .MapOp = .{ .op = op, .x = get_node(x) } }, Out);
        }
    };
    return Out.init(impl.trace);
}

pub fn zip(op: ops.ZipOp, a: anytype, b: anytype) @TypeOf(a.*).Broadcast(@TypeOf(b.*)) {
    const Out: type = @TypeOf(a.*).Broadcast(@TypeOf(b.*));
    const impl = struct {
        fn trace(out: *const Out) void {
            a.trace();
            b.trace();
            new_node(out, .{ .ZipOp = .{ .op = op, .a = get_node(a), .b = get_node(b) } }, Out);
        }
    };
    return Out.init(impl.trace);
}

pub fn reduce(op: ops.ReduceOp, x: anytype, comptime dim: ?u8) @TypeOf(x.*).Reduce(dim) {
    const Out: type = @TypeOf(x.*).Reduce(dim);
    const impl = struct {
        fn trace(out: *const Out) void {
            x.trace();
            new_node(out, .{ .ReduceOp = .{ .op = op, .x = get_node(x), .dim = dim } }, Out);
        }
    };
    return Out.init(impl.trace);
}
