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
var graph: *Graph = undefined;
var last_node: ?*Node = null;

pub fn init() void {
    arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    allocator = arena.allocator();
    cache = std.AutoHashMap(usize, usize).init(allocator);
    ids = std.AutoHashMap(usize, usize).init(allocator);
    nodes = std.AutoHashMap(usize, *Node).init(allocator);
    graph = allocator.create(Graph) catch @panic("Out of memory");
}

pub fn deinit() void {
    arena.deinit();
}

const Node = struct {
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

    id: usize,
    link: ?Link,
    visited: bool = false,
    str: []const u8,

    ndims: u8,
    dtype: []const u8,
    shape: []const usize,
    strides: []const usize,

    pub fn show(node: *Node) void {
        if (node.link != null) {
            switch (node.link.?) {
                .MapOp => |link| {
                    if (node.visited) {
                        return;
                    }
                    node.visited = true;
                    link.x.show();
                    std.debug.print("\t%{d} = \"ops.MapOp.{s}\" (%{d}) : ({s}) -> {s}\n", .{ node.id, @tagName(link.op), link.x.id, link.x.str, node.str });
                },
                .ZipOp => |link| {
                    if (node.visited) {
                        return;
                    }
                    node.visited = true;
                    link.a.show();
                    link.b.show();
                    std.debug.print("\t%{d} = \"ops.ZipOp.{s}\" (%{d}, %{d}) : ({s}, {s}) -> {s} \n", .{ node.id, @tagName(link.op), link.a.id, link.b.id, link.a.str, link.b.str, node.str });
                },
                .ReduceOp => |link| {
                    if (node.visited) {
                        return;
                    }
                    node.visited = true;
                    link.x.show();
                    std.debug.print("\t%{d} = \"ops.ReduceOp.{s}\" (%{d}) {{ dim = {?} }} : ({s}) -> {s}\n", .{ node.id, @tagName(link.op), link.x.id, link.dim, link.x.str, node.str });
                },
                .TypeOp => |link| {
                    if (node.visited) {
                        return;
                    }
                    node.visited = true;
                    link.x.show();
                    std.debug.print("\t%{d} = \"ops.TypeOp.{s}\" (%{d}) : ({s}) -> {s}\n", .{ node.id, @tagName(link.op), link.x.id, link.x.str, node.str });
                },
                .InitOp => |link| {
                    if (node.visited) {
                        return;
                    }
                    node.visited = true;
                    std.debug.print("\t%{d} = \"ops.InitOp.{s}\" () -> {s}\n", .{ node.id, @tagName(link.op), node.str });
                },
            }
        }
    }
};

pub fn new_node(ptr: anytype, link: ?Node.Link, comptime Tensor: type) void {
    const key = @intFromPtr(ptr);
    if (!ids.contains(key)) {
        const id = ids.count();
        ids.put(key, id) catch @panic("Out of memory");
        const n = allocator.create(Node) catch @panic("Out of memory");
        n.* = .{
            .id = id,
            .link = link,
            .str = utils.tensorString(Tensor),
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

pub fn show() void {
    std.debug.print("func @main() {{\n", .{});
    last_node.?.show();
    std.debug.print("\t\"return\" (%{d}) : ({s}) -> ()\n", .{ last_node.?.id, last_node.?.str });
    std.debug.print("}}\n", .{});
}

pub fn cast(comptime new_dtype: type, x: anytype) @TypeOf(x.*).as_type(new_dtype) {
    const Out: type = @TypeOf(x.*).as_type(new_dtype);
    const impl = struct {
        fn trace(out: *const Out) void {
            x.trace();
            new_node(out, .{ .TypeOp = .{ .op = .AsType, .x = get_node(x) } }, Out);
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
