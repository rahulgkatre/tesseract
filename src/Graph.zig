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

pub const Node = struct {
    id: usize,
    link: Link,
    visited: bool = false,
    str: []const u8,

    // Tensor metadata which will be used for lowering and optimization
    ndims: u8,
    dtype: dtypes.DType,
    shape: []const usize,
    strides: []const usize,

    // Track which nodes are going to be fused (part of the same loop)
    fused: std.ArrayList(*Node),

    fn register(ptr: anytype, link: Link, comptime TensorType: type) !void {
        const key = @intFromPtr(ptr);
        if (!ids.contains(key)) {
            const id = ids.count();
            try ids.put(key, id);
            const n = try allocator.create(Node);
            n.* = .{
                .id = id,
                .link = link,
                .str = utils.tensorString(TensorType),
                .ndims = TensorType.ndims,
                .dtype = TensorType.dtype,
                .shape = TensorType.shape[0..],
                .strides = TensorType.strides[0..],
                .fused = std.ArrayList(*Node).init(allocator),
            };
            last_node = n;
            try nodes.put(id, n);
        }
    }

    pub fn new(ptr: anytype, link: Link, comptime TensorType: type) void {
        register(ptr, link, TensorType) catch @panic("Out of memory");
    }

    pub fn get(ptr: anytype) *Node {
        const id: usize = ids.get(@intFromPtr(ptr)).?;
        return nodes.get(id).?;
    }

    pub fn show(node: *Node) void {
        if (node.visited) {
            return;
        }
        node.visited = true;
        // TODO: Improve the visualization of fused ops
        // Should probably print all the fused ops in addition the current one
        switch (node.link) {
            .MapOp => |map_link| {
                var fused_x = false;
                for (node.fused.items) |fused_node| {
                    fused_x = fused_x or (map_link.x == fused_node);
                }
                map_link.x.show();
                std.debug.print("%{d} = \"MapOp.{s}\" ({s}%{d}) : ({s}) -> {s}\n", .{
                    node.id,
                    @tagName(map_link.op),
                    if (fused_x) "fused " else "",
                    map_link.x.id,
                    map_link.x.str,
                    node.str,
                });
            },
            .ZipOp => |zip_link| {
                var fused_a = false;
                var fused_b = false;
                for (node.fused.items) |fused_node| {
                    fused_a = fused_a or (zip_link.a == fused_node);
                    fused_b = fused_b or (zip_link.b == fused_node);
                }
                zip_link.a.show();
                zip_link.b.show();
                std.debug.print("%{d} = \"ZipOp.{s}\" ({s}%{d}, {s}%{d}) : ({s}, {s}) -> {s} \n", .{
                    node.id,
                    @tagName(zip_link.op),
                    if (fused_a) "fused" else "",
                    zip_link.a.id,
                    if (fused_b) "fused" else "",
                    zip_link.b.id,
                    zip_link.a.str,
                    zip_link.b.str,
                    node.str,
                });
            },
            .ReduceOp => |reduce_link| {
                var fused_x = false;
                for (node.fused.items) |fused_node| {
                    fused_x = fused_x or (reduce_link.x == fused_node);
                }

                reduce_link.x.show();
                std.debug.print("%{d} = \"ReduceOp.{s}\" ({s}%{d}) {{ dim = {?} }} : ({s}) -> {s}\n", .{
                    node.id,
                    @tagName(reduce_link.op),
                    if (fused_x) "fused " else "",
                    reduce_link.x.id,
                    reduce_link.dim,
                    reduce_link.x.str,
                    node.str,
                });
            },
            .TypeOp => |type_link| {
                type_link.x.show();
                std.debug.print("%{d} = \"TypeOp.{s}\" (%{d}) : ({s}) -> {s}\n", .{ node.id, @tagName(type_link.op), type_link.x.id, type_link.x.str, node.str });
            },
            .InitOp => |init_link| {
                std.debug.print("%{d} = \"InitOp.{s}\" () -> {s}\n", .{ node.id, @tagName(init_link.op), node.str });
            },
        }
    }
};

// Function for manually fusing two nodes so that node2's operation gets inlined in node1's operation that uses node2
// Naturally, this translated to node1 being directly dependent on (have a link to) node2
// This same function will be used for automatic fusion
fn fuse(node1: *Node, node2: *Node) !void {
    switch (node1.link) {
        .MapOp => |map_link| {
            if (map_link.x == node2) {
                try node1.fused.append(node2);
                return;
            }
        },
        .ZipOp => |zip_link| {
            if (zip_link.a == node2 or zip_link.b == node2) {
                try node1.fused.append(node2);
                return;
            }
        },
        .ReduceOp => |reduce_link| {
            if (reduce_link.x == node2) {
                try node1.fused.append(node2);
                return;
            }
        },
        else => @panic("Cannot fuse nodes as the operation is not fuseable"),
    }
    @panic("Cannot fuse nodes as node1 does not directly depend on node2");
}

pub fn show() void {
    std.debug.print("\n", .{});
    last_node.?.show();
}

// TODO: Need to add function to lower a node to the Loop representation
// Independent of automatic fusion but should be able to lower fused nodes correctly

test "manual_fuse" {
    const out = comptime blk: {
        const x1 = tensor.constant(.i32, 1);
        const x2 = x1.neg();
        const x3 = x2.exp2();
        const x4 = x3.recip();
        break :blk x4;
    };

    init();
    defer deinit();

    // Call trace on the output to build its computation graph
    out.trace();

    const x4_node = Graph.last_node.?;
    const x3_node = x4_node.link.MapOp.x;
    const x2_node = x3_node.link.MapOp.x;
    try fuse(x4_node, x3_node);
    try fuse(x3_node, x2_node);

    try std.testing.expect(x4_node.fused.items[0] == x3_node);
    try std.testing.expect(x3_node.fused.items[0] == x2_node);

    // Show the graph
    show();
}
