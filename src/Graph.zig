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
pub var entrypoint: ?*Vertex = null;

pub fn init(backing_allocator: std.mem.Allocator) void {
    arena = std.heap.ArenaAllocator.init(backing_allocator);
    allocator = arena.allocator();
    cache = std.AutoHashMap(usize, usize).init(allocator);
    ids = std.AutoHashMap(usize, usize).init(allocator);
    nodes = std.AutoHashMap(usize, *Vertex).init(allocator);
}

pub fn deinit() void {
    arena.deinit();
}

/// Build the computation graph for a tensor.
/// Any new nodes are added to the global computation graph
/// by recursively calling each tensor's `traceFn` callback.
pub fn trace(comptime _tensor: anytype) void {
    switch (@typeInfo(@TypeOf(_tensor))) {
        .Pointer => _tensor.traceFn(_tensor),
        else => _tensor.traceFn(&_tensor),
    }
}

/// Edges in the computation graph (child -> parent).
///
/// Stores a pointer to a node (parent)
/// and the opperation applied to it to produce the child.
/// Also contains operation inlining/fusing info.
pub const Edge = union(ops.GraphOps) {
    pub const MapOp = struct {
        op: ops.MapOp,
        x: *Vertex,
        fused_x: bool = false,
    };
    pub const ZipOp = struct {
        op: ops.ZipOp,
        a: *Vertex,
        b: *Vertex,
        fused_a: bool = false,
        fused_b: bool = false,
    };
    pub const ReduceOp = struct {
        op: ops.ReduceOp,
        x: *Vertex,
        dims: []const bool,
        fused_x: bool = false,
    };
    pub const TypeOp = struct {
        op: ops.TypeOp,
        x: *Vertex,
        fused_x: bool = true,
    };
    pub const InitOp = struct {
        pub const InitValue = union(ops.InitOp) {
            Input: void,
            Full: []const u8,
            Rand: void,
            Range: struct {
                start: []const u8,
                stop: []const u8,
            },
        };
        op: ops.InitOp,
        value: InitValue,
    };
    MapOp: MapOp,
    ZipOp: ZipOp,
    ReduceOp: ReduceOp,
    TypeOp: TypeOp,
    InitOp: InitOp,
};

var viz_hash_table: std.AutoHashMap(usize, bool) = undefined;
pub const Vertex = struct {
    id: usize,
    edge: Edge,
    group_id: ?usize = null,
    cached: bool = false,

    // Tensor metadata
    tensor: struct {
        ndims: u8,
        dtype: dtypes.DType,
        shape: []const usize,
        strides: []const usize,
    },

    fn register(ptr: anytype, edge: Edge, comptime Tensor: type) !void {
        const key = @intFromPtr(ptr);
        if (!ids.contains(key)) {
            const id = ids.count();
            try ids.put(key, id);
            const vertex = try allocator.create(Vertex);
            vertex.* = .{
                .id = id,
                .edge = edge,
                .tensor = .{
                    .ndims = Tensor.ndims,
                    .dtype = Tensor.dtype,
                    .shape = Tensor.shape[0..],
                    .strides = Tensor.strides[0..],
                },
            };
            entrypoint = vertex;
            try nodes.put(id, vertex);
        }
    }

    pub fn new(ptr: anytype, edge: Edge, comptime Tensor: type) void {
        register(ptr, edge, Tensor) catch @panic("Out of memory");
    }

    pub fn get(ptr: anytype) *Vertex {
        return nodes.get(ids.get(@intFromPtr(ptr)).?).?;
    }

    fn nodeViz(node: *Vertex, writer: anytype) std.mem.Allocator.Error!void {
        // To avoid printing the same thing multiple times use the hash table to check/mark as already printed
        if (viz_hash_table.get(node.id).? == true) {
            return;
        }
        try viz_hash_table.put(node.id, true);
        switch (node.edge) {
            inline else => |e| {
                writer.print("T{d}[label=\"T{d}\"shape=box];\n", .{ node.id, node.id });
                if (node.group_id != null and node.cached) {
                    writer.print("subgraph cluster{d}{{T{d}_{d}[label=\"T{d}_{d}\"shape=box];}}\n", .{ node.group_id.?, node.id, node.group_id.?, node.id, node.group_id.? });
                    writer.print("T{d}_{d}->T{d}[label=\" {s}{any}\"];\n", .{ node.id, node.group_id.?, node.id, @tagName(node.tensor.dtype), node.tensor.shape });
                    writer.print("{s}{d}->T{d}_{d}[label=\" {s}{any}\"];\n", .{ @tagName(e.op), node.id, node.id, node.group_id.?, @tagName(node.tensor.dtype), node.tensor.shape });
                } else {
                    writer.print("{s}{d}->T{d}[label=\" {s}{any}\"];\n", .{ @tagName(e.op), node.id, node.id, @tagName(node.tensor.dtype), node.tensor.shape });
                }
            },
        }
    }

    fn initOpViz(node: *Vertex, edge: Edge.InitOp, writer: anytype) std.mem.Allocator.Error!void {
        if (node.group_id != null) {
            writer.print("subgraph cluster{d}{{{s}{d}[label=\"{s}\"];}}\n", .{ node.group_id.?, @tagName(edge.op), node.id, @tagName(edge.op) });
        } else {
            writer.print("{s}{d}[label=\"{s}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.op) });
        }
    }

    fn mapOpViz(node: *Vertex, edge: Edge.MapOp, writer: anytype) std.mem.Allocator.Error!void {
        try edge.x.viz(writer);
        if (node.group_id != null) {
            writer.print("subgraph cluster{d}{{{s}{d}[label=\"{s}\"];}}\n", .{ node.group_id.?, @tagName(edge.op), node.id, @tagName(edge.op) });
        } else {
            writer.print("{s}{d}[label=\"{s}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.op) });
        }
        if (edge.fused_x) {
            switch (edge.x.edge) {
                inline else => |x_edge| writer.print("{s}{d}", .{ @tagName(x_edge.op), edge.x.id }),
            }
        } else {
            try edge.x.nodeViz(writer);
            if (node.group_id != null and node.group_id == edge.x.group_id and edge.x.cached) {
                writer.print("T{d}_{?}", .{ edge.x.id, edge.x.group_id });
            } else {
                writer.print("T{d}", .{edge.x.id});
            }
        }
        writer.print("->{s}{d}[label=\" {s}{any}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.x.tensor.dtype), edge.x.tensor.shape });
    }

    fn zipOpViz(node: *Vertex, edge: Edge.ZipOp, writer: anytype) std.mem.Allocator.Error!void {
        try edge.a.viz(writer);
        try edge.b.viz(writer);
        if (node.group_id != null) {
            writer.print("subgraph cluster{d}{{{s}{d}[label=\"{s}\"];}}\n", .{ node.group_id.?, @tagName(edge.op), node.id, @tagName(edge.op) });
        } else {
            writer.print("{s}{d}[label=\"{s}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.op) });
        }
        if (edge.fused_a) {
            switch (edge.a.edge) {
                inline else => |a_edge| writer.print("{s}{d}", .{ @tagName(a_edge.op), edge.a.id }),
            }
        } else {
            try edge.a.nodeViz(writer);
            if (node.group_id != null and node.group_id == edge.a.group_id and edge.a.cached) {
                writer.print("T{d}_{?}", .{ edge.a.id, edge.a.group_id });
            } else {
                writer.print("T{d}", .{edge.a.id});
            }
        }
        writer.print("->{s}{d}[label=\" A: {s}{any}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.a.tensor.dtype), edge.a.tensor.shape });
        if (edge.fused_b) {
            switch (edge.b.edge) {
                inline else => |b_edge| writer.print("{s}{d}", .{ @tagName(b_edge.op), edge.b.id }),
            }
        } else {
            try edge.b.nodeViz(writer);
            if (node.group_id != null and node.group_id == edge.b.group_id and edge.b.cached) {
                writer.print("T{d}_{?}", .{ edge.b.id, edge.b.group_id });
            } else {
                writer.print("T{d}", .{edge.b.id});
            }
        }
        writer.print("->{s}{d}[label=\" B: {s}{any}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.b.tensor.dtype), edge.b.tensor.shape });
    }

    fn reduceOpViz(node: *Vertex, edge: Edge.ReduceOp, writer: anytype) std.mem.Allocator.Error!void {
        try edge.x.viz(writer);
        if (node.group_id != null) {
            writer.print("subgraph cluster{d}{{{s}{d}[label=\"{s}{any}\"];}}\n", .{ node.group_id.?, @tagName(edge.op), node.id, @tagName(edge.op), edge.dims });
        } else {
            writer.print("{s}{d}[label=\"{s}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.op) });
        }
        if (edge.fused_x) {
            switch (edge.x.edge) {
                inline else => |x_edge| writer.print("{s}{d}", .{ @tagName(x_edge.op), edge.x.id }),
            }
        } else {
            try edge.x.nodeViz(writer);
            if (node.group_id != null and node.group_id == edge.x.group_id and edge.x.cached) {
                writer.print("T{d}_{?}", .{ edge.x.id, edge.x.group_id });
            } else {
                writer.print("T{d}", .{edge.x.id});
            }
        }
        writer.print("->{s}{d}[label=\" {s}{any}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.x.tensor.dtype), edge.x.tensor.shape });
    }

    fn typeOpViz(node: *Vertex, edge: Edge.TypeOp, writer: anytype) std.mem.Allocator.Error!void {
        // TypeOps are fused by default
        try edge.x.viz(writer);
        if (node.group_id != null) {
            switch (edge.op) {
                .AsType => writer.print("subgraph cluster{d}{{{s}{d}[label=\"{s}{{{s}}}\"];}}\n", .{ node.group_id.?, @tagName(edge.op), node.id, @tagName(edge.op), @tagName(node.tensor.dtype) }),
                .View => writer.print("subgraph cluster{d}{{{s}{d}[label=\"{s}{{shape:{any}}}\"];}}\n", .{ node.group_id.?, @tagName(edge.op), node.id, @tagName(edge.op), node.tensor.shape }),
                .AsStrided => writer.print("subgraph cluster{d}{{{s}{d}[label=\"{s}{{shape:{any}, strides:{any}}}\"];}}\n", .{ node.group_id.?, @tagName(edge.op), node.id, @tagName(edge.op), node.tensor.shape, node.tensor.strides }),
            }
        } else {
            switch (edge.op) {
                .AsType => writer.print("{s}{d}[label=\"{s}{{{s}}}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.op), @tagName(node.tensor.dtype) }),
                .View => writer.print("{s}{d}[label=\"{s}{{shape:{any}}}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.op), node.tensor.shape }),
                .AsStrided => writer.print("{s}{d}[label=\"{s}{{shape:{any}, strides:{any}}}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.op), node.tensor.shape, node.tensor.strides }),
            }
        }
        switch (edge.x.edge) {
            inline else => |x_edge| writer.print("{s}{d}->{s}{d}[label=\" {s}{any}\"];\n", .{ @tagName(x_edge.op), edge.x.id, @tagName(edge.op), node.id, @tagName(edge.x.tensor.dtype), edge.x.tensor.shape }),
        }
    }

    pub fn viz(node: *Vertex, writer: anytype) std.mem.Allocator.Error!void {
        if (viz_hash_table.contains(node.id)) {
            return;
        }
        try viz_hash_table.put(node.id, false);
        try switch (node.edge) {
            .InitOp => |edge| node.initOpViz(edge, writer),
            .MapOp => |edge| node.mapOpViz(edge, writer),
            .ZipOp => |edge| node.zipOpViz(edge, writer),
            .ReduceOp => |edge| node.reduceOpViz(edge, writer),
            .TypeOp => |edge| node.typeOpViz(edge, writer),
        };
    }
};

pub fn viz(writer: anytype) !void {
    std.testing.expect(entrypoint != null) catch @panic("Graph has not been created, remember to call Graph.trace() on the output tensor");
    viz_hash_table = std.AutoHashMap(usize, bool).init(allocator);
    defer {
        viz_hash_table.deinit();
        viz_hash_table = undefined;
    }
    writer.print("digraph G {{\ncompound=true;\n", .{});
    try entrypoint.?.viz(writer);
    // Need to print the entrypoint separately because there is no other vertex
    // calling the entrypoint's print function
    try entrypoint.?.nodeViz(writer);
    writer.print("}}\n", .{});
}

const FusionError = error{
    ParentReduce,
    ParentInit,
    NotParentChild,
};

fn fusionError(node1: *Vertex, node2: *Vertex, err: FusionError) FusionError {
    switch (err) {
        FusionError.ParentReduce => std.log.err(
            \\
            \\Cannot fuse these two nodes as child depends on a parent which is the result of a reduction
            \\
            \\parent: {any}
            \\
            \\child: {any}
            \\
        , .{ node1, node2 }),
        FusionError.ParentInit => std.log.err(
            \\
            \\Cannot fuse these two nodes as child depends on initialization of parent
            \\
            \\parent: {any}
            \\
            \\child: {any}
            \\
        , .{ node1, node2 }),
        FusionError.NotParentChild => std.log.err(
            \\
            \\Cannot fuse these two nodes as node2 is not a child of parent node1
            \\
            \\node1: {any}
            \\
            \\node2: {any}
            \\
        , .{ node1, node2 }),
    }
    return err;
}

pub fn applyVerticalFusion(parent: *Vertex, child: *Vertex) FusionError!void {
    switch (child.edge) {
        .InitOp => return FusionError.ParentInit,
        .ZipOp => |*edge| {
            if (edge.a.id != parent.id and edge.b.id != parent.id) {
                return fusionError(parent, child, FusionError.NotParentChild);
            }
            if (edge.a.id == parent.id) {
                if (edge.a.edge == .ReduceOp) {
                    return fusionError(parent, child, FusionError.ParentReduce);
                }
                edge.fused_a = true;
            }
            if (edge.b.id == parent.id) {
                if (edge.b.edge == .ReduceOp) {
                    return fusionError(parent, child, FusionError.ParentReduce);
                }
                edge.fused_b = true;
            }
        },
        .TypeOp => |*edge| {
            if (edge.x.id == parent.id) {
                edge.fused_x = true;
            } else {
                return fusionError(parent, child, FusionError.NotParentChild);
            }
        },
        inline else => |*edge| {
            if (edge.x.id == parent.id) {
                if (edge.x.edge == .ReduceOp) {
                    return fusionError(parent, child, FusionError.ParentReduce);
                }
                edge.fused_x = true;
            } else {
                return fusionError(parent, child, FusionError.NotParentChild);
            }
        },
    }
}

pub fn unfuseIfCached(node: *Vertex) void {
    switch (node.edge) {
        .InitOp => {},
        .ZipOp => |*edge| {
            if (edge.a.cached) {
                edge.fused_a = false;
            }
            unfuseIfCached(edge.a);
            if (edge.b.cached) {
                edge.fused_b = false;
            }
            unfuseIfCached(edge.b);
        },
        inline else => |*edge| {
            if (edge.x.cached) {
                edge.fused_x = false;
            }
            unfuseIfCached(edge.x);
        },
    }
}

fn greedyClusteringFusion(cluster_id: usize, node: *Vertex) usize {
    if (node.group_id != null or node.cached) {
        node.cached = true;
        return node.group_id.?;
    }
    switch (node.edge) {
        .MapOp => |*edge| {
            node.group_id = greedyClusteringFusion(cluster_id, edge.x);
            if (edge.x.group_id == node.group_id and !node.cached) {
                applyVerticalFusion(edge.x, node) catch unreachable;
            }
            return node.group_id.?;
        },
        .ZipOp => |*edge| {
            const b_kernel_id = greedyClusteringFusion(cluster_id, edge.a);
            node.group_id = greedyClusteringFusion(b_kernel_id, edge.b);
            if (edge.a.group_id == node.group_id) {
                applyVerticalFusion(edge.a, node) catch unreachable;
            }
            if (edge.b.group_id == node.group_id and !node.cached) {
                applyVerticalFusion(edge.b, node) catch unreachable;
            }
            return node.group_id.?;
        },
        .ReduceOp => |*edge| {
            node.group_id = greedyClusteringFusion(cluster_id, edge.x);
            if (edge.x.group_id == node.group_id and !node.cached) {
                applyVerticalFusion(edge.x, node) catch unreachable;
            }
            // Increment kernel id to prevent multiple reduces from being in the same kernel
            return node.group_id.? + 1;
        },
        .TypeOp => |*edge| {
            // TypeOps can always be fused into the preceding kernel even if the typeop follows a reduce
            // This is because it is either just index manipulation (and does not correspond to a loop)
            // or it is a cast which can be inlined during the accumulation of a reduction
            const group_id = greedyClusteringFusion(cluster_id, edge.x);
            // Sometimes the preceding operation might be an init which can happen in global scope (not in a kernel)
            // and it might not have a kernel id, in which case just use the returned kernel id
            node.group_id = edge.x.group_id orelse group_id;
            if (edge.x.group_id == node.group_id and !node.cached) {
                applyVerticalFusion(edge.x, node) catch unreachable;
            }
            return node.group_id.?;
        },
        .InitOp => return cluster_id,
    }
}

/// Traverse the graph and group nodes into clusters (kernels)
/// Each cluster can have at most one reduce op, but any amount of other ops
/// The reduce op will be the last op unless it is followed by a type op
pub fn applyGreedyFusion() void {
    _ = greedyClusteringFusion(0, entrypoint.?);
    unfuseIfCached(entrypoint.?);
}

fn softmax(x: anytype, comptime dim: u8) @TypeOf(x) {
    const max = x.max(null);
    const x_minus_max = x.sub(max);
    const exp = x_minus_max.exp();
    const sumexp = exp.sum(dim);
    const sm = x_minus_max.div(sumexp);
    return sm;
}

test "manual vertical fusion" {
    const x = comptime tensor.InferredStrides(.f32, .{ 2, 16 }).full(0);
    const sm = comptime softmax(x, 1);

    Graph.init(std.testing.allocator);
    defer Graph.deinit();
    Graph.trace(sm);

    const t9 = Graph.entrypoint.?;
    const t8 = t9.edge.ZipOp.b;
    try applyVerticalFusion(t8, t9);
    const t7 = t8.edge.MapOp.x;
    const t6 = t7.edge.ReduceOp.x;
    try applyVerticalFusion(t6, t7);
    const t5 = t6.edge.MapOp.x;
    try applyVerticalFusion(t5, t6);

    // const t3 = t9.edge.ZipOp.a;
    // try fuse(t3, t9);
    // try fuse(t3, t5);
    // const t2 = t3.edge.ZipOp.b;
    // try fuse(t2, t3);
    // const t1 = t2.edge.MapOp.x;
    // try fuse(t1, t2);
    // writer.print("\n", .{});
    // Graph.viz(std.debug);
}

test "greedy fusion" {
    const out = comptime blk: {
        const a = tensor.InferredStrides(.f32, .{ 1024, 2048 }).full(2);
        const b = tensor.InferredStrides(.f32, .{ 2048, 4096 }).full(3);
        break :blk a.matmul(b);
    };
    Graph.init(std.testing.allocator);
    defer Graph.deinit();
    Graph.trace(out);
    Graph.applyGreedyFusion();
    // std.debug.print("\n", .{});
    // try Graph.viz(std.debug);
}
