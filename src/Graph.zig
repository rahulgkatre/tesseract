const std = @import("std");
const ops = @import("ops.zig");
const tensor = @import("tensor.zig");
const utils = @import("utils.zig");
const Graph = @This();
const Program = @import("Program.zig");
const dtypes = @import("dtypes.zig");

var arena: std.heap.ArenaAllocator = undefined;
var allocator: std.mem.Allocator = undefined;
var cache: std.AutoHashMap(usize, usize) = undefined;
var ids: std.AutoHashMap(usize, usize) = undefined;
var nodes: std.AutoHashMap(usize, *Vertex) = undefined;
var entrypoint: ?*Vertex = null;

pub fn init() void {
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    allocator = arena.allocator();
    cache = std.AutoHashMap(usize, usize).init(allocator);
    ids = std.AutoHashMap(usize, usize).init(allocator);
    nodes = std.AutoHashMap(usize, *Vertex).init(allocator);
    Program.init();
}

pub fn deinit() void {
    arena.deinit();
    Program.deinit();
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
        op_info: union(ops.TypeOp) {
            AsStrided: struct {
                shape: []const usize,
                strides: []const usize,
            },
            AsType: dtypes.DType,
            View: []const usize,
        },
        x: *Vertex,
        fused_x: bool = true,
    };
    pub const InitOp = struct {
        op: ops.InitOp,
        // TODO: op_info for InitOp
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
    fused: bool = false,
    kernel_id: ?usize = null,
    cache_in_kernel: bool = false,

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

    fn nodeViz(node: *Vertex) std.mem.Allocator.Error!void {
        // To avoid printing the same thing multiple times use the hash table to check/mark as already printed
        if (viz_hash_table.get(node.id).? == true) {
            return;
        }
        try viz_hash_table.put(node.id, true);
        switch (node.edge) {
            inline else => |e| {
                std.debug.print("T{d}[label=\"T{d}\"shape=box];\n", .{ node.id, node.id });
                if (node.kernel_id != null and node.cache_in_kernel) {
                    std.debug.print("subgraph cluster{d}{{T{d}_{d}[label=\"T{d}_{d}\"shape=box];}}\n", .{ node.kernel_id.?, node.id, node.kernel_id.?, node.id, node.kernel_id.? });
                    std.debug.print("T{d}_{d}->T{d}[label=\" {s}{any}\"];\n", .{ node.id, node.kernel_id.?, node.id, @tagName(node.tensor.dtype), node.tensor.shape });
                    std.debug.print("{s}{d}->T{d}_{d}[label=\" {s}{any}\"];\n", .{ @tagName(e.op), node.id, node.id, node.kernel_id.?, @tagName(node.tensor.dtype), node.tensor.shape });
                } else {
                    std.debug.print("{s}{d}->T{d}[label=\" {s}{any}\"];\n", .{ @tagName(e.op), node.id, node.id, @tagName(node.tensor.dtype), node.tensor.shape });
                }
            },
        }
    }

    fn initOpViz(node: *Vertex, edge: Edge.InitOp) std.mem.Allocator.Error!void {
        if (node.kernel_id != null) {
            std.debug.print("subgraph cluster{d}{{{s}{d}[label=\"{s}\"];}}\n", .{ node.kernel_id.?, @tagName(edge.op), node.id, @tagName(edge.op) });
        } else {
            std.debug.print("{s}{d}[label=\"{s}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.op) });
        }
    }

    fn mapOpViz(node: *Vertex, edge: Edge.MapOp) std.mem.Allocator.Error!void {
        try edge.x.viz();
        if (node.kernel_id != null) {
            std.debug.print("subgraph cluster{d}{{{s}{d}[label=\"{s}\"];}}\n", .{ node.kernel_id.?, @tagName(edge.op), node.id, @tagName(edge.op) });
        } else {
            std.debug.print("{s}{d}[label=\"{s}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.op) });
        }
        if (edge.fused_x) {
            switch (edge.x.edge) {
                inline else => |x_edge| std.debug.print("{s}{d}", .{ @tagName(x_edge.op), edge.x.id }),
            }
        } else {
            try edge.x.nodeViz();
            if (node.kernel_id != null and node.kernel_id == edge.x.kernel_id and edge.x.cache_in_kernel) {
                std.debug.print("T{d}_{?}", .{ edge.x.id, edge.x.kernel_id });
            } else {
                std.debug.print("T{d}", .{edge.x.id});
            }
        }
        std.debug.print("->{s}{d}[label=\" {s}{any}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.x.tensor.dtype), edge.x.tensor.shape });
    }

    fn zipOpViz(node: *Vertex, edge: Edge.ZipOp) std.mem.Allocator.Error!void {
        try edge.a.viz();
        try edge.b.viz();
        if (node.kernel_id != null) {
            std.debug.print("subgraph cluster{d}{{{s}{d}[label=\"{s}\"];}}\n", .{ node.kernel_id.?, @tagName(edge.op), node.id, @tagName(edge.op) });
        } else {
            std.debug.print("{s}{d}[label=\"{s}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.op) });
        }
        if (edge.fused_a) {
            switch (edge.a.edge) {
                inline else => |a_edge| std.debug.print("{s}{d}", .{ @tagName(a_edge.op), edge.a.id }),
            }
        } else {
            try edge.a.nodeViz();
            if (node.kernel_id != null and node.kernel_id == edge.a.kernel_id and edge.a.cache_in_kernel) {
                std.debug.print("T{d}_{?}", .{ edge.a.id, edge.a.kernel_id });
            } else {
                std.debug.print("T{d}", .{edge.a.id});
            }
        }
        std.debug.print("->{s}{d}[label=\" A: {s}{any}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.a.tensor.dtype), edge.a.tensor.shape });
        if (edge.fused_b) {
            switch (edge.b.edge) {
                inline else => |b_edge| std.debug.print("{s}{d}", .{ @tagName(b_edge.op), edge.b.id }),
            }
        } else {
            try edge.b.nodeViz();
            if (node.kernel_id != null and node.kernel_id == edge.b.kernel_id and edge.b.cache_in_kernel) {
                std.debug.print("T{d}_{?}", .{ edge.b.id, edge.b.kernel_id });
            } else {
                std.debug.print("T{d}", .{edge.b.id});
            }
        }
        std.debug.print("->{s}{d}[label=\" B: {s}{any}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.b.tensor.dtype), edge.b.tensor.shape });
    }

    fn reduceOpViz(node: *Vertex, edge: Edge.ReduceOp) std.mem.Allocator.Error!void {
        try edge.x.viz();
        if (node.kernel_id != null) {
            std.debug.print("subgraph cluster{d}{{{s}{d}[label=\"{s}{any}\"];}}\n", .{ node.kernel_id.?, @tagName(edge.op), node.id, @tagName(edge.op), edge.dims });
        } else {
            std.debug.print("{s}{d}[label=\"{s}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.op) });
        }
        if (edge.fused_x) {
            switch (edge.x.edge) {
                inline else => |x_edge| std.debug.print("{s}{d}", .{ @tagName(x_edge.op), edge.x.id }),
            }
        } else {
            try edge.x.nodeViz();
            if (node.kernel_id != null and node.kernel_id == edge.x.kernel_id and edge.x.cache_in_kernel) {
                std.debug.print("T{d}_{?}", .{ edge.x.id, edge.x.kernel_id });
            } else {
                std.debug.print("T{d}", .{edge.x.id});
            }
        }
        std.debug.print("->{s}{d}[label=\" {s}{any}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.x.tensor.dtype), edge.x.tensor.shape });
    }

    fn typeOpViz(node: *Vertex, edge: Edge.TypeOp) std.mem.Allocator.Error!void {
        // TypeOps are fused by default
        try edge.x.viz();
        if (node.kernel_id != null) {
            switch (edge.op_info) {
                .AsType => |new_dtype| std.debug.print("subgraph cluster{d}{{{s}{d}[label=\"{s}{{{s}}}\"];}}\n", .{ node.kernel_id.?, @tagName(edge.op), node.id, @tagName(edge.op), @tagName(new_dtype) }),
                .AsStrided => |new_info| std.debug.print("subgraph cluster{d}{{{s}{d}[label=\"{s}{{shape:{any}, strides:{any}}}\"];}}\n", .{ node.kernel_id.?, @tagName(edge.op), node.id, @tagName(edge.op), new_info.shape, new_info.strides }),
                inline else => |new_info| std.debug.print("subgraph cluster{d}{{{s}{d}[label=\"{s}{any}\"];}}\n", .{ node.kernel_id.?, @tagName(edge.op), node.id, @tagName(edge.op), new_info }),
            }
        } else {
            switch (edge.op_info) {
                .AsType => |new_dtype| std.debug.print("{s}{d}[label=\"{s}{{{s}}}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.op), @tagName(new_dtype) }),
                .AsStrided => |new_info| std.debug.print("{s}{d}[label=\"{s}{{shape:{any}, strides:{any}}}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.op), new_info.shape, new_info.strides }),
                inline else => |new_info| std.debug.print("{s}{d}[label=\"{s}{any}\"];\n", .{ @tagName(edge.op), node.id, @tagName(edge.op), new_info }),
            }
        }
        switch (edge.x.edge) {
            inline else => |x_edge| std.debug.print("{s}{d}->{s}{d}[label=\" {s}{any}\"];\n", .{ @tagName(x_edge.op), edge.x.id, @tagName(edge.op), node.id, @tagName(edge.x.tensor.dtype), edge.x.tensor.shape }),
        }
    }

    pub fn viz(node: *Vertex) std.mem.Allocator.Error!void {
        if (viz_hash_table.contains(node.id)) {
            return;
        }
        try viz_hash_table.put(node.id, false);
        try switch (node.edge) {
            .InitOp => |edge| initOpViz(node, edge),
            .MapOp => |edge| mapOpViz(node, edge),
            .ZipOp => |edge| zipOpViz(node, edge),
            .ReduceOp => |edge| reduceOpViz(node, edge),
            .TypeOp => |edge| typeOpViz(node, edge),
        };
    }
};

pub fn viz() !void {
    std.testing.expect(entrypoint != null) catch @panic("Graph has not been created, remember to call Graph.trace() on the output tensor");
    viz_hash_table = std.AutoHashMap(usize, bool).init(allocator);
    defer {
        viz_hash_table.deinit();
        viz_hash_table = undefined;
    }
    std.debug.print("digraph G {{\ncompound=true;\n", .{});
    try entrypoint.?.viz();
    // Need to print the entrypoint separately because there is no other vertex
    // calling the entrypoint's print function
    try entrypoint.?.nodeViz();
    std.debug.print("}}\n", .{});
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
            if (edge.a.cache_in_kernel) {
                edge.fused_a = false;
            }
            unfuseIfCached(edge.a);
            if (edge.b.cache_in_kernel) {
                edge.fused_b = false;
            }
            unfuseIfCached(edge.b);
        },
        inline else => |*edge| {
            if (edge.x.cache_in_kernel) {
                edge.fused_x = false;
            }
            unfuseIfCached(edge.x);
        },
    }
}

fn greedyClusteringFusion(cluster_id: usize, node: *Vertex) usize {
    if (node.kernel_id != null or node.cache_in_kernel) {
        node.cache_in_kernel = true;
        return node.kernel_id.?;
    }
    switch (node.edge) {
        .MapOp => |*edge| {
            node.kernel_id = greedyClusteringFusion(cluster_id, edge.x);
            if (edge.x.kernel_id == node.kernel_id and !node.cache_in_kernel) {
                applyVerticalFusion(edge.x, node) catch unreachable;
            }
            return node.kernel_id.?;
        },
        .ZipOp => |*edge| {
            const b_kernel_id = greedyClusteringFusion(cluster_id, edge.a);
            node.kernel_id = greedyClusteringFusion(b_kernel_id, edge.b);
            if (edge.a.kernel_id == node.kernel_id) {
                applyVerticalFusion(edge.a, node) catch unreachable;
            }
            if (edge.b.kernel_id == node.kernel_id and !node.cache_in_kernel) {
                applyVerticalFusion(edge.b, node) catch unreachable;
            }
            return node.kernel_id.?;
        },
        .ReduceOp => |*edge| {
            node.kernel_id = greedyClusteringFusion(cluster_id, edge.x);
            if (edge.x.kernel_id == node.kernel_id and !node.cache_in_kernel) {
                applyVerticalFusion(edge.x, node) catch unreachable;
            }
            // Increment kernel id to prevent multiple reduces from being in the same kernel
            return node.kernel_id.? + 1;
        },
        .TypeOp => |*edge| {
            // TypeOps can always be fused into the preceding kernel even if the typeop follows a reduce
            // This is because it is either just index manipulation (and does not correspond to a loop)
            // or it is a cast which can be inlined during the accumulation of a reduction
            const kernel_id = greedyClusteringFusion(cluster_id, edge.x);
            // Sometimes the preceding operation might be an init which can happen in global scope (not in a kernel)
            // and it might not have a kernel id, in which case just use the returned kernel id
            node.kernel_id = edge.x.kernel_id orelse kernel_id;
            if (edge.x.kernel_id == node.kernel_id and !node.cache_in_kernel) {
                applyVerticalFusion(edge.x, node) catch unreachable;
            }
            return node.kernel_id.?;
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

    Graph.init();
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
    // std.debug.print("\n", .{});
    // Graph.viz();
}

test "greedy fusion" {
    const out = comptime blk: {
        const a = tensor.InferredStrides(.f32, .{ 1024, 2048 }).full(2);
        const b = tensor.InferredStrides(.f32, .{ 2048, 4096 }).full(3);
        break :blk a.matmul(b);
    };
    Graph.init();
    defer Graph.deinit();
    Graph.trace(out);
    Graph.applyGreedyFusion();
    std.debug.print("\n", .{});
    try Graph.viz();
}
