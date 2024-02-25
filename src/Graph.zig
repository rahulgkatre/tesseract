const std = @import("std");
const ops = @import("ops.zig");
const tensor = @import("tensor.zig");
const utils = @import("utils.zig");
const Graph = @This();
const dtypes = @import("dtypes.zig");

var arena: std.heap.ArenaAllocator = undefined;
var allocator: std.mem.Allocator = undefined;
var ids: std.AutoArrayHashMap(usize, usize) = undefined;
var nodes: std.AutoHashMap(usize, *Vertex) = undefined;
var reduction_groups: std.AutoHashMap(usize, bool) = undefined;
var entry_node: ?*Vertex = null;

pub fn entry() *Vertex {
    return entry_node orelse @panic("Graph has no entrypoint. Remember to call trace() on an output tensor pointer");
}

pub fn init(backing_allocator: std.mem.Allocator) void {
    arena = std.heap.ArenaAllocator.init(backing_allocator);
    allocator = arena.allocator();
    ids = std.AutoArrayHashMap(usize, usize).init(allocator);
    nodes = std.AutoHashMap(usize, *Vertex).init(allocator);
    reduction_groups = std.AutoHashMap(usize, bool).init(allocator);
}

pub fn deinit() void {
    arena.deinit();
}

/// Build the computation graph for a tensor.
/// Any new nodes are added to the global computation graph
/// by recursively calling each tensor's `traceFn` callback.
pub fn trace(tensor_ptr: anytype) void {
    switch (@typeInfo(@TypeOf(tensor_ptr))) {
        .Pointer => tensor_ptr.traceFn(tensor_ptr),
        else => @compileError("Must pass a tensor pointer"),
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
        op: ops.InitOp,
        value: ops.InitValue,
        fused_init: bool = true,
    };
    MapOp: MapOp,
    ZipOp: ZipOp,
    ReduceOp: ReduceOp,
    TypeOp: TypeOp,
    InitOp: InitOp,
};

var viz_hash_table: std.AutoHashMap(usize, bool) = undefined;
const TensorInfo = struct {
    ndims: u8,
    dtype: dtypes.DType,
    shape: []const usize,
    strides: []const usize,
};

pub const Vertex = struct {
    ptr: usize,
    edge: Edge,
    group_id: ?usize = null,
    cached: bool = false,

    // Tensor metadata
    tensor: TensorInfo,

    pub fn nodeId(node: *const Vertex) usize {
        return ids.get(node.ptr).?;
    }

    pub fn tensorId(node: *const Vertex) usize {
        return switch (node.edge) {
            .InitOp => node.nodeId(),
            .ZipOp => |edge| if (edge.fused_a and !edge.fused_b) edge.a.tensorId() else if (edge.fused_b and !edge.fused_a) edge.b.tensorId() else node.nodeId(),
            inline else => |edge| if (edge.fused_x) edge.x.tensorId() else node.nodeId(),
        };
    }

    fn nodeViz(node: *const Vertex, writer: anytype) std.mem.Allocator.Error!void {
        // To avoid printing the same thing multiple times use the hash table to check/mark as already printed
        if (viz_hash_table.get(node.nodeId()).? == true) {
            return;
        }
        try viz_hash_table.put(node.nodeId(), true);
        switch (node.edge) {
            inline else => |e| {
                writer.print("T{d}[label=\"T{d}\"shape=box];\n", .{ node.tensorId(), node.tensorId() });
                if (node.cached) {
                    writer.print("subgraph cluster{d}{{T{d}_{d}[label=\"T{d}_{d}\"shape=box];}}\n", .{ node.group_id.?, node.tensorId(), node.group_id.?, node.tensorId(), node.group_id.? });
                    writer.print("T{d}_{d}->T{d}[label=\" {s}{any}\"];\n", .{ node.tensorId(), node.group_id.?, node.tensorId(), @tagName(node.tensor.dtype), node.tensor.shape });
                    writer.print("{s}{d}->T{d}_{d}[label=\" {s}{any}\"];\n", .{ @tagName(e.op), node.nodeId(), node.tensorId(), node.group_id.?, @tagName(node.tensor.dtype), node.tensor.shape });
                } else {
                    writer.print("{s}{d}->T{d}[label=\" {s}{any}\"];\n", .{ @tagName(e.op), node.nodeId(), node.tensorId(), @tagName(node.tensor.dtype), node.tensor.shape });
                }
            },
        }
    }

    fn edgeViz(dst: *const Graph.Vertex, src: *const Graph.Vertex, op: anytype, fused: bool, writer: anytype) !void {
        if (fused) {
            switch (src.edge) {
                inline else => |src_edge| writer.print("{s}{d}", .{ @tagName(src_edge.op), src.nodeId() }),
            }
        } else {
            try src.nodeViz(writer);
            if (dst.group_id != null and src.group_id == dst.group_id and src.cached) {
                writer.print("T{d}_{?}", .{ src.tensorId(), src.group_id });
            } else {
                writer.print("T{d}", .{src.tensorId()});
            }
        }
        writer.print("->{s}{d}[label=\" {s}{any}\"];\n", .{ @tagName(op), dst.nodeId(), @tagName(src.tensor.dtype), src.tensor.shape });
    }

    fn initOpViz(node: *const Vertex, edge: Edge.InitOp, writer: anytype) std.mem.Allocator.Error!void {
        if (node.group_id != null) {
            writer.print("subgraph cluster{d}{{{s}{d}[label=\"{s}\"];}}\n", .{ node.group_id.?, @tagName(edge.op), node.nodeId(), @tagName(edge.op) });
        } else {
            writer.print("{s}{d}[label=\"{s}\"];\n", .{ @tagName(edge.op), node.nodeId(), @tagName(edge.op) });
        }
    }

    fn mapOpViz(node: *const Vertex, edge: Edge.MapOp, writer: anytype) std.mem.Allocator.Error!void {
        try edge.x.viz(writer);
        if (node.group_id != null) {
            writer.print("subgraph cluster{d}{{{s}{d}[label=\"{s}\"];}}\n", .{ node.group_id.?, @tagName(edge.op), node.nodeId(), @tagName(edge.op) });
        } else {
            writer.print("{s}{d}[label=\"{s}\"];\n", .{ @tagName(edge.op), node.nodeId(), @tagName(edge.op) });
        }
        try node.edgeViz(edge.x, edge.op, edge.fused_x, writer);
    }

    fn zipOpViz(node: *const Vertex, edge: Edge.ZipOp, writer: anytype) std.mem.Allocator.Error!void {
        try edge.a.viz(writer);
        try edge.b.viz(writer);
        if (node.group_id != null) {
            writer.print("subgraph cluster{d}{{{s}{d}[label=\"{s}\"];}}\n", .{ node.group_id.?, @tagName(edge.op), node.nodeId(), @tagName(edge.op) });
        } else {
            writer.print("{s}{d}[label=\"{s}\"];\n", .{ @tagName(edge.op), node.nodeId(), @tagName(edge.op) });
        }
        try node.edgeViz(edge.a, edge.op, edge.fused_a, writer);
        try node.edgeViz(edge.b, edge.op, edge.fused_b, writer);
    }

    fn reduceOpViz(node: *const Vertex, edge: Edge.ReduceOp, writer: anytype) std.mem.Allocator.Error!void {
        try edge.x.viz(writer);
        if (node.group_id != null) {
            writer.print("subgraph cluster{d}{{{s}{d}[label=\"{s}{any}\"];}}\n", .{ node.group_id.?, @tagName(edge.op), node.nodeId(), @tagName(edge.op), edge.dims });
        } else {
            writer.print("{s}{d}[label=\"{s}\"];\n", .{ @tagName(edge.op), node.nodeId(), @tagName(edge.op) });
        }
        try node.edgeViz(edge.x, edge.op, edge.fused_x, writer);
    }

    fn typeOpViz(node: *const Vertex, edge: Edge.TypeOp, writer: anytype) std.mem.Allocator.Error!void {
        try edge.x.viz(writer);
        if (node.group_id != null) {
            switch (edge.op) {
                .AsType => writer.print("subgraph cluster{d}{{{s}{d}[label=\"{s}{{{s}}}\"];}}\n", .{ node.group_id.?, @tagName(edge.op), node.nodeId(), @tagName(edge.op), @tagName(node.tensor.dtype) }),
                .View, .Broadcast => writer.print("subgraph cluster{d}{{{s}{d}[label=\"{s}{{shape:{any}}}\"];}}\n", .{ node.group_id.?, @tagName(edge.op), node.nodeId(), @tagName(edge.op), node.tensor.shape }),
                .AsStrided => writer.print("subgraph cluster{d}{{{s}{d}[label=\"{s}{{shape:{any}, strides:{any}}}\"];}}\n", .{ node.group_id.?, @tagName(edge.op), node.nodeId(), @tagName(edge.op), node.tensor.shape, node.tensor.strides }),
            }
        } else {
            switch (edge.op) {
                .AsType => writer.print("{s}{d}[label=\"{s}{{{s}}}\"];\n", .{ @tagName(edge.op), node.nodeId(), @tagName(edge.op), @tagName(node.tensor.dtype) }),
                .View, .Broadcast => writer.print("{s}{d}[label=\"{s}{{shape:{any}}}\"];\n", .{ @tagName(edge.op), node.nodeId(), @tagName(edge.op), node.tensor.shape }),
                .AsStrided => writer.print("{s}{d}[label=\"{s}{{shape:{any}, strides:{any}}}\"];\n", .{ @tagName(edge.op), node.nodeId(), @tagName(edge.op), node.tensor.shape, node.tensor.strides }),
            }
        }
        try node.edgeViz(edge.x, edge.op, edge.fused_x, writer);
    }

    pub fn viz(node: *const Vertex, writer: anytype) std.mem.Allocator.Error!void {
        if (viz_hash_table.contains(node.nodeId())) {
            return;
        }
        try viz_hash_table.putNoClobber(node.nodeId(), false);
        try switch (node.edge) {
            .InitOp => |edge| node.initOpViz(edge, writer),
            .MapOp => |edge| node.mapOpViz(edge, writer),
            .ZipOp => |edge| node.zipOpViz(edge, writer),
            .ReduceOp => |edge| node.reduceOpViz(edge, writer),
            .TypeOp => |edge| node.typeOpViz(edge, writer),
        };
    }
};

pub fn vertex(ptr: anytype, edge: Edge, comptime Tensor: type) !void {
    const key = @intFromPtr(ptr);
    if (!ids.contains(key)) {
        const id = ids.count();
        try ids.putNoClobber(key, id);
        const v = try allocator.create(Vertex);
        v.* = .{
            .ptr = @intFromPtr(ptr),
            .edge = edge,
            .tensor = .{
                .ndims = Tensor.ndims,
                .dtype = Tensor.dtype,
                .shape = Tensor.shape[0..],
                .strides = Tensor.strides[0..],
            },
        };
        entry_node = v;
        try nodes.putNoClobber(id, v);
    }
}

pub fn vertexOf(tensor_ptr: anytype) *Vertex {
    return nodes.get(ids.get(@intFromPtr(tensor_ptr)).?).?;
}

pub fn viz(writer: anytype) !void {
    viz_hash_table = std.AutoHashMap(usize, bool).init(allocator);
    defer {
        viz_hash_table.deinit();
        viz_hash_table = undefined;
    }
    writer.print("digraph G {{\ncompound=true;\n", .{});
    try entry().viz(writer);
    // Need to print the entrypoint separately because there is no other vertex
    // calling the entrypoint's print function
    try entry().nodeViz(writer);
    writer.print("}}\n", .{});
}

pub const Fusion = struct {
    const FusionError = error{
        ParentReduce,
        ParentInit,
        NotParentChild,
    };

    pub fn verticalFusion(parent: *Vertex, child: *Vertex) FusionError!void {
        const parent_group_contains_reduction = if (parent.group_id) |group_id| reduction_groups.get(group_id) orelse false else false;
        switch (child.edge) {
            .InitOp => |*edge| {
                switch (edge.op) {
                    // Only fuse an init op if it is a constant value fill
                    // This is trival to inline, but others may not be
                    .Full => edge.fused_init = true,
                    else => return FusionError.ParentInit,
                }
            },
            .ZipOp => |*edge| {
                if (edge.a.tensorId() != parent.tensorId() and edge.b.tensorId() != parent.tensorId()) {
                    return FusionError.NotParentChild;
                }
                if (edge.a.tensorId() == parent.tensorId()) {
                    if (parent_group_contains_reduction) {
                        return FusionError.ParentReduce;
                    }
                    edge.fused_a = true;
                }
                if (edge.b.tensorId() == parent.tensorId()) {
                    if (parent_group_contains_reduction) {
                        return FusionError.ParentReduce;
                    }
                    edge.fused_b = true;
                }
            },
            .TypeOp => |*edge| {
                // Fuse a TypeOp even when the previous op is a reduce op
                // The only type op that has any loop is a cast, and that is trivial enough
                // to warrant not needing an extra loop anyways
                if (edge.x.tensorId() != parent.tensorId()) {
                    return FusionError.NotParentChild;
                }
                edge.fused_x = true;
            },
            .MapOp => |*edge| {
                if (edge.x.tensorId() != parent.tensorId()) {
                    return FusionError.NotParentChild;
                }
                if (parent_group_contains_reduction) {
                    return FusionError.ParentReduce;
                }
                edge.fused_x = true;
            },
            .ReduceOp => |*edge| {
                if (edge.x.tensorId() != parent.tensorId()) {
                    return FusionError.NotParentChild;
                }
                if (parent_group_contains_reduction) {
                    return FusionError.ParentReduce;
                }
                edge.fused_x = true;
            },
        }
    }

    fn greedyFusionHelper(group_id: usize, node: *Vertex) usize {
        if (node.group_id == group_id) {
            // A node can be cached in the kernel if it is being reused by 1 or more dependents
            // in the same tensor
            // Could also make this a counter to determine the number of times a tensor is reused
            // to see if just repeatedly calculating it again is faster than reading it out of memory
            node.cached = true;
        }
        if (node.cached) {
            if (node.group_id != group_id) {
                node.cached = false;
            }
            return node.group_id.?;
        }
        switch (node.edge) {
            .MapOp => |*edge| {
                node.group_id = greedyFusionHelper(group_id, edge.x);
                if (edge.x.group_id == node.group_id and !edge.x.cached) {
                    verticalFusion(edge.x, node) catch {
                        // If we get a fusion error, move the current node to the next group
                        node.group_id = node.group_id.? + 1;
                    };
                }
                while (reduction_groups.get(node.group_id.?) orelse false) {
                    node.group_id = node.group_id.? + 1;
                }
            },
            .ZipOp => |*edge| {
                // Greedy fusion helper returns the next group id so here it is passed from a -> b -> current
                node.group_id = greedyFusionHelper(greedyFusionHelper(group_id, edge.a), edge.b);
                if (edge.a.group_id == node.group_id and !edge.a.cached) {
                    verticalFusion(edge.a, node) catch {
                        node.group_id = node.group_id.? + 1;
                    };
                }
                if (edge.b.group_id == node.group_id and !edge.b.cached) {
                    verticalFusion(edge.b, node) catch {
                        node.group_id = node.group_id.? + 1;
                    };
                }
                while (reduction_groups.get(node.group_id.?) orelse false) {
                    node.group_id = node.group_id.? + 1;
                }
            },
            .ReduceOp => |*edge| {
                node.group_id = greedyFusionHelper(group_id, edge.x);
                if (edge.x.group_id == node.group_id and !edge.x.cached) {
                    verticalFusion(edge.x, node) catch {
                        node.group_id = node.group_id.? + 1;
                    };
                }
                while (reduction_groups.get(node.group_id.?) orelse false) {
                    node.group_id = node.group_id.? + 1;
                }
                reduction_groups.putNoClobber(node.group_id.?, true) catch unreachable;
            },
            .TypeOp => |*edge| {
                // TypeOps can always be fused into the preceding kernel even if the typeop follows a reduce
                // This is because it is either just index manipulation (and does not correspond to a loop)
                // or it is a cast which can be inlined during the accumulation of a reduction
                node.group_id = greedyFusionHelper(group_id, edge.x);
                if (edge.x.group_id == node.group_id and !edge.x.cached) {
                    verticalFusion(edge.x, node) catch {
                        node.group_id = node.group_id.? + 1;
                    };
                }
            },
            // Init will happen outside a kernel unless it is a full init
            .InitOp => |*edge| {
                if (edge.op == .Full) {
                    node.group_id = group_id;
                }
                return group_id;
            },
        }
        return node.group_id.?;
    }

    /// Traverse the graph and group nodes into clusters (kernels/functions)
    /// Each cluster can have at most one reduce op, but any amount of other ops
    /// The reduce op will be the last op unless it is followed by a type op
    pub fn greedyFusion() !void {
        _ = greedyFusionHelper(0, entry());
    }
};

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
    Graph.trace(&sm);

    const t9 = Graph.entry();
    const t8 = t9.edge.ZipOp.b;
    try Fusion.verticalFusion(t8, t9);
    const t7 = t8.edge.TypeOp.x.edge.MapOp.x;
    const t6 = t7.edge.ReduceOp.x;
    try Fusion.verticalFusion(t6, t7);
    const t5 = t6.edge.MapOp.x;
    try Fusion.verticalFusion(t5, t6);

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
    Graph.trace(&out);
    // Graph.Fusion.applyGreedyFusion();
    // std.debug.print("\n", .{});
    // try Graph.viz(std.debug);
}
