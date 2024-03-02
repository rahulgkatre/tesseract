const std = @import("std");
const ops = @import("ops.zig");
const tensor = @import("tensor.zig");
const utils = @import("utils.zig");
const Graph = @This();
const dtypes = @import("dtypes.zig");

var arena: std.heap.ArenaAllocator = undefined;
var allocator: std.mem.Allocator = undefined;
var tensor_nodes: std.AutoHashMap(usize, *TensorNode) = undefined;
var preceding_op_nodes: std.AutoHashMap(*const TensorNode, OpNode) = undefined;
var reduction_groups: std.AutoHashMap(usize, bool) = undefined;
var entry_node: ?*TensorNode = null;

pub fn entry() *TensorNode {
    return entry_node orelse @panic("Graph has no entrypoint. Remember to call trace() on an output tensor pointer");
}

pub fn init(backing_allocator: std.mem.Allocator) void {
    arena = std.heap.ArenaAllocator.init(backing_allocator);
    allocator = arena.allocator();
    tensor_nodes = std.AutoHashMap(usize, *TensorNode).init(allocator);
    preceding_op_nodes = std.AutoHashMap(*const TensorNode, OpNode).init(allocator);
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

pub fn operation(comptime op: ops.GraphOp, input: anytype, output: anytype, comptime args: anytype) void {
    if (!tensor_nodes.contains(@intFromPtr(output))) {
        const op_node: OpNode = switch (op) {
            .MapOp => .{ .MapOp = .{
                .op = op.MapOp,
                .x = .{ .node = TensorNode.of(input) },
                .out = TensorNode.of(output),
                .op_node_label = std.fmt.comptimePrint("{s}", .{@tagName(op)}),
            } },
            .ZipOp => .{ .ZipOp = .{
                .op = op.ZipOp,
                .a = .{ .node = TensorNode.of(input.a) },
                .b = .{ .node = TensorNode.of(input.b) },
                .out = TensorNode.of(output),
                .op_node_label = std.fmt.comptimePrint("{s}", .{@tagName(op)}),
            } },
            .ReduceOp => .{ .ReduceOp = .{
                .op = op.ReduceOp,
                .x = .{ .node = TensorNode.of(input) },
                .out = TensorNode.of(output),
                .dims = args.dims,
                .op_node_label = std.fmt.comptimePrint("{s}{any}", .{ @tagName(op), args.dims }),
            } },
            .TypeOp => .{ .TypeOp = .{
                .op = op.TypeOp,
                .x = .{ .node = TensorNode.of(input) },
                .op_node_label = switch (op.TypeOp) {
                    .AsType => std.fmt.comptimePrint("{s}{any}", .{ @tagName(op), @TypeOf(output.*).dtype }),
                    .View, .Broadcast => std.fmt.comptimePrint("{s}{any}", .{ @tagName(op), @TypeOf(output.*).shape }),
                    .AsStrided => std.fmt.comptimePrint("{s}{{{any},{any}}}", .{ @tagName(op), @TypeOf(output.*).shape, @TypeOf(output.*).strides }),
                },
                .out = TensorNode.of(output),
            } },
            .InitOp => .{ .InitOp = .{
                .op = op.InitOp,
                .value = args,
                .out = TensorNode.of(output),
                .op_node_label = std.fmt.comptimePrint("{s}", .{@tagName(op)}),
            } },
        };
        preceding_op_nodes.putNoClobber(TensorNode.of(output), op_node) catch unreachable;
    }
}

pub const OpNode = union(ops.OpTypes) {
    const Input = struct {
        node: *TensorNode,
        fused: bool = false,
    };
    const Output = *TensorNode;
    pub const MapOp = struct {
        op: ops.MapOp,
        x: Input,
        out: Output,
        op_node_label: []const u8,
    };
    pub const ZipOp = struct {
        op: ops.ZipOp,
        a: Input,
        b: Input,
        out: Output,
        op_node_label: []const u8,
    };
    pub const ReduceOp = struct {
        op: ops.ReduceOp,
        x: Input,
        dims: []const bool,
        out: Output,
        op_node_label: []const u8,
    };
    pub const TypeOp = struct {
        op: ops.TypeOp,
        x: Input,
        out: Output,
        op_node_label: []const u8,
    };
    pub const InitOp = struct {
        op: ops.InitOp,
        value: ops.InitValue,
        out: Output,
        op_node_label: []const u8,
    };
    MapOp: MapOp,
    ZipOp: ZipOp,
    ReduceOp: ReduceOp,
    TypeOp: TypeOp,
    InitOp: InitOp,
};

var viz_hash_table: std.AutoHashMap(usize, bool) = undefined;

pub const TensorNode = struct {
    t_edge_label: []const u8,
    node_id: usize,
    group_id: ?usize = null,
    cached: bool = false,

    /// Get the tensor node or create it and add it to the map
    pub fn of(tensor_ptr: anytype) *TensorNode {
        const Tensor = @TypeOf(tensor_ptr.*);
        const key = @intFromPtr(tensor_ptr);
        if (!tensor_nodes.contains(key)) {
            const t_node = allocator.create(TensorNode) catch unreachable;
            t_node.* = .{
                .node_id = tensor_nodes.count(),
                .t_edge_label = std.fmt.comptimePrint("{s}{any}", .{ @tagName(Tensor.dtype), Tensor.shape }),
            };
            tensor_nodes.putNoClobber(key, t_node) catch unreachable;
            entry_node = t_node;
            return t_node;
        } else {
            return tensor_nodes.get(key).?;
        }
    }

    pub fn nodeId(node: *const TensorNode) usize {
        return node.node_id;
    }

    pub fn tensorId(node: *const TensorNode) usize {
        const preceding_op_node = preceding_op_nodes.get(node).?;
        return switch (preceding_op_node) {
            .InitOp => node.nodeId(),
            .ZipOp => |op_node| if (op_node.a.fused and !op_node.b.fused) op_node.a.node.tensorId() else if (op_node.b.fused and !op_node.a.fused) op_node.b.node.tensorId() else node.nodeId(),
            inline else => |op_node| if (op_node.x.fused) op_node.x.node.tensorId() else node.nodeId(),
        };
    }

    fn tensorNodeViz(self: *const TensorNode, writer: anytype) void {
        // To avoid printing the same thing multiple times use the hash table to check/mark as already printed
        if (viz_hash_table.get(self.nodeId()) orelse false) {
            return;
        }
        viz_hash_table.put(self.nodeId(), true) catch unreachable;
        const preceding_op_node = preceding_op_nodes.get(self).?;
        switch (preceding_op_node) {
            inline else => |op_node| {
                writer.print("T{d}[label=\"T{d}\"shape=box];\n", .{ self.tensorId(), self.tensorId() });
                if (self.cached) {
                    writer.print("subgraph cluster{d}{{T{d}_{d}[label=\"T{d}_{d}\"shape=box];}}\n", .{ self.group_id.?, self.tensorId(), self.group_id.?, self.tensorId(), self.group_id.? });
                    writer.print("T{d}_{d}->T{d}[label=\"{s}\"];\n", .{ self.tensorId(), self.group_id.?, self.tensorId(), self.t_edge_label });
                    writer.print("{s}{d}->T{d}_{d}[label=\"{s}\"];\n", .{ @tagName(op_node.op), self.nodeId(), self.tensorId(), self.group_id.?, self.t_edge_label });
                } else {
                    writer.print("{s}{d}->T{d}[label=\"{s}\"];\n", .{ @tagName(op_node.op), self.nodeId(), self.tensorId(), self.t_edge_label });
                }
            },
        }
    }

    fn precedingOpNodeViz(self: *const TensorNode, input: OpNode.Input, writer: anytype) void {
        const preceding_op_node = preceding_op_nodes.get(self).?;
        switch (preceding_op_node) {
            inline else => |op_node| {
                if (self.group_id != null) {
                    writer.print("subgraph cluster{d}{{{s}{d}[label=\"{s}\"];}}\n", .{ self.group_id.?, @tagName(op_node.op), self.nodeId(), op_node.op_node_label });
                } else {
                    writer.print("{s}{d}[label=\"{s}\"];\n", .{ @tagName(op_node.op), self.nodeId(), op_node.op_node_label });
                }
            },
        }
        switch (preceding_op_node) {
            .InitOp => {}, // InitOp will not have a previous tensor node to connect to
            inline else => |op_node| {
                if (input.fused) {
                    writer.print("{s}{d}->{s}{d}[label=\"{s}\"];\n", .{ @tagName(op_node.op), input.node.nodeId(), @tagName(op_node.op), self.nodeId(), input.node.t_edge_label });
                } else {
                    input.node.tensorNodeViz(writer);
                    if (op_node.out.group_id != null and input.node.group_id == op_node.out.group_id and input.node.cached) {
                        writer.print("T{d}_{?}->{s}{d}[label=\"{s}\"];\n", .{ input.node.tensorId(), input.node.group_id, @tagName(op_node.op), self.nodeId(), input.node.t_edge_label });
                    } else {
                        writer.print("T{d}->{s}{d}[label=\"{s}\"];\n", .{ input.node.tensorId(), @tagName(op_node.op), self.nodeId(), input.node.t_edge_label });
                    }
                }
            },
        }
    }

    fn viz(self: *const TensorNode, writer: anytype) void {
        if (viz_hash_table.contains(self.nodeId())) {
            return;
        }
        viz_hash_table.putNoClobber(self.nodeId(), false) catch unreachable;
        const preceding_op_node = preceding_op_nodes.get(self).?;
        // Recursive calls
        switch (preceding_op_node) {
            .InitOp => {},
            .MapOp => |op_node| op_node.x.node.viz(writer),
            .ZipOp => |op_node| {
                op_node.a.node.viz(writer);
                op_node.b.node.viz(writer);
            },
            .ReduceOp => |op_node| op_node.x.node.viz(writer),
            .TypeOp => |op_node| op_node.x.node.viz(writer),
        }
        switch (preceding_op_node) {
            .InitOp => {},
            .MapOp => |op_node| self.precedingOpNodeViz(op_node.x, writer),
            .ZipOp => |op_node| {
                self.precedingOpNodeViz(op_node.a, writer);
                self.precedingOpNodeViz(op_node.b, writer);
            },
            .ReduceOp => |op_node| self.precedingOpNodeViz(op_node.x, writer),
            .TypeOp => |op_node| self.precedingOpNodeViz(op_node.x, writer),
        }
        tensorNodeViz(self, writer);
    }
};

pub fn viz(writer: anytype) !void {
    viz_hash_table = std.AutoHashMap(usize, bool).init(allocator);
    defer {
        viz_hash_table.deinit();
        viz_hash_table = undefined;
    }
    writer.print("digraph G {{\ncompound=true;\n", .{});
    entry().viz(writer);
    writer.print("}}\n", .{});
}

pub const Fusion = struct {
    const FusionError = error{
        ParentReduce,
        ParentInit,
        NotParentChild,
    };

    pub fn verticalFusion(parent: *TensorNode, child: *TensorNode) FusionError!void {
        const parent_group_contains_reduction = if (parent.group_id) |group_id| reduction_groups.get(group_id) orelse false else false;
        switch (child.op_node) {
            .InitOp => |*op_node| {
                switch (op_node.op) {
                    // Only fuse an init op if it is a constant value fill
                    // This is trival to inline, but others may not be
                    .Full => op_node.fused_init = true,
                    else => return FusionError.ParentInit,
                }
            },
            .ZipOp => |*op_node| {
                if (op_node.a.tensorId() != parent.tensorId() and op_node.b.tensorId() != parent.tensorId()) {
                    return FusionError.NotParentChild;
                }
                if (op_node.a.tensorId() == parent.tensorId()) {
                    if (parent_group_contains_reduction) {
                        return FusionError.ParentReduce;
                    }
                    op_node.fused_a = true;
                }
                if (op_node.b.tensorId() == parent.tensorId()) {
                    if (parent_group_contains_reduction) {
                        return FusionError.ParentReduce;
                    }
                    op_node.fused_b = true;
                }
            },
            .TypeOp => |*op_node| {
                // Fuse a TypeOp even when the previous op is a reduce op
                // The only type op that has any loop is a cast, and that is trivial enough
                // to warrant not needing an extra loop anyways
                if (op_node.x.tensorId() != parent.tensorId()) {
                    return FusionError.NotParentChild;
                }
                op_node.fused_x = true;
            },
            .MapOp => |*op_node| {
                if (op_node.x.tensorId() != parent.tensorId()) {
                    return FusionError.NotParentChild;
                }
                if (parent_group_contains_reduction) {
                    return FusionError.ParentReduce;
                }
                op_node.fused_x = true;
            },
            .ReduceOp => |*op_node| {
                if (op_node.x.tensorId() != parent.tensorId()) {
                    return FusionError.NotParentChild;
                }
                if (parent_group_contains_reduction) {
                    return FusionError.ParentReduce;
                }
                op_node.fused_x = true;
            },
        }
    }

    fn greedyFusionHelper(group_id: usize, node: *TensorNode) usize {
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
        switch (node.op_node) {
            .MapOp => |*op_node| {
                node.group_id = greedyFusionHelper(group_id, op_node.x);
                if (op_node.x.group_id == node.group_id and !op_node.x.cached) {
                    verticalFusion(op_node.x, node) catch {
                        // If we get a fusion error, move the current node to the next group
                        node.group_id = node.group_id.? + 1;
                    };
                }
                while (reduction_groups.get(node.group_id.?) orelse false) {
                    node.group_id = node.group_id.? + 1;
                }
            },
            .ZipOp => |*op_node| {
                // Greedy fusion helper returns the next group id so here it is passed from a -> b -> current
                node.group_id = greedyFusionHelper(greedyFusionHelper(group_id, op_node.a), op_node.b);
                if (op_node.a.group_id == node.group_id and !op_node.a.cached) {
                    verticalFusion(op_node.a, node) catch {
                        node.group_id = node.group_id.? + 1;
                    };
                }
                if (op_node.b.group_id == node.group_id and !op_node.b.cached) {
                    verticalFusion(op_node.b, node) catch {
                        node.group_id = node.group_id.? + 1;
                    };
                }
                while (reduction_groups.get(node.group_id.?) orelse false) {
                    node.group_id = node.group_id.? + 1;
                }
            },
            .ReduceOp => |*op_node| {
                node.group_id = greedyFusionHelper(group_id, op_node.x);
                if (op_node.x.group_id == node.group_id and !op_node.x.cached) {
                    verticalFusion(op_node.x, node) catch {
                        node.group_id = node.group_id.? + 1;
                    };
                }
                while (reduction_groups.get(node.group_id.?) orelse false) {
                    node.group_id = node.group_id.? + 1;
                }
                reduction_groups.putNoClobber(node.group_id.?, true) catch unreachable;
            },
            .TypeOp => |*op_node| {
                // TypeOps can always be fused into the preceding kernel even if the typeop follows a reduce
                // This is because it is either just index manipulation (and does not correspond to a loop)
                // or it is a cast which can be inlined during the accumulation of a reduction
                node.group_id = greedyFusionHelper(group_id, op_node.x);
                if (op_node.x.group_id == node.group_id and !op_node.x.cached) {
                    verticalFusion(op_node.x, node) catch {
                        node.group_id = node.group_id.? + 1;
                    };
                }
            },
            // Init will happen outside a kernel unless it is a full init
            .InitOp => |*op_node| {
                if (op_node.op == .Full) {
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
    const t8 = t9.op_node.ZipOp.b;
    try Fusion.verticalFusion(t8, t9);
    const t7 = t8.op_node.TypeOp.x.op_node.MapOp.x;
    const t6 = t7.op_node.ReduceOp.x;
    try Fusion.verticalFusion(t6, t7);
    const t5 = t6.op_node.MapOp.x;
    try Fusion.verticalFusion(t5, t6);

    // const t3 = t9.op_node.ZipOp.a;
    // try fuse(t3, t9);
    // try fuse(t3, t5);
    // const t2 = t3.op_node.ZipOp.b;
    // try fuse(t2, t3);
    // const t1 = t2.op_node.MapOp.x;
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
