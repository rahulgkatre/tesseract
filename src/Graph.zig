const std = @import("std");
const ops = @import("ops.zig");
const tensor = @import("tensor.zig");
const utils = @import("utils.zig");
const Graph = @This();
const dtypes = @import("dtypes.zig");

var arena: std.heap.ArenaAllocator = undefined;
var allocator: std.mem.Allocator = undefined;
var tensor_nodes: std.AutoHashMap(usize, *TensorNode) = undefined;
var op_nodes: std.AutoHashMap(usize, OpNode) = undefined;
var reduction_groups: std.AutoHashMap(usize, bool) = undefined;
var entry_node: ?*TensorNode = null;

pub fn entry() *TensorNode {
    return entry_node orelse @panic("Graph has no entrypoint. Remember to call trace() on an output tensor pointer");
}

pub fn init(backing_allocator: std.mem.Allocator) void {
    arena = std.heap.ArenaAllocator.init(backing_allocator);
    allocator = arena.allocator();
    tensor_nodes = std.AutoHashMap(usize, *TensorNode).init(allocator);
    op_nodes = std.AutoHashMap(usize, OpNode).init(allocator);
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
        .Pointer => tensor_ptr.trace_fn(tensor_ptr),
        else => @compileError("Must pass a tensor pointer to Graph.trace()"),
    }
    entry_node = TensorNode.getOrInit(tensor_ptr);
}

/// Each tensor trace callback uses this function to add its dependencies (input), operation (op), and result (output)
/// to the computation graph
pub fn addOp(comptime op: ops.GraphOp, input: anytype, output: anytype, comptime args: anytype) void {
    _ = OpNode.getOrInit(op, input, output, args);
}

pub const OpNode = union(ops.OpTypes) {
    const Input = struct {
        node: *TensorNode,
        fused: bool = false,
    };
    const Output = struct {
        node: *TensorNode,
    };
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

    fn viz(self: *const OpNode, in: ?OpNode.Input, writer: anytype) void {
        switch (self.*) {
            inline else => |op_node| {
                if (op_node.out.node.group_id != null) {
                    writer.print("subgraph cluster{d}{{{s}{d}[label=\"{s}\"];}}\n", .{ op_node.out.node.group_id.?, @tagName(op_node.op), op_node.out.node.nodeId(), op_node.op_node_label });
                } else {
                    writer.print("{s}{d}[label=\"{s}\"];\n", .{ @tagName(op_node.op), op_node.out.node.nodeId(), op_node.op_node_label });
                }
            },
        }
        switch (self.*) {
            .InitOp => {}, // InitOp will not have a previous tensor node to connect to
            inline else => |op_node| {
                if (in.?.fused) {
                    switch (op_nodes.get(in.?.node.tensor_addr).?) {
                        inline else => |in_op_node| writer.print("{s}{d}->{s}{d}[label=\"{s}\"];\n", .{ @tagName(in_op_node.op), in.?.node.nodeUid(), @tagName(op_node.op), op_node.out.node.nodeId(), in.?.node.edge_label }),
                    }
                } else {
                    if (op_node.out.node.group_id != null and in.?.node.group_id == op_node.out.node.group_id and in.?.node.cached) {
                        writer.print("T{d}_{?}->{s}{d}[label=\"{s}\"];\n", .{ in.?.node.tensorId(), in.?.node.group_id, @tagName(op_node.op), op_node.out.node.nodeId(), in.?.node.edge_label });
                    } else {
                        writer.print("T{d}->{s}{d}[label=\"{s}\"];\n", .{ in.?.node.tensorId(), @tagName(op_node.op), op_node.out.node.nodeId(), in.?.node.edge_label });
                    }
                }
            },
        }
    }

    fn getOrInit(comptime op: ops.GraphOp, input: anytype, output: anytype, comptime args: anytype) OpNode {
        const Out = @TypeOf(output.*);
        const out_addr = @intFromPtr(output);
        if (op_nodes.get(out_addr)) |op_node| {
            return op_node;
        } else {
            const op_node: OpNode = switch (op) {
                .MapOp => |map_op| blk: {
                    trace(input.x);
                    break :blk .{ .MapOp = .{
                        .op = map_op,
                        .x = .{ .node = TensorNode.getOrInit(input.x) },
                        .out = .{ .node = TensorNode.getOrInit(output) },
                        .op_node_label = std.fmt.comptimePrint("{s}", .{@tagName(map_op)}),
                    } };
                },
                .ZipOp => |zip_op| blk: {
                    trace(input.a);
                    trace(input.b);
                    break :blk .{ .ZipOp = .{
                        .op = zip_op,
                        .a = .{ .node = TensorNode.getOrInit(input.a) },
                        .b = .{ .node = TensorNode.getOrInit(input.b) },
                        .out = .{ .node = TensorNode.getOrInit(output) },
                        .op_node_label = std.fmt.comptimePrint("{s}", .{@tagName(zip_op)}),
                    } };
                },
                .ReduceOp => |reduce_op| blk: {
                    trace(input.x);
                    break :blk .{
                        .ReduceOp = .{
                            .op = reduce_op,
                            .x = .{ .node = TensorNode.getOrInit(input.x) },
                            .out = .{ .node = TensorNode.getOrInit(output) },
                            .dims = args.dims,
                            .op_node_label = std.fmt.comptimePrint("{s}{any}", .{ @tagName(reduce_op), @as([]const bool, args.dims) }),
                        },
                    };
                },
                .TypeOp => |type_op| blk: {
                    trace(input.x);
                    break :blk .{ .TypeOp = .{
                        .op = type_op,
                        .x = .{ .node = TensorNode.getOrInit(input.x) },
                        .op_node_label = switch (type_op) {
                            .AsType => std.fmt.comptimePrint("{s}{any}", .{ @tagName(type_op), Out.dtype }),
                            .View, .Broadcast => std.fmt.comptimePrint("{s}{any}", .{ @tagName(type_op), Out.shape }),
                            .AsStrided => std.fmt.comptimePrint("{s}{{{any},{any}}}", .{ @tagName(type_op), Out.shape, Out.strides }),
                        },
                        .out = .{ .node = TensorNode.getOrInit(output) },
                    } };
                },
                .InitOp => |init_op| .{ .InitOp = .{
                    .op = init_op,
                    .value = args,
                    .out = .{ .node = TensorNode.getOrInit(output) },
                    .op_node_label = std.fmt.comptimePrint("{s}", .{@tagName(init_op)}),
                } },
            };
            const out_node = TensorNode.getOrInit(output);
            op_nodes.putNoClobber(out_node.tensor_addr, op_node) catch unreachable;
            return op_node;
        }
    }
};

pub const TensorNode = struct {
    edge_label: []const u8,
    tensor_addr: usize,
    node_id: usize,
    group_id: ?usize = null,
    cached: bool = false,

    /// Get the tensor node or create it and add it to the map
    fn getOrInit(tensor_ptr: anytype) *TensorNode {
        const Tensor = @TypeOf(tensor_ptr.*);
        const tensor_addr = @intFromPtr(tensor_ptr);
        if (tensor_nodes.get(tensor_addr)) |t_node| {
            return t_node;
        } else {
            const t_node = allocator.create(TensorNode) catch unreachable;
            t_node.* = .{
                .tensor_addr = tensor_addr,
                .node_id = tensor_nodes.count(),
                .edge_label = std.fmt.comptimePrint("{s}{any}", .{ @tagName(Tensor.dtype), Tensor.shape }),
            };
            tensor_nodes.putNoClobber(tensor_addr, t_node) catch unreachable;
            return t_node;
        }
    }

    /// A node id is a unique id for each tensor node
    pub fn nodeUid(self: *const TensorNode) usize {
        return self.node_id;
    }

    /// A tensor id is a id for each tensor's underlying buffer, which can be shared
    pub fn tensorId(self: *const TensorNode) usize {
        const preceding_op_node = op_nodes.get(self.tensor_addr).?;
        return switch (preceding_op_node) {
            .InitOp => self.nodeUid(),
            .ZipOp => |op_node| if (op_node.a.fused and !op_node.b.fused) op_node.a.node.tensorId() else if (op_node.b.fused and !op_node.a.fused) op_node.b.node.tensorId() else self.nodeUid(),
            inline else => |op_node| if (op_node.x.fused) op_node.x.node.tensorId() else self.nodeUid(),
        };
    }

    fn viz(self: *const TensorNode, writer: anytype, visited: []bool) void {
        // To avoid printing the same thing multiple times use the table to check/mark as already printed
        if (visited[self.nodeUid()]) {
            return;
        }
        switch (op_nodes.get(self.tensor_addr).?) {
            inline else => |op_node| {
                writer.print("T{d}[label=\"T{d}\"shape=box];\n", .{ self.tensorId(), self.tensorId() });
                if (self.cached) {
                    writer.print("subgraph cluster{d}{{T{d}_{d}[label=\"T{d}_{d}\"shape=box];}}\n", .{ self.group_id.?, self.tensorId(), self.group_id.?, self.tensorId(), self.group_id.? });
                    writer.print("T{d}_{d}->T{d}[label=\"{s}\"];\n", .{ self.tensorId(), self.group_id.?, self.tensorId(), self.edge_label });
                    writer.print("{s}{d}->T{d}_{d}[label=\"{s}\"];\n", .{ @tagName(op_node.op), self.nodeUid(), self.tensorId(), self.group_id.?, self.edge_label });
                } else {
                    writer.print("{s}{d}->T{d}[label=\"{s}\"];\n", .{ @tagName(op_node.op), self.nodeUid(), self.tensorId(), self.edge_label });
                }
            },
        }
        visited[self.nodeUid()] = true;
    }
};

fn vizHelper(input: OpNode.Input, writer: anytype, visited: []bool) void {
    if (visited[input.node.nodeUid()]) {
        return;
    }
    const op_node = op_nodes.get(input.node.tensor_addr).?;
    // Recursive calls
    switch (op_node) {
        .InitOp => op_node.viz(null, writer),
        .ZipOp => |binary_op_node| {
            vizHelper(binary_op_node.a, writer, visited);
            op_node.viz(binary_op_node.a, writer);
            vizHelper(binary_op_node.b, writer, visited);
            op_node.viz(binary_op_node.b, writer);
        },
        inline else => |unary_op_node| {
            vizHelper(unary_op_node.x, writer, visited);
            op_node.viz(unary_op_node.x, writer);
        },
    }
    if (!input.fused) {
        input.node.viz(writer, visited);
    }
}

pub fn viz(writer: anytype) !void {
    const visited = try allocator.alloc(bool, tensor_nodes.count());
    defer allocator.free(visited);
    writer.print(
        \\digraph G {{
        \\compound=true;
        \\
    , .{});
    vizHelper(.{ .node = entry() }, writer, visited);
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
        var op_node = op_nodes.get(child.tensor_addr).?;
        switch (op_node) {
            .InitOp => unreachable, // Impossible as init op will only have a child (output) and no tensor input
            .ZipOp => |zip_op_node| {
                if (zip_op_node.a.node.tensorId() != parent.tensorId() and zip_op_node.b.node.tensorId() != parent.tensorId()) {
                    return FusionError.NotParentChild;
                }
                if (zip_op_node.a.node.tensorId() == parent.tensorId()) {
                    if (parent_group_contains_reduction) {
                        return FusionError.ParentReduce;
                    }
                    op_node.ZipOp.a.fused = true;
                }
                if (zip_op_node.b.node.tensorId() == parent.tensorId()) {
                    if (parent_group_contains_reduction) {
                        return FusionError.ParentReduce;
                    }
                    op_node.ZipOp.b.fused = true;
                }
            },
            .TypeOp => |type_op_node| {
                // Fuse a TypeOp even when the previous op is a reduce op
                // The only type op that has any loop is a cast, and that is trivial enough to inline
                if (type_op_node.x.node.tensorId() != parent.tensorId()) {
                    return FusionError.NotParentChild;
                }
                op_node.TypeOp.x.fused = true;
            },
            .MapOp => |map_op_node| {
                if (map_op_node.x.node.tensorId() != parent.tensorId()) {
                    return FusionError.NotParentChild;
                }
                if (parent_group_contains_reduction) {
                    return FusionError.ParentReduce;
                }
                op_node.MapOp.x.fused = true;
            },
            .ReduceOp => |reduce_op_node| {
                if (reduce_op_node.x.node.tensorId() != parent.tensorId()) {
                    return FusionError.NotParentChild;
                }
                if (parent_group_contains_reduction) {
                    return FusionError.ParentReduce;
                }
                op_node.ReduceOp.x.fused = true;
            },
        }
        // Replace op node after marking the input as fused
        op_nodes.putAssumeCapacity(child.tensor_addr, op_node);
    }

    /// Recursive function to fuse every parent child pair when possible.
    /// Keeps track of group ids (i.e. kernels) to prevent multiple thread synchronization requiring operations
    /// (reductions) from being in the same kernel. This might change after further testing.
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

        switch (op_nodes.get(node.tensor_addr).?) {
            .MapOp => |op_node| {
                node.group_id = greedyFusionHelper(group_id, op_node.x.node);
                if (op_node.x.node.group_id == node.group_id and !op_node.x.node.cached) {
                    verticalFusion(op_node.x.node, node) catch {
                        // If we get a fusion error, move the current node to the next group
                        node.group_id = node.group_id.? + 1;
                    };
                }
                while (reduction_groups.get(node.group_id.?) orelse false) {
                    node.group_id = node.group_id.? + 1;
                }
            },
            .ZipOp => |op_node| {
                // Greedy fusion helper returns the next group id so here it is passed from a -> b -> current
                node.group_id = greedyFusionHelper(greedyFusionHelper(group_id, op_node.a.node), op_node.b.node);
                if (op_node.a.node.group_id == node.group_id and !op_node.a.node.cached) {
                    verticalFusion(op_node.a.node, node) catch {
                        node.group_id = node.group_id.? + 1;
                    };
                }
                if (op_node.b.node.group_id == node.group_id and !op_node.b.node.cached) {
                    verticalFusion(op_node.b.node, node) catch {
                        node.group_id = node.group_id.? + 1;
                    };
                }
                while (reduction_groups.get(node.group_id.?) orelse false) {
                    node.group_id = node.group_id.? + 1;
                }
            },
            .ReduceOp => |op_node| {
                node.group_id = greedyFusionHelper(group_id, op_node.x.node);
                if (op_node.x.node.group_id == node.group_id and !op_node.x.node.cached) {
                    verticalFusion(op_node.x.node, node) catch {
                        node.group_id = node.group_id.? + 1;
                    };
                }
                while (reduction_groups.get(node.group_id.?) orelse false) {
                    node.group_id = node.group_id.? + 1;
                }
                reduction_groups.putNoClobber(node.group_id.?, true) catch unreachable;
            },
            .TypeOp => |op_node| {
                // TypeOps can always be fused into the preceding kernel even if the typeop follows a reduce
                // This is because it is either just index manipulation and does not produce a loop
                // or it is a cast which can be inlined when assigning the value in the output tensor
                node.group_id = greedyFusionHelper(group_id, op_node.x.node);
                if (op_node.x.node.group_id == node.group_id and !op_node.x.node.cached) {
                    verticalFusion(op_node.x.node, node) catch {
                        node.group_id = node.group_id.? + 1;
                    };
                }
            },
            // Init will happen outside a kernel unless it is a full init
            .InitOp => |op_node| {
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
    // const x = comptime tensor.InferredStrides(.f32, .{ 2, 16 }).full(0);
    // const sm = comptime softmax(x, 1);

    // Graph.init(std.testing.allocator);
    // defer Graph.deinit();
    // Graph.trace(&sm);

    // const t9 = Graph.entry();
    // const t8 = t9.op_node.ZipOp.b;
    // try Fusion.verticalFusion(t8, t9);
    // const t7 = t8.op_node.TypeOp.x.op_node.MapOp.x;
    // const t6 = t7.op_node.ReduceOp.x;
    // try Fusion.verticalFusion(t6, t7);
    // const t5 = t6.op_node.MapOp.x;
    // try Fusion.verticalFusion(t5, t6);

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
