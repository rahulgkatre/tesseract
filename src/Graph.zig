const std = @import("std");
const ops = @import("ops.zig");
const tensor = @import("tensor.zig");
const utils = @import("utils.zig");
const Graph = @This();
const dtypes = @import("dtypes.zig");

var arena: std.heap.ArenaAllocator = undefined;
var allocator: std.mem.Allocator = undefined;
var tensor_nodes: std.AutoHashMap(*const anyopaque, *TensorNode) = undefined;
var op_nodes: std.AutoHashMap(*const anyopaque, *OpNode) = undefined;
var reduction_groups: std.AutoHashMap(usize, bool) = undefined;
var entry_node: ?*TensorNode = null;

pub fn entry() *TensorNode {
    return entry_node orelse @panic("Graph has no entrypoint. Remember to call trace() on an output tensor pointer");
}

pub fn init(backing_allocator: std.mem.Allocator) void {
    arena = std.heap.ArenaAllocator.init(backing_allocator);
    allocator = arena.allocator();
    tensor_nodes = std.AutoHashMap(*const anyopaque, *TensorNode).init(allocator);
    op_nodes = std.AutoHashMap(*const anyopaque, *OpNode).init(allocator);
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
    pub const Input = struct {
        ptr: *const anyopaque,
        fused: bool = false,

        pub fn node(self: *const Input) *TensorNode {
            return tensor_nodes.get(self.ptr).?;
        }
    };
    pub const Output = struct {
        ptr: *const anyopaque,

        pub fn node(self: *const Output) *TensorNode {
            return tensor_nodes.get(self.ptr).?;
        }
    };
    pub const MapOp = struct {
        op: ops.MapOp,
        x: Input,
        out: Output,
        node_label: []const u8,
    };
    pub const ZipOp = struct {
        op: ops.ZipOp,
        a: Input,
        b: Input,
        out: Output,
        node_label: []const u8,
    };
    pub const ReduceOp = struct {
        op: ops.ReduceOp,
        x: Input,
        dims: []const bool,
        out: Output,
        node_label: []const u8,
    };
    pub const TypeOp = struct {
        op: ops.TypeOp,
        x: Input,
        out: Output,
        node_label: []const u8,
    };
    pub const InitOp = struct {
        op: ops.InitOp,
        value: ops.InitValue,
        out: Output,
        node_label: []const u8,
    };
    MapOp: MapOp,
    ZipOp: ZipOp,
    ReduceOp: ReduceOp,
    TypeOp: TypeOp,
    InitOp: InitOp,

    fn viz(self: *const OpNode, target: ?OpNode.Input, writer: anytype) void {
        switch (self.*) {
            inline else => |op_node| {
                if (op_node.out.node().group != null) {
                    writer.print("subgraph cluster{d}{{{s}{d}[label=\"{s}\"];}}\n", .{ op_node.out.node().group.?, @tagName(op_node.op), op_node.out.node().uid, op_node.node_label });
                } else {
                    writer.print("{s}{d}[label=\"{s}\"];\n", .{ @tagName(op_node.op), op_node.out.node().uid, op_node.node_label });
                }
            },
        }
        switch (self.*) {
            .InitOp => {}, // InitOp will not have a previous tensor node to connect to
            inline else => |op_node| {
                if (target.?.fused) {
                    switch (op_nodes.get(target.?.node().tensor.ptr).?.*) {
                        inline else => |in_op_node| writer.print("{s}{d}->{s}{d}[label=\"{s}\"];\n", .{ @tagName(in_op_node.op), target.?.node().uid, @tagName(op_node.op), op_node.out.node().uid, target.?.node().edge_label }),
                    }
                } else {
                    if (op_node.out.node().group != null and target.?.node().group == op_node.out.node().group and target.?.node().cached) {
                        writer.print("T{d}_{?}->{s}{d}[label=\"{s}\"];\n", .{ target.?.node().tensor.id(), target.?.node().group, @tagName(op_node.op), op_node.out.node().uid, target.?.node().edge_label });
                    } else {
                        writer.print("T{d}->{s}{d}[label=\"{s}\"];\n", .{ target.?.node().tensor.id(), @tagName(op_node.op), op_node.out.node().uid, target.?.node().edge_label });
                    }
                }
            },
        }
    }

    fn getOrInit(comptime op: ops.GraphOp, comptime input: anytype, output: anytype, comptime args: anytype) *OpNode {
        const Out = @TypeOf(output.*);
        const ptr = @as(*const anyopaque, output);
        if (op_nodes.get(ptr)) |op_node| {
            return op_node;
        } else {
            switch (op) {
                .MapOp, .TypeOp, .ReduceOp => trace(input.x),
                .ZipOp => {
                    trace(input.a);
                    trace(input.b);
                },
                else => {},
            }
            const op_node = allocator.create(OpNode) catch unreachable;
            op_node.* = switch (op) {
                .MapOp => |map_op| @unionInit(OpNode, @tagName(op), .{
                    .op = map_op,
                    .x = .{ .ptr = (input.x) },
                    .out = .{ .ptr = (output) },
                    .node_label = std.fmt.comptimePrint("{s}", .{@tagName(map_op)}),
                }),
                .ZipOp => |zip_op| @unionInit(OpNode, @tagName(op), .{
                    .op = zip_op,
                    .a = .{ .ptr = (input.a) },
                    .b = .{ .ptr = (input.b) },
                    .out = .{ .ptr = (output) },
                    .node_label = std.fmt.comptimePrint("{s}", .{@tagName(zip_op)}),
                }),
                .ReduceOp => |reduce_op| @unionInit(OpNode, @tagName(op), .{
                    .op = reduce_op,
                    .x = .{ .ptr = (input.x) },
                    .out = .{ .ptr = (output) },
                    .dims = args.dims,
                    .node_label = std.fmt.comptimePrint("{s}{any}", .{ @tagName(reduce_op), @as([]const bool, args.dims) }),
                }),
                .TypeOp => |type_op| @unionInit(OpNode, @tagName(op), .{
                    .op = type_op,
                    .x = .{ .ptr = (input.x) },
                    .node_label = switch (type_op) {
                        .AsType => std.fmt.comptimePrint("{s}{any}", .{ @tagName(type_op), Out.dtype }),
                        .View, .Broadcast => std.fmt.comptimePrint("{s}{any}", .{ @tagName(type_op), Out.shape }),
                        .AsStrided => std.fmt.comptimePrint("{s}{{{any},{any}}}", .{ @tagName(type_op), Out.shape, Out.strides }),
                    },
                    .out = .{ .ptr = (output) },
                }),
                .InitOp => |init_op| @unionInit(OpNode, @tagName(op), .{
                    .op = init_op,
                    .value = args,
                    .out = .{ .ptr = (output) },
                    .node_label = std.fmt.comptimePrint("{s}", .{@tagName(init_op)}),
                }),
            };
            const out = TensorNode.getOrInit(output);
            op_nodes.putNoClobber(out.tensor.ptr, op_node) catch unreachable;
            return op_node;
        }
    }
};

pub const TensorNode = struct {
    const Tensor = struct {
        ptr: *const anyopaque,
        dtype: dtypes.DType,
        ndims: u8,
        shape: []const usize,
        strides: []const usize,

        /// A tensor id is a id for each tensor's underlying buffer, which can be shared
        pub fn id(field_ptr: *const TensorNode.Tensor) usize {
            const tensor_node = @fieldParentPtr(TensorNode, "tensor", field_ptr);
            return switch (tensor_node.opNode().*) {
                .InitOp => tensor_node.uid,
                .ZipOp => |op_node| blk: {
                    if (op_node.a.fused and !op_node.b.fused) {
                        break :blk op_node.a.node().tensor.id();
                    } else if (op_node.b.fused and !op_node.a.fused) {
                        break :blk op_node.b.node().tensor.id();
                    } else {
                        break :blk tensor_node.uid;
                    }
                },
                inline else => |op_node| if (op_node.x.fused) op_node.x.node().tensor.id() else tensor_node.uid,
            };
        }
    };

    tensor: Tensor,
    edge_label: []const u8,
    uid: usize,
    group: ?usize = null,
    cached: bool = false,

    /// Get the tensor node or create it and add it to the map
    fn getOrInit(tensor_ptr: anytype) *TensorNode {
        const T = @TypeOf(tensor_ptr.*);
        const ptr = @as(*const anyopaque, tensor_ptr);
        if (tensor_nodes.get(ptr)) |tensor_node| {
            return tensor_node;
        } else {
            const node = allocator.create(TensorNode) catch unreachable;
            node.* = .{
                .tensor = .{
                    .ptr = ptr,
                    .dtype = T.dtype,
                    .ndims = T.ndims,
                    .shape = T.shape[0..T.ndims],
                    .strides = T.strides[0 .. T.ndims + 1],
                },
                .uid = tensor_nodes.count(),
                .edge_label = std.fmt.comptimePrint("{s}{any}", .{ @tagName(T.dtype), T.shape }),
            };
            tensor_nodes.putNoClobber(ptr, node) catch unreachable;
            return node;
        }
    }

    /// Retrieve the op node for which the tensor node is the output of
    pub fn opNode(tensor_node: *const TensorNode) *OpNode {
        return op_nodes.get(tensor_node.tensor.ptr).?;
    }

    fn viz(tensor_node: *const TensorNode, writer: anytype, visited: []bool) void {
        // To avoid printing the same thing multiple times use the table to check/mark as already printed
        if (visited[tensor_node.uid]) {
            return;
        }
        switch (tensor_node.opNode().*) {
            inline else => |op_node| {
                writer.print("T{d}[label=\"T{d}\"shape=box];\n", .{ tensor_node.tensor.id(), tensor_node.tensor.id() });
                if (tensor_node.cached) {
                    writer.print("subgraph cluster{d}{{T{d}_{d}[label=\"T{d}_{d}\"shape=box];}}\n", .{ tensor_node.group.?, tensor_node.tensor.id(), tensor_node.group.?, tensor_node.tensor.id(), tensor_node.group.? });
                    writer.print("T{d}_{d}->T{d}[label=\"{s}\"];\n", .{ tensor_node.tensor.id(), tensor_node.group.?, tensor_node.tensor.id(), tensor_node.edge_label });
                    writer.print("{s}{d}->T{d}_{d}[label=\"{s}\"];\n", .{ @tagName(op_node.op), tensor_node.uid, tensor_node.tensor.id(), tensor_node.group.?, tensor_node.edge_label });
                } else {
                    writer.print("{s}{d}->T{d}[label=\"{s}\"];\n", .{ @tagName(op_node.op), tensor_node.uid, tensor_node.tensor.id(), tensor_node.edge_label });
                }
            },
        }
        visited[tensor_node.uid] = true;
    }
};

fn vizHelper(input: OpNode.Input, writer: anytype, visited: []bool) void {
    if (visited[input.node().uid]) {
        return;
    }
    const op_node = op_nodes.get(input.node().tensor.ptr).?.*;
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
        input.node().viz(writer, visited);
    }
}

pub fn viz(writer: anytype) void {
    const visited = allocator.alloc(bool, tensor_nodes.count()) catch unreachable;
    defer allocator.free(visited);
    writer.print(
        \\digraph G {{
        \\compound=true;
        \\
    , .{});
    vizHelper(.{ .ptr = entry().tensor.ptr }, writer, visited);
    writer.print("}}\n", .{});
}

pub const Fusion = struct {
    const FusionError = error{
        ParentReduce,
        ParentInit,
        NotParentChild,
    };

    pub fn verticalFusion(parent: *TensorNode, child: *TensorNode) FusionError!void {
        const parent_group_contains_reduction = if (parent.group) |group| reduction_groups.get(group) orelse false else false;
        var op_node = op_nodes.get(child.tensor.ptr).?;
        switch (op_node.*) {
            .InitOp => unreachable, // Impossible as init op will only have a child (output) and no tensor input
            .ZipOp => |zip_op_node| {
                if (zip_op_node.a.node().tensor.id() != parent.tensor.id() and zip_op_node.b.node().tensor.id() != parent.tensor.id()) {
                    return FusionError.NotParentChild;
                }
                if (zip_op_node.a.node().tensor.id() == parent.tensor.id()) {
                    if (parent_group_contains_reduction) {
                        return FusionError.ParentReduce;
                    }
                    op_node.ZipOp.a.fused = true;
                }
                if (zip_op_node.b.node().tensor.id() == parent.tensor.id()) {
                    if (parent_group_contains_reduction) {
                        return FusionError.ParentReduce;
                    }
                    op_node.ZipOp.b.fused = true;
                }
            },
            .TypeOp => |type_op_node| {
                // Fuse a TypeOp even when the previous op is a reduce op
                // The only type op that has any loop is a cast, and that is trivial enough to inline
                if (type_op_node.x.node().tensor.id() != parent.tensor.id()) {
                    return FusionError.NotParentChild;
                }
                op_node.TypeOp.x.fused = true;
            },
            .MapOp => |map_op_node| {
                if (map_op_node.x.node().tensor.id() != parent.tensor.id()) {
                    return FusionError.NotParentChild;
                }
                if (parent_group_contains_reduction) {
                    return FusionError.ParentReduce;
                }
                op_node.MapOp.x.fused = true;
            },
            .ReduceOp => |reduce_op_node| {
                if (reduce_op_node.x.node().tensor.id() != parent.tensor.id()) {
                    return FusionError.NotParentChild;
                }
                if (parent_group_contains_reduction) {
                    return FusionError.ParentReduce;
                }
                op_node.ReduceOp.x.fused = true;
            },
        }
        // Replace op node after marking the input as fused
        op_nodes.putAssumeCapacity(child.tensor.ptr, op_node);
    }

    /// Recursive function to fuse every parent child pair when possible.
    /// Keeps track of group ids (i.e. kernels) to prevent multiple thread synchronization requiring operations
    /// (reductions) from being in the same kernel. This might change after further testing.
    fn greedyFusionHelper(group: usize, node: *TensorNode) usize {
        if (node.group == group) {
            // A node can be cached in the kernel if it is being reused by 1 or more dependents
            // in the same tensor
            // Could also make this a counter to determine the number of times a tensor is reused
            // to see if just repeatedly calculating it again is faster than reading it out of memory
            node.cached = true;
        }
        if (node.cached) {
            if (node.group != group) {
                node.cached = false;
            }
            return node.group.?;
        }
        switch (op_nodes.get(node.tensor.ptr).?.*) {
            .MapOp => |op_node| {
                node.group = greedyFusionHelper(group, op_node.x.node());
                if (op_node.x.node().group == node.group and !op_node.x.node().cached) {
                    verticalFusion(op_node.x.node(), node) catch {
                        // If we get a fusion error, move the current node to the next group
                        node.group = node.group.? + 1;
                    };
                }
                while (reduction_groups.get(node.group.?) orelse false) {
                    node.group = node.group.? + 1;
                }
            },
            .ZipOp => |op_node| {
                // Greedy fusion helper returns the next group id so here it is passed from a -> b -> current
                node.group = greedyFusionHelper(greedyFusionHelper(group, op_node.a.node()), op_node.b.node());
                if (op_node.a.node().group == node.group and !op_node.a.node().cached) {
                    verticalFusion(op_node.a.node(), node) catch {
                        node.group = node.group.? + 1;
                    };
                }
                if (op_node.b.node().group == node.group and !op_node.b.node().cached) {
                    verticalFusion(op_node.b.node(), node) catch {
                        node.group = node.group.? + 1;
                    };
                }
                while (reduction_groups.get(node.group.?) orelse false) {
                    node.group = node.group.? + 1;
                }
            },
            .ReduceOp => |op_node| {
                node.group = greedyFusionHelper(group, op_node.x.node());
                if (op_node.x.node().group == node.group and !op_node.x.node().cached) {
                    verticalFusion(op_node.x.node(), node) catch {
                        node.group = node.group.? + 1;
                    };
                }
                while (reduction_groups.get(node.group.?) orelse false) {
                    node.group = node.group.? + 1;
                }
                reduction_groups.putNoClobber(node.group.?, true) catch unreachable;
            },
            .TypeOp => |op_node| {
                // TypeOps can always be fused into the preceding kernel even if the typeop follows a reduce
                // This is because it is either just index manipulation and does not produce a loop
                // or it is a cast which can be inlined when assigning the value in the output tensor
                node.group = greedyFusionHelper(group, op_node.x.node());
                if (op_node.x.node().group == node.group and !op_node.x.node().cached) {
                    verticalFusion(op_node.x.node(), node) catch {
                        node.group = node.group.? + 1;
                    };
                }
            },
            // Init will happen outside a kernel unless it is a full init
            .InitOp => |op_node| {
                if (op_node.op == .Full) {
                    node.group = group;
                }
                return group;
            },
        }
        return node.group.?;
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
