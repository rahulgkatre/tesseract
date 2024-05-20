pub const TensorNode = struct {
    ptr: usize,
    dtype: dtypes.DType,
    ndims: u8,
    shape: []const Dim,
    strides: []const Dim,

    // label: []const u8,
    uid: u64,
    group: ?u64 = null,
    consumer_count: u16 = 0,
    global: bool = false,
    op_node: OpNode = undefined,

    pub fn isCached(tensor: *const TensorNode) bool {
        return (tensor.consumer_count > 1 and tensor.group != tensor.uid);
    }

    pub fn memoryView(self: *const TensorNode) u64 {
        switch (self.op_node) {
            .InitOp => return self.uid,
            .BinaryOp => |op_node| {
                if (!op_node.a.fused and !op_node.b.fused) {
                    return self.uid;
                }
                const a_mv = op_node.a.tensor.memoryView();
                const b_mv = op_node.b.tensor.memoryView();
                if (op_node.a.fused and !op_node.b.fused and !op_node.a.tensor.isCached()) {
                    return a_mv;
                } else if (op_node.b.fused and !op_node.a.fused and !op_node.b.tensor.isCached()) {
                    return b_mv;
                } else if (!op_node.a.tensor.isCached() and !op_node.b.tensor.isCached()) {
                    return @min(a_mv, b_mv);
                } else {
                    return self.uid;
                }
            },
            inline else => |op_node| return if (op_node.x.fused and !op_node.x.tensor.isCached()) op_node.x.tensor.memoryView() else self.uid,
        }
    }

    pub fn get(ptr: anytype) *TensorNode {
        return switch (@typeInfo(@TypeOf(ptr))) {
            .Pointer => tensors.get(@intFromPtr(ptr)).?,
            .Int => tensors.get(ptr).?,
            else => @panic("Must use a tensor pointer or int from tensor pointer"),
        };
    }

    fn create(tensor: anytype) void {
        // const Tensor = @TypeOf(tensor.*);
        // const fields = @typeInfo
        const ptr = @intFromPtr(tensor);
        if (!tensors.contains(ptr)) {
            const tensor_node = arena.allocator().create(TensorNode) catch unreachable;
            tensor_node.* = .{
                .ptr = ptr,
                .dtype = tensor.dtype,
                .ndims = tensor.ndims,
                .shape = tensor.shape[0..tensor.ndims],
                .strides = tensor.strides[0 .. tensor.ndims + 1],
                .group = tensors.count(),
                .uid = tensors.count(),
                // .label = std.fmt.comptimePrint("{s}{any}", .{ @tagName(tensor.dtype), tensor.shape }),
            };
            tensors.putNoClobber(ptr, tensor_node) catch unreachable;
        }
    }
};

pub fn viz(writer: anytype) void {
    const Viz = struct {
        fn vizHelper(target: OpNode.Input, visited: []bool) void {
            if (visited[target.tensor.uid]) {
                return;
            }
            const op_node = target.tensor.op_node;
            // Recursive calls
            switch (op_node) {
                .InitOp => opNodeViz(op_node, visited), // the undefined tensor field is never accessed for an init op
                .BinaryOp => |binary_op_node| {
                    vizHelper(binary_op_node.a, visited);
                    opNodeViz(op_node, visited);
                    opNodeInputViz(op_node, binary_op_node.a, visited);
                    vizHelper(binary_op_node.b, visited);
                    opNodeInputViz(op_node, binary_op_node.b, visited);
                },
                inline else => |unary_op_node| {
                    vizHelper(unary_op_node.x, visited);
                    opNodeViz(op_node, visited);
                    opNodeInputViz(op_node, unary_op_node.x, visited);
                },
            }
            if (!target.fused) {
                tensorNodeViz(target.tensor, visited);
            }
            visited[target.tensor.uid] = true;
        }

        fn tensorNodeViz(tensor: *const TensorNode, visited: []bool) void {
            // To avoid printing the same thing multiple times use the table to check/mark as already printed
            if (visited[tensor.uid]) {
                return;
            }
            switch (tensor.op_node) {
                inline else => |op_node| {
                    if (tensor.global) {
                        writer.print("T{d}[label=\"T{d}\"shape=box];\n", .{ tensor.memoryView(), tensor.memoryView() });
                    }
                    if (tensor.isCached() and tensor.group != null) {
                        writer.print("subgraph cluster{d}{{t{d}[label=\"t{d}\"shape=box];}}\n", .{ tensor.group.?, tensor.uid, tensor.uid });
                        if (tensor.global) {
                            writer.print("t{d}->T{d}[label=\"{s}\"];\n", .{ tensor.uid, tensor.uid, tensor.label });
                        }
                        writer.print("{s}{d}->t{d}[label=\"{s}\"];\n", .{ @tagName(op_node.op), tensor.uid, tensor.uid, tensor.label });
                    } else {
                        writer.print("{s}{d}->T{d}[label=\"{s}\"];\n", .{ @tagName(op_node.op), tensor.uid, tensor.memoryView(), tensor.label });
                    }
                },
            }
        }

        fn opNodeViz(op_node: OpNode, visited: []bool) void {
            switch (op_node) {
                inline else => |node| {
                    if (visited[node.out.tensor.uid]) {
                        return;
                    }
                    if (node.out.tensor.group != null) {
                        writer.print("subgraph cluster{d}{{{s}{d}[label=\"{d} : {s}\"];}}\n", .{ node.out.tensor.group.?, @tagName(node.op), node.out.tensor.uid, node.out.tensor.uid, node.label });
                    } else {
                        writer.print("{s}{d}[label=\"{d} : {s}\"];\n", .{ @tagName(node.op), node.out.tensor.uid, node.out.tensor.uid, node.label });
                    }
                },
            }
        }

        fn opNodeInputViz(op_node: OpNode, target: OpNode.Input, visited: []bool) void {
            switch (op_node) {
                .InitOp => unreachable,
                inline else => |node| {
                    if (visited[node.out.tensor.uid]) {
                        return;
                    }

                    if (target.fused and !target.tensor.isCached()) {
                        switch (target.tensor.op_node) {
                            inline else => |in_op_node| writer.print("{s}{d}->{s}{d}[label=\"{s}\"];\n", .{ @tagName(in_op_node.op), target.tensor.uid, @tagName(node.op), node.out.tensor.uid, target.tensor.label }),
                        }
                    } else {
                        if (node.out.tensor.group != null and target.tensor.group == node.out.tensor.group and target.tensor.isCached()) {
                            writer.print("t{d}->{s}{d}[label=\"{s}\"];\n", .{ target.tensor.uid, @tagName(node.op), node.out.tensor.uid, target.tensor.label });
                        } else {
                            writer.print("T{d}->{s}{d}[label=\"{s}\"];\n", .{ target.tensor.memoryView(), @tagName(node.op), node.out.tensor.uid, target.tensor.label });
                        }
                    }
                },
            }
        }
    };

    const visited = arena.allocator().alloc(bool, tensors.count()) catch unreachable;
    defer arena.allocator().free(visited);
    writer.print(
        \\digraph G {{
        \\compound=true;
        \\
    , .{});
    // TODO: Support for multiple entrypoints in the case of a DAG with multiple sinks
    for (dagSinks()) |entry| {
        Viz.vizHelper(.{ .tensor = entry }, visited);
    }
    writer.print("}}\n", .{});
}

pub const Fusion = struct {
    const FusionError = error{
        AfterReduce,
        MultipleReduce,
        ParentInit,
        ParentCached,
        NotParentChild,
        DoubleFuse,
    };

    pub fn verticalFusion(parent: *TensorNode, child: *TensorNode) FusionError!void {
        if (parent.group == null) {
            return FusionError.ParentInit;
        }
        if (parent.group == child.group) {
            return FusionError.DoubleFuse;
        }
        const parent_group_contains_reduction = (if (parent.group) |group| reduction_groups.get(group) orelse false else false);
        const child_group_contains_reduction = (if (child.group) |group| reduction_groups.get(group) orelse false else false);
        if (parent_group_contains_reduction and child_group_contains_reduction) {
            return FusionError.MultipleReduce;
        }

        switch (child.op_node) {
            .InitOp => unreachable, // Impossible as init op will only have a child (output) and no tensor input
            .BinaryOp => |*op_node| {
                if (op_node.a.tensor.uid != parent.uid and op_node.b.tensor.uid != parent.uid) {
                    return FusionError.NotParentChild;
                }
                if (op_node.a.tensor.uid == parent.uid) {
                    if (op_node.a.fused) return FusionError.DoubleFuse;
                    if (parent.isCached()) return FusionError.ParentCached;
                    op_node.a.fused = true;
                } else if (op_node.b.tensor.uid == parent.uid) {
                    if (op_node.b.fused) return FusionError.DoubleFuse;
                    if (parent.isCached()) return FusionError.ParentCached;
                    op_node.b.fused = true;
                }
            },
            .ReduceOp => |*op_node| {
                if (op_node.x.tensor.uid != parent.uid) return FusionError.NotParentChild;
                if (op_node.x.fused) return FusionError.DoubleFuse;
                if (parent.isCached()) return FusionError.ParentCached;
                op_node.x.fused = true;
                reduction_groups.put(child.group.?, true) catch unreachable;
            },
            inline else => |*op_node| {
                if (op_node.x.tensor.uid != parent.uid) return FusionError.NotParentChild;
                if (op_node.x.fused) return FusionError.DoubleFuse;
                if (parent.isCached()) return FusionError.ParentCached;
                op_node.x.fused = true;
            },
        }
        switch (parent.op_node) {
            .ReduceOp => {
                if (parent.group) |old_reduce_group| {
                    std.debug.assert(reduction_groups.remove(old_reduce_group));
                    reduction_groups.putNoClobber(child.group.?, true) catch unreachable;
                }
            },
            else => {},
        }

        if (parent.group != null) {
            parent.group = child.group;
        }
    }

    /// Recursive function to fuse every parent child pair when possible.
    /// Keeps track of group ids (i.e. kernels) to prevent multiple thread synchronization requiring operations
    /// (reductions) from being in the same kernel. This might change after further testing.
    fn greedyFusionHelper(node: *TensorNode) void {
        switch (node.op_node) {
            .UnaryOp => |*op_node| {
                verticalFusion(op_node.x.tensor, node) catch {};
                greedyFusionHelper(op_node.x.tensor);
                if (op_node.x.tensor.group != node.group) op_node.x.tensor.global = true;
            },
            .BinaryOp => |*op_node| {
                // Process the temporally closer input first
                const inputs: std.meta.Tuple(&[_]type{OpNode.Input} ** 2) = if (op_node.a.tensor.uid > op_node.b.tensor.uid) .{ op_node.a, op_node.b } else .{ op_node.b, op_node.a };
                verticalFusion(inputs[0].tensor, node) catch {};
                greedyFusionHelper(inputs[0].tensor);
                if (op_node.a.tensor.group != node.group) op_node.a.tensor.global = true;
                verticalFusion(inputs[1].tensor, node) catch {};
                greedyFusionHelper(inputs[1].tensor);
                if (op_node.b.tensor.group != node.group) op_node.b.tensor.global = true;
            },
            .ReduceOp => |*op_node| {
                verticalFusion(op_node.x.tensor, node) catch {};
                greedyFusionHelper(op_node.x.tensor);
                if (op_node.x.tensor.group != node.group) op_node.x.tensor.global = true;
            },
            .BufferOp => |*op_node| {
                verticalFusion(op_node.x.tensor, node) catch {};
                greedyFusionHelper(op_node.x.tensor);
                if (op_node.x.tensor.group != node.group) op_node.x.tensor.global = true;
            },
            // Init will happen outside a kernel unless it is a full init
            .InitOp => |op_node| {
                if (op_node.op != .Full) node.group = null;
            },
        }
    }

    /// Traverse the graph and group nodes into clusters (kernels/functions)
    /// Each cluster can have at most one reduce op, but any amount of other ops
    /// The reduce op will be the last op unless it is followed by a type op
    pub fn greedyFusion() void {
        for (dagSinks()) |entry| greedyFusionHelper(entry);
    }
};
