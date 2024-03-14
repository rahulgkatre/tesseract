const std = @import("std");
const ops = @import("ops.zig");
const utils = @import("utils.zig");
const Graph = @This();
const dtypes = @import("dtypes.zig");

var gpa: std.heap.GeneralPurposeAllocator(.{}) = undefined;
var arena: std.heap.ArenaAllocator = undefined;
var tensors: std.AutoHashMap(usize, *TensorNode) = undefined;
var reduction_groups: std.AutoHashMap(usize, bool) = undefined;
var dag_sinks: std.AutoArrayHashMap(usize, *TensorNode) = undefined;

pub fn dagSinks() []*TensorNode {
    return dag_sinks.values();
}

pub fn init() void {
    gpa = .{};
    arena = std.heap.ArenaAllocator.init(gpa.allocator());
    tensors = std.AutoHashMap(usize, *TensorNode).init(arena.allocator());
    reduction_groups = std.AutoHashMap(usize, bool).init(arena.allocator());
    dag_sinks = std.AutoArrayHashMap(usize, *TensorNode).init(arena.allocator());
}

pub fn deinit() void {
    arena.deinit();
}

/// Build the computation graph for a tensor.
/// Any new nodes are added to the global computation graph
/// by recursively calling each tensor's `trace_fn` callback.
pub fn trace(tensor: anytype) void {
    switch (@typeInfo(@TypeOf(tensor))) {
        .Pointer => tensor.trace_fn(tensor),
        else => @compileError("Must pass a tensor pointer to Graph.trace()"),
    }
    const key = @intFromPtr(tensor);
    const node = TensorNode.get(key);
    node.global = true;
    dag_sinks.putNoClobber(key, node) catch unreachable;
}

/// Each tensor trace callback uses this function to add its dependencies (input), operation (op), and result (output)
/// to the computation graph
pub fn addOp(comptime op: ops.GraphOp, input: anytype, output: anytype, comptime args: anytype) void {
    _ = OpNode.init(op, input, output, args);
}

pub const OpNode = union(ops.OpTypes) {
    pub const Input = struct {
        tensor: *TensorNode,
        fused: bool = false,

        fn init(ptr: anytype) Input {
            const tensor = TensorNode.get(ptr);
            tensor.consumer_count += 1;
            if (tensor.consumer_count > 1) {
                tensor.cached = true;
            }
            // tensors.put(ptr, tensor);
            return .{
                .tensor = TensorNode.get(ptr),
            };
        }
    };
    pub const Output = struct {
        tensor: *TensorNode,
        fn init(ptr: anytype) Output {
            return .{
                .tensor = TensorNode.get(ptr),
            };
        }
    };
    pub const MapOp = struct {
        op: ops.MapOp,
        x: Input,
        out: Output,
        label: []const u8,
    };
    pub const ZipOp = struct {
        op: ops.ZipOp,
        a: Input,
        b: Input,
        out: Output,
        label: []const u8,
    };
    pub const ReduceOp = struct {
        op: ops.ReduceOp,
        x: Input,
        dims: []const bool,
        out: Output,
        label: []const u8,
    };
    pub const TypeOp = struct {
        op: ops.TypeOp,
        x: Input,
        out: Output,
        label: []const u8,
    };
    pub const InitOp = struct {
        op: ops.InitOp,
        args: ops.InitOp.Args,
        out: Output,
        label: []const u8,
    };
    MapOp: MapOp,
    ZipOp: ZipOp,
    ReduceOp: ReduceOp,
    TypeOp: TypeOp,
    InitOp: InitOp,

    fn init(comptime op: ops.GraphOp, input: anytype, output: anytype, comptime args: anytype) OpNode {
        const Out = @TypeOf(output.*);
        switch (op) {
            .MapOp, .TypeOp, .ReduceOp => {
                if (!tensors.contains(@intFromPtr(input.x))) input.x.trace_fn(input.x);
                TensorNode.create(input.x);
                TensorNode.create(output);
            },
            .ZipOp => {
                if (!tensors.contains(@intFromPtr(input.a))) input.a.trace_fn(input.a);
                if (!tensors.contains(@intFromPtr(input.b))) input.b.trace_fn(input.b);
                TensorNode.create(input.a);
                TensorNode.create(input.b);
                TensorNode.create(output);
            },
            else => {
                TensorNode.create(output);
            },
        }
        const out = TensorNode.get(output);
        out.op_node = switch (op) {
            .MapOp => |map_op| @unionInit(OpNode, @tagName(op), .{
                .op = map_op,
                .x = Input.init(input.x),
                .out = Output.init(output),
                .label = std.fmt.comptimePrint("{s}", .{@tagName(map_op)}),
            }),
            .ZipOp => |zip_op| @unionInit(OpNode, @tagName(op), .{
                .op = zip_op,
                .a = Input.init(input.a),
                .b = Input.init(input.b),
                .out = Output.init(output),
                .label = std.fmt.comptimePrint("{s}", .{@tagName(zip_op)}),
            }),
            .ReduceOp => |reduce_op| blk: {
                reduction_groups.put(out.group.?, true) catch unreachable;
                break :blk @unionInit(OpNode, @tagName(op), .{
                    .op = reduce_op,
                    .x = Input.init(input.x),
                    .out = Output.init(output),
                    .dims = args.dims,
                    .label = std.fmt.comptimePrint("{s}{any}", .{ @tagName(reduce_op), @as([]const bool, args.dims) }),
                });
            },
            .TypeOp => |type_op| @unionInit(OpNode, @tagName(op), .{
                .op = type_op,
                .x = Input.init(input.x),
                .label = switch (type_op) {
                    .AsType => std.fmt.comptimePrint("{s}{any}", .{ @tagName(type_op), Out.dtype }),
                    .View, .Broadcast => std.fmt.comptimePrint("{s}{any}", .{ @tagName(type_op), Out.shape }),
                    .AsStrided => std.fmt.comptimePrint("{s}{{{any},{any}}}", .{ @tagName(type_op), Out.shape, Out.strides }),
                },
                .out = Output.init(output),
            }),
            .InitOp => |init_op| blk: {
                if (init_op == .Input) {
                    out.group = null;
                    out.global = true;
                }
                break :blk @unionInit(OpNode, @tagName(op), .{
                    .op = init_op,
                    .args = args,
                    .out = Output.init(output),
                    .label = std.fmt.comptimePrint("{s}", .{@tagName(init_op)}),
                });
            },
        };
        return out.op_node;
    }
};

pub const TensorNode = struct {
    ptr: usize,
    dtype: dtypes.DType,
    ndims: u8,
    shape: []const u64,
    strides: []const u64,

    label: []const u8,
    uid: u64,
    group: ?u64 = null,
    consumer_count: u16 = 0,
    cached: bool = false,
    global: bool = false,
    op_node: OpNode = undefined,

    const JsonCompatibleTensorNode = struct {
        dtype: dtypes.DType,
        ndims: u8,
        shape: []const u64,
        strides: []const u64,
        uid: u64,
        mem_id: u64,
        group: ?usize = null,
        cached: bool = false,
        global: bool = false,
    };

    pub fn jsonStringify(self: *const TensorNode, write_stream: anytype) !void {
        const compatible: JsonCompatibleTensorNode = .{
            .dtype = self.dtype,
            .ndims = self.ndims,
            .shape = self.shape,
            .strides = self.strides,
            .uid = self.uid,
            .mem_id = self.memoryView(),
            .group = self.group,
            .cached = self.cached,
            .global = self.global,
        };
        try write_stream.write(compatible);
    }

    pub fn uniqueId(self: *const TensorNode) u64 {
        return self.uid;
    }

    const MemoryViewError = error{Immutable};

    fn memoryViewHelper(self: *const TensorNode) MemoryViewError!u64 {
        return switch (self.op_node) {
            .InitOp => self.uid,
            .ZipOp => |op_node| blk: {
                if (!op_node.a.fused and !op_node.b.fused) {
                    return self.uid;
                }

                const a_mv = op_node.a.tensor.memoryViewHelper() catch {
                    break :blk op_node.b.tensor.memoryViewHelper() catch self.uid;
                };

                const b_mv = op_node.b.tensor.memoryViewHelper() catch {
                    break :blk op_node.a.tensor.memoryViewHelper() catch self.uid;
                };

                if (op_node.a.fused and !op_node.b.fused) {
                    break :blk a_mv;
                } else if (op_node.b.fused and !op_node.a.fused) {
                    break :blk b_mv;
                } else {
                    break :blk @min(a_mv, b_mv);
                }
            },
            inline else => |op_node| if (op_node.x.fused) op_node.x.tensor.memoryViewHelper() catch self.uid else self.uid,
        };
    }

    pub fn memoryView(self: *const TensorNode) u64 {
        return self.memoryViewHelper() catch self.uid;
    }

    pub fn get(ptr: anytype) *TensorNode {
        return switch (@typeInfo(@TypeOf(ptr))) {
            .Pointer => tensors.get(@intFromPtr(ptr)).?,
            .Int => tensors.get(ptr).?,
            else => @panic("Must use a tensor pointer or int from tensor pointer"),
        };
    }

    fn create(tensor: anytype) void {
        const Tensor = @TypeOf(tensor.*);
        const ptr = @intFromPtr(tensor);
        if (!tensors.contains(ptr)) {
            const tensor_node = arena.allocator().create(TensorNode) catch unreachable;
            tensor_node.* = .{
                .ptr = ptr,
                .dtype = Tensor.dtype,
                .ndims = Tensor.ndims,
                .shape = Tensor.shape[0..Tensor.ndims],
                .strides = Tensor.strides[0 .. Tensor.ndims + 1],
                .group = tensors.count(),
                .uid = tensors.count(),
                .label = std.fmt.comptimePrint("{s}{any}", .{ @tagName(Tensor.dtype), Tensor.shape }),
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
                .ZipOp => |binary_op_node| {
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
                    if (tensor.cached and tensor.group != null) {
                        writer.print("subgraph cluster{d}{{t{d}[label=\"t{d}\"shape=box];}}\n", .{ tensor.group.?, tensor.uid, tensor.uid });
                        if (tensor.global) {
                            writer.print("t{d}->T{d}[label=\"{s}\"];\n", .{ tensor.uid, tensor.memoryView(), tensor.label });
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

                    if (target.fused) {
                        switch (target.tensor.op_node) {
                            inline else => |in_op_node| writer.print("{s}{d}->{s}{d}[label=\"{s}\"];\n", .{ @tagName(in_op_node.op), target.tensor.uid, @tagName(node.op), node.out.tensor.uid, target.tensor.label }),
                        }
                    } else {
                        if (node.out.tensor.group != null and target.tensor.group == node.out.tensor.group and target.tensor.cached) {
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
            .ZipOp => |op_node| {
                if (op_node.a.tensor.uid != parent.uid and op_node.b.tensor.uid != parent.uid) {
                    return FusionError.NotParentChild;
                }
                if (op_node.a.tensor.uid == parent.uid) {
                    if (!parent.cached) child.op_node.ZipOp.a.fused = true;
                } else if (op_node.b.tensor.uid == parent.uid) {
                    if (!parent.cached) child.op_node.ZipOp.b.fused = true;
                }
            },
            .TypeOp => |op_node| {
                // Fuse a TypeOp even when the previous op is a reduce op
                // The only type op that has any loop is a cast, and that is trivial enough to inline
                if (op_node.x.tensor.uid != parent.uid) {
                    return FusionError.NotParentChild;
                }
                if (!parent.cached) child.op_node.TypeOp.x.fused = true;
            },
            .MapOp => |op_node| {
                if (op_node.x.tensor.uid != parent.uid) {
                    return FusionError.NotParentChild;
                }
                if (!parent.cached) child.op_node.MapOp.x.fused = true;
            },
            .ReduceOp => |op_node| {
                if (op_node.x.tensor.uid != parent.uid) {
                    return FusionError.NotParentChild;
                }

                if (!parent.cached) {
                    child.op_node.ReduceOp.x.fused = true;
                }
                reduction_groups.put(child.group.?, true) catch unreachable;
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
            .MapOp => |*op_node| {
                verticalFusion(op_node.x.tensor, node) catch {};
                greedyFusionHelper(op_node.x.tensor);
                if (op_node.x.tensor.group != node.group) {
                    op_node.x.tensor.global = true;
                }
            },
            .ZipOp => |*op_node| {
                // Process the closer input first for better locality
                const inputs: std.meta.Tuple(&[_]type{OpNode.Input} ** 2) = if (op_node.a.tensor.uid > op_node.b.tensor.uid) .{ op_node.a, op_node.b } else .{ op_node.b, op_node.a };
                verticalFusion(inputs[0].tensor, node) catch {};
                greedyFusionHelper(inputs[0].tensor);
                if (op_node.a.tensor.group != node.group) {
                    op_node.a.tensor.global = true;
                }
                verticalFusion(inputs[1].tensor, node) catch {};
                greedyFusionHelper(inputs[1].tensor);
                if (op_node.b.tensor.group != node.group) {
                    op_node.b.tensor.global = true;
                }
            },
            .ReduceOp => |*op_node| {
                verticalFusion(op_node.x.tensor, node) catch {};
                greedyFusionHelper(op_node.x.tensor);
                if (op_node.x.tensor.group != node.group) {
                    op_node.x.tensor.global = true;
                }
            },
            .TypeOp => |*op_node| {
                // TypeOps can always be fused into the preceding kernel even if the typeop follows a reduce
                // This is because it is either just index manipulation and does not produce a loop
                // or it is a cast which can be inlined when assigning the value in the output tensor
                verticalFusion(op_node.x.tensor, node) catch {};
                greedyFusionHelper(op_node.x.tensor);
                if (op_node.x.tensor.group != node.group) {
                    op_node.x.tensor.global = true;
                }
            },
            // Init will happen outside a kernel unless it is a full init
            .InitOp => |op_node| {
                if (op_node.op != .Full) {
                    node.group = null;
                }
            },
        }
    }

    /// Traverse the graph and group nodes into clusters (kernels/functions)
    /// Each cluster can have at most one reduce op, but any amount of other ops
    /// The reduce op will be the last op unless it is followed by a type op
    pub fn greedyFusion() void {
        for (dagSinks()) |entry| {
            greedyFusionHelper(entry);
        }
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

    // Graph.init(std.testing.arena.allocator());
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
    const tensor = @import("tensor.zig");
    const out = comptime blk: {
        const a = tensor.InferredStrides(.f32, .{ 1024, 2048 }).full(2);
        const b = tensor.InferredStrides(.f32, .{ 2048, 4096 }).full(3);
        break :blk a.matmul(b);
    };
    Graph.init();
    defer Graph.deinit();
    Graph.trace(&out);
    // Graph.Fusion.applyGreedyFusion();
    // std.debug.print("\n", .{});
    // try Graph.viz(std.debug);
}
