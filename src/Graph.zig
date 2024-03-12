const std = @import("std");
const ops = @import("ops.zig");
const utils = @import("utils.zig");
const Graph = @This();
const dtypes = @import("dtypes.zig");

var gpa: std.heap.GeneralPurposeAllocator(.{}) = undefined;
var arena: std.heap.ArenaAllocator = undefined;
var tensors: std.AutoHashMap(usize, *TensorNode) = undefined;
var reduction_groups: std.AutoHashMap(usize, bool) = undefined;
var entry_tensor: ?*TensorNode = null;

pub fn entry() *TensorNode {
    return entry_tensor orelse @panic("Graph has no entrypoint. Remember to call trace() on an output tensor pointer");
}

pub fn init() void {
    gpa = .{};
    arena = std.heap.ArenaAllocator.init(gpa.allocator());
    tensors = std.AutoHashMap(usize, *TensorNode).init(arena.allocator());
    reduction_groups = std.AutoHashMap(usize, bool).init(arena.allocator());
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
    entry_tensor = TensorNode.get(@intFromPtr(tensor));
}

/// Each tensor trace callback uses this function to add its dependencies (input), operation (op), and result (output)
/// to the computation graph
pub fn addOp(comptime op: ops.GraphOp, input: anytype, output: anytype, comptime options: anytype) void {
    _ = OpNode.getOrInit(op, input, output, options);
}

pub const OpNode = union(ops.OpTypes) {
    pub const Input = struct {
        tensor: *TensorNode,
        fused: bool = false,

        fn init(ptr: anytype) Input {
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
        options: ops.InitOp.Options,
        out: Output,
        label: []const u8,
    };
    MapOp: MapOp,
    ZipOp: ZipOp,
    ReduceOp: ReduceOp,
    TypeOp: TypeOp,
    InitOp: InitOp,

    fn viz(self: OpNode, target: OpNode.Input, writer: anytype) void {
        switch (self) {
            inline else => |op_node| {
                if (op_node.out.tensor.group != null) {
                    writer.print("subgraph cluster{d}{{{s}{d}[label=\"{d}: {s}\"];}}\n", .{ op_node.out.tensor.group.?, @tagName(op_node.op), op_node.out.tensor.uniqueId(), op_node.out.tensor.uniqueId(), op_node.label });
                } else {
                    writer.print("{s}{d}[label=\"{d}: {s}\"];\n", .{ @tagName(op_node.op), op_node.out.tensor.uniqueId(), op_node.out.tensor.uniqueId(), op_node.label });
                }
            },
        }
        switch (self) {
            .InitOp => {}, // InitOp will not have a previous tensor node to connect to
            inline else => |op_node| {
                if (target.fused) {
                    switch (target.tensor.op_node) {
                        inline else => |in_op_node| writer.print("{s}{d}->{s}{d}[label=\"{s}\"];\n", .{ @tagName(in_op_node.op), target.tensor.uid, @tagName(op_node.op), op_node.out.tensor.uniqueId(), target.tensor.label }),
                    }
                } else {
                    if (op_node.out.tensor.group != null and target.tensor.group == op_node.out.tensor.group and target.tensor.cached) {
                        writer.print("T{d}_{?}->{s}{d}[label=\"{s}\"];\n", .{ target.tensor.memId(), target.tensor.group, @tagName(op_node.op), op_node.out.tensor.uniqueId(), target.tensor.label });
                    } else {
                        writer.print("T{d}->{s}{d}[label=\"{s}\"];\n", .{ target.tensor.memId(), @tagName(op_node.op), op_node.out.tensor.uniqueId(), target.tensor.label });
                    }
                }
            },
        }
    }

    fn getOrInit(comptime op: ops.GraphOp, input: anytype, output: anytype, comptime options: anytype) OpNode {
        const Out = @TypeOf(output.*);
        switch (op) {
            .MapOp, .TypeOp, .ReduceOp => {
                trace(input.x);
                TensorNode.create(input.x);
                TensorNode.create(output);
            },
            .ZipOp => {
                trace(input.a);
                TensorNode.create(input.a);
                trace(input.b);
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
            .ReduceOp => |reduce_op| @unionInit(OpNode, @tagName(op), .{
                .op = reduce_op,
                .x = Input.init(input.x),
                .out = Output.init(output),
                .dims = options.dims,
                .label = std.fmt.comptimePrint("{s}{any}", .{ @tagName(reduce_op), @as([]const bool, options.dims) }),
            }),
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
            .InitOp => |init_op| @unionInit(OpNode, @tagName(op), .{
                .op = init_op,
                .options = options,
                .out = Output.init(output),
                .label = std.fmt.comptimePrint("{s}", .{@tagName(init_op)}),
            }),
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
    group: ?usize = null,
    cached: bool = false,
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
    };

    pub fn jsonStringify(self: *const TensorNode, write_stream: anytype) !void {
        const compatible: JsonCompatibleTensorNode = .{
            .dtype = self.dtype,
            .ndims = self.ndims,
            .shape = self.shape,
            .strides = self.strides,
            .uid = self.uid,
            .mem_id = self.memId(),
            .group = self.group,
            .cached = self.cached,
        };
        try write_stream.write(compatible);
    }

    pub fn uniqueId(self: *const TensorNode) u64 {
        return self.uid;
    }

    pub fn memId(self: *const TensorNode) u64 {
        return switch (self.op_node) {
            .InitOp => self.uid,
            .ZipOp => |op_node| blk: {
                if (op_node.a.fused and !op_node.b.fused) {
                    break :blk op_node.a.tensor.memId();
                } else if (op_node.b.fused and !op_node.a.fused) {
                    break :blk op_node.b.tensor.memId();
                } else {
                    break :blk self.uid;
                }
            },
            inline else => |op_node| if (op_node.x.fused) op_node.x.tensor.memId() else self.uid,
        };
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
                .uid = tensors.count(),
                .label = std.fmt.comptimePrint("{s}{any}", .{ @tagName(Tensor.dtype), Tensor.shape }),
            };
            tensors.putNoClobber(ptr, tensor_node) catch unreachable;
        }
    }

    fn viz(tensor: *const TensorNode, writer: anytype, visited: []bool) void {
        // To avoid printing the same thing multiple times use the table to check/mark as already printed
        if (visited[tensor.uid]) {
            return;
        }
        switch (tensor.op_node) {
            inline else => |op_node| {
                writer.print("T{d}[label=\"T{d}\"shape=box];\n", .{ tensor.memId(), tensor.memId() });
                if (tensor.cached) {
                    writer.print("subgraph cluster{d}{{T{d}_{d}[label=\"T{d}_{d}\"shape=box];}}\n", .{ tensor.group.?, tensor.memId(), tensor.group.?, tensor.memId(), tensor.group.? });
                    writer.print("T{d}_{d}->T{d}[label=\"{s}\"];\n", .{ tensor.memId(), tensor.group.?, tensor.memId(), tensor.label });
                    writer.print("{s}{d}->T{d}_{d}[label=\"{s}\"];\n", .{ @tagName(op_node.op), tensor.uid, tensor.memId(), tensor.group.?, tensor.label });
                } else {
                    writer.print("{s}{d}->T{d}[label=\"{s}\"];\n", .{ @tagName(op_node.op), tensor.uid, tensor.memId(), tensor.label });
                }
            },
        }
        visited[tensor.uid] = true;
    }
};

fn vizHelper(target: OpNode.Input, writer: anytype, visited: []bool) void {
    if (visited[target.tensor.uid]) {
        return;
    }
    const op_node = target.tensor.op_node;
    // Recursive calls
    switch (op_node) {
        .InitOp => op_node.viz(.{ .tensor = undefined }, writer), // the undefined tensor field is never accessed for an init op
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
    if (!target.fused) {
        target.tensor.viz(writer, visited);
    }
}

pub fn viz(writer: anytype) void {
    const visited = arena.allocator().alloc(bool, tensors.count()) catch unreachable;
    defer arena.allocator().free(visited);
    writer.print(
        \\digraph G {{
        \\compound=true;
        \\
    , .{});
    // TODO: Support for multiple entrypoints in the case of a DAG with multiple sinks
    vizHelper(.{ .tensor = entry() }, writer, visited);
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
        const op_node = child.op_node;
        switch (op_node) {
            .InitOp => unreachable, // Impossible as init op will only have a child (output) and no tensor input
            .ZipOp => |zip_op_node| {
                if (zip_op_node.a.tensor.memId() != parent.memId() and zip_op_node.b.tensor.memId() != parent.memId()) {
                    return FusionError.NotParentChild;
                }
                if (zip_op_node.a.tensor.memId() == parent.memId()) {
                    if (parent_group_contains_reduction) {
                        return FusionError.ParentReduce;
                    }
                    child.op_node.ZipOp.a.fused = true;
                } else {
                    if (parent_group_contains_reduction) {
                        return FusionError.ParentReduce;
                    }
                    child.op_node.ZipOp.b.fused = true;
                }
            },
            .TypeOp => |type_op_node| {
                // Fuse a TypeOp even when the previous op is a reduce op
                // The only type op that has any loop is a cast, and that is trivial enough to inline
                if (type_op_node.x.tensor.memId() != parent.memId()) {
                    return FusionError.NotParentChild;
                }
                child.op_node.TypeOp.x.fused = true;
            },
            .MapOp => |map_op_node| {
                if (map_op_node.x.tensor.memId() != parent.memId()) {
                    return FusionError.NotParentChild;
                }
                if (parent_group_contains_reduction) {
                    return FusionError.ParentReduce;
                }
                child.op_node.MapOp.x.fused = true;
            },
            .ReduceOp => |reduce_op_node| {
                if (reduce_op_node.x.tensor.memId() != parent.memId()) {
                    return FusionError.NotParentChild;
                }
                if (parent_group_contains_reduction) {
                    return FusionError.ParentReduce;
                }
                child.op_node.ReduceOp.x.fused = true;
            },
        }
        std.debug.print("fused {d} and {d}\n", .{ parent.uniqueId(), child.uniqueId() });
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
            return node.group.?;
        }
        switch (node.op_node) {
            .MapOp => |op_node| {
                node.group = greedyFusionHelper(group, op_node.x.tensor);
                if (op_node.x.tensor.group == node.group and !op_node.x.tensor.cached) {
                    verticalFusion(op_node.x.tensor, node) catch {
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
                node.group = greedyFusionHelper(greedyFusionHelper(group, op_node.a.tensor), op_node.b.tensor);
                if (op_node.a.tensor.group == node.group and !op_node.a.tensor.cached) {
                    verticalFusion(op_node.a.tensor, node) catch {
                        node.group = node.group.? + 1;
                    };
                }
                if (op_node.b.tensor.group == node.group and !op_node.b.tensor.cached) {
                    verticalFusion(op_node.b.tensor, node) catch {
                        node.group = node.group.? + 1;
                    };
                }
                while (reduction_groups.get(node.group.?) orelse false) {
                    node.group = node.group.? + 1;
                }
            },
            .ReduceOp => |op_node| {
                node.group = greedyFusionHelper(group, op_node.x.tensor);
                if (op_node.x.tensor.group == node.group and !op_node.x.tensor.cached) {
                    verticalFusion(op_node.x.tensor, node) catch {
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
                node.group = greedyFusionHelper(group, op_node.x.tensor);
                if (op_node.x.tensor.group == node.group and !op_node.x.tensor.cached) {
                    verticalFusion(op_node.x.tensor, node) catch {
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
