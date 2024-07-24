const std = @import("std");
const ops = @import("ops.zig");
const dtypes = @import("dtypes.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;
const tensor = @import("tensor.zig");

pub const ComputeNode = struct {
    // ComputeNode will wrap around AnyTensor, and there will be a hash map for *const AnyTensor -> ComputeNode
    // ComputeNode will also contain scheduling data such as:
    // https://tvm.apache.org/docs/reference/api/python/te.html?highlight=compute_at#tvm.te.Stage

    output_node: *DataNode,
    compute_location: ComputeLocation = .{ .Root = {} },

    const ComputeLocation = union(enum) {
        Root: void,
        At: *const DataNode,
        Inline: void,
    };
};

pub const DataNode = struct {
    // DataNode will wrap around AnyTensor, and there will be a hash map for *const AnyTensor -> DataNode
    // DataNode will also contain scheduling data such as grouping, caching, axis information
    tensor: *const AnyTensor,
};

const Node = union(enum) {
    Data: DataNode,
    Compute: ComputeNode,
};

pub const Graph = struct {
    arena: *std.heap.ArenaAllocator,
    nodes: std.MultiArrayList(Node),
    compute_nodes: std.AutoArrayHashMap(*const AnyTensor, *const ComputeNode),
    data_nodes: std.AutoArrayHashMap(*const AnyTensor, *const DataNode),

    pub fn init(arena: *std.heap.ArenaAllocator) !*Graph {
        var graph = try arena.allocator().create(Graph);
        graph.arena = arena;
        graph.nodes = std.MultiArrayList(Node);
        graph.compute_nodes = std.AutoArrayHashMap(*const AnyTensor, ComputeNode).init(arena.allocator());
        graph.data_nodes = std.AutoArrayHashMap(*const AnyTensor, DataNode).init(arena.allocator());
        return graph;
    }

    pub fn deinit(graph: *Graph) void {
        graph.arena.deinit();
    }

    pub fn compute(graph: *Graph, t: *const AnyTensor) !void {
        try graph.data(t);
        const compute_node: ComputeNode = .{
            .output_node = graph.data_nodes.getPtr(t).?,
        };
        try graph.nodes.append(graph.arena, .{ .Compute = compute_node });
        try graph.compute_nodes.put(t, &graph.nodes.get(graph.nodes.len).Compute);
    }

    pub fn data(graph: *Graph, t: *const AnyTensor) !void {
        const data_node: DataNode = .{ .tensor = t };
        try graph.nodes.append(graph.arena, .{ .Data = data_node });
        try graph.data_nodes.put(t, &graph.nodes.get(graph.nodes.len).Data);
    }

    pub fn trace(graph: *Graph, comptime out: anytype, printBytecode: bool) !void {
        const out_any = comptime tensor.asTensor(out).toAnyTensor();
        switch (comptime out_any.meta.instr) {
            inline else => |instr| inline for (instr.in) |in| {
                const in_tensor = comptime @as(*const AnyTensor, in);
                if (!graph.compute_nodes.contains(in_tensor)) {
                    try graph.trace(in_tensor, printBytecode);
                }
            },
        }
        if (!graph.compute_nodes.contains(out_any)) {
            try graph.compute(out_any);
            if (printBytecode) {
                std.debug.print("{s: <32}  {:<24}@{x}  {any}\n", .{
                    out_any.meta.label orelse "",
                    tensor.TensorTypeOf(out_any).ArrayType(),
                    @intFromPtr(out_any),
                    out_any.meta.instr,
                });
            }
        }
    }
};
