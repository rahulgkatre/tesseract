const std = @import("std");
const ops = @import("ops.zig");
const dtypes = @import("dtypes.zig");
const AnyTensor = tensor.AnyTensor;
const tensor = @import("tensor/tensor.zig");
const types = @import("tensor/types.zig");

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

pub const Graph = struct {
    arena: *std.heap.ArenaAllocator,
    compute_nodes: std.AutoArrayHashMap(*const AnyTensor, ComputeNode),
    data_nodes: std.AutoArrayHashMap(*const AnyTensor, DataNode),

    pub fn init(arena: *std.heap.ArenaAllocator) !*Graph {
        var graph = try arena.allocator().create(Graph);
        graph.arena = arena;
        graph.compute_nodes = std.AutoArrayHashMap(*const AnyTensor, ComputeNode).init(arena.allocator());
        graph.data_nodes = std.AutoArrayHashMap(*const AnyTensor, DataNode).init(arena.allocator());
        return graph;
    }

    pub fn deinit(graph: *Graph) void {
        graph.arena.deinit();
    }

    pub fn compute(graph: *Graph, t: *const AnyTensor) !void {
        try graph.data(t);
        try graph.compute_nodes.put(t, .{
            .output_node = graph.data_nodes.getPtr(t).?,
        });
    }

    pub fn data(graph: *Graph, t: *const AnyTensor) !void {
        try graph.data_nodes.put(t, .{ .tensor = t });
    }

    pub fn trace(graph: *Graph, comptime out: anytype, printBytecode: bool) !void {
        const out_any = comptime types.asTensor(out).toAnyTensor();
        switch (comptime out_any.instr.*) {
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
                    out_any.labels.name orelse "",
                    types.TensorTypeOf(out_any).ArrayType(),
                    @intFromPtr(out_any),
                    out_any.instr,
                });
            }
        }
    }
};
