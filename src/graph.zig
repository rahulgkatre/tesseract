const std = @import("std");
const anytensor = @import("anytensor.zig").anytensor;
const Record = @import("record.zig").Record;

pub const ComputeNode = struct {
    // ComputeNode will wrap around Record, and there will be a hash map for *const Record -> ComputeNode
    // ComputeNode will also contain scheduling data such as:
    // https://tvm.apache.org/docs/reference/api/python/te.html?highlight=compute_at#tvm.te.Stage

    record: *const Record,
    data_node: *const DataNode,
    compute_location: ComputeLocation = .{ .Root = {} },

    const ComputeLocation = union(enum) {
        Root: void,
        At: *const DataNode,
        Inline: *const DataNode,
    };
};

pub const DataNode = struct {
    // DataNode will wrap around anytensor, and there will be a hash map for *const anytensor -> DataNode
    // DataNode will also contain scheduling data such as grouping, caching, axis information
    tensor: *const anytensor,
};

pub const Graph = struct {
    arena: std.heap.ArenaAllocator,
    ordinals: std.AutoArrayHashMap(*const Record, usize),
    comp_nodes: std.AutoArrayHashMap(*const Record, ComputeNode),
    data_nodes: std.AutoArrayHashMap(*const Record, DataNode),

    pub fn init(arena: std.heap.ArenaAllocator) !*Graph {
        var graph = try arena.allocator().create(Graph);
        graph.arena = arena;
        graph.ordinals = std.AutoArrayHashMap(usize, usize).init(arena.allocator());
        graph.comp_nodes = std.AutoArrayHashMap(usize, ComputeNode).init(arena.allocator());
        graph.data_nodes = std.AutoArrayHashMap(usize, DataNode).init(arena.allocator());
        return graph;
    }

    pub fn deinit(graph: *Graph) void {
        graph.arena.deinit();
    }

    pub fn trace(graph: *Graph, tensor: *const anytensor) !void {
        if (!graph.comp_nodes.contains(tensor.record)) {
            try graph.ordinals.putNoClobber(tensor.record, graph.ordinals.count());
            const data_node: DataNode = .{ .tensor = tensor };
            try graph.data_nodes.putNoClobber(tensor.record, data_node);
            const comp_node: ComputeNode = .{ .record = tensor.record, .data_node = graph.data_nodes.getPtr(tensor.record) };
            try graph.comp_nodes.putNoClobber(tensor.record, comp_node);
        }

        switch (tensor.record.*) {
            .TernaryOp => |ternary_op| {
                try graph.trace(ternary_op.a.tensor);
                try graph.trace(ternary_op.b.tensor);
                try graph.trace(ternary_op.c.tensor);
            },
            .BinaryOp => |binary_op| {
                try graph.trace(binary_op.a.tensor);
                try graph.trace(binary_op.b.tensor);
            },
            .InitOp => {},
            inline else => |unary_op| {
                try graph.trace(unary_op.a.tensor);
            },
        }
    }
};
