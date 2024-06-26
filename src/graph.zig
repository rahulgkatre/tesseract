const std = @import("std");
const dtypes = @import("dtypes.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;

pub const ComputeNode = struct {
    // ComputeNode will wrap around AnyTensor, and there will be a hash map for *const AnyTensor -> ComputeNode
    // ComputeNode will also contain scheduling data such as:
    // https://tvm.apache.org/docs/reference/api/python/te.html?highlight=compute_at#tvm.te.Stage

    output_node: *const DataNode,
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
    ordinals: std.AutoArrayHashMap(*const AnyTensor, usize),
    comp_nodes: std.AutoArrayHashMap(*const AnyTensor, ComputeNode),
    data_nodes: std.AutoArrayHashMap(*const AnyTensor, DataNode),
    node_consumers: std.AutoArrayHashMap(*const AnyTensor, std.AutoArrayHashMap(*const AnyTensor, void)),

    pub fn init(arena: *std.heap.ArenaAllocator) !*Graph {
        var graph = try arena.allocator().create(Graph);
        graph.arena = arena;
        graph.ordinals = std.AutoArrayHashMap(*const AnyTensor, usize).init(arena.allocator());
        graph.comp_nodes = std.AutoArrayHashMap(*const AnyTensor, ComputeNode).init(arena.allocator());
        graph.data_nodes = std.AutoArrayHashMap(*const AnyTensor, DataNode).init(arena.allocator());
        graph.node_consumers = std.AutoArrayHashMap(*const AnyTensor, std.AutoArrayHashMap(*const AnyTensor, void)).init(arena.allocator());
        return graph;
    }

    pub fn deinit(graph: *Graph) void {
        graph.arena.deinit();
    }

    fn addConsumer(graph: *Graph, producer: *const AnyTensor, consumer: *const AnyTensor) !void {
        if (graph.node_consumers.getPtr(producer)) |consumers| {
            try consumers.put(consumer, {});
        } else {
            var consumers = std.AutoArrayHashMap(*const AnyTensor, void).init(graph.arena.allocator());
            try consumers.put(consumer, {});
            try graph.node_consumers.put(producer, consumers);
        }
    }

    pub fn trace(graph: *Graph, tensor: *const AnyTensor) !void {
        if (graph.comp_nodes.contains(tensor)) {
            return;
        }

        try graph.ordinals.putNoClobber(tensor, graph.ordinals.count());
        const data_node: DataNode = .{ .tensor = tensor };
        try graph.data_nodes.putNoClobber(tensor, data_node);
        // Inline the computation if it is a literal (comptime int/float)
        const comp_node: ComputeNode = .{ .output_node = graph.data_nodes.getPtr(tensor).?, .compute_location = if (dtypes.isLiteral(tensor.dtype)) .Inline else .Root };
        try graph.comp_nodes.putNoClobber(tensor, comp_node);

        switch (tensor.op_tracker.*) {
            .TernaryOp => |opt| {
                try graph.trace(opt.in[0]);
                try graph.addConsumer(opt.in[0], tensor);
                try graph.trace(opt.in[1]);
                try graph.addConsumer(opt.in[1], tensor);
                try graph.trace(opt.in[2]);
                try graph.addConsumer(opt.in[2], tensor);
            },
            .BinaryOp => |opt| {
                try graph.trace(opt.in[0]);
                try graph.addConsumer(opt.in[0], tensor);
                try graph.trace(opt.in[1]);
                try graph.addConsumer(opt.in[1], tensor);
            },
            .InitOp => {},
            inline else => |opt| {
                try graph.trace(opt.in[0]);
                try graph.addConsumer(opt.in[0], tensor);
            },
        }
    }

    /// Tensors with exactly 1 consumer will be inlined every
    pub fn inlineSingleConsumers(graph: *Graph) void {
        for (graph.node_consumers.keys()) |producer| {
            if (graph.node_consumers.get(producer).?.keys().len == 1) {
                graph.comp_nodes.getPtr(producer).?.compute_location = .{ .Inline = {} };
            }
        }
    }
};
