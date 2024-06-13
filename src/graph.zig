const std = @import("std");
const ops = @import("ops.zig");
const dtypes = @import("dtypes.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;

pub const ComputeNode = struct {
    // ComputeNode will wrap around AnyTensor, and there will be a hash map for *const AnyTensor -> ComputeNode
    // ComputeNode will also contain scheduling data such as:
    // https://tvm.apache.org/docs/reference/api/python/te.html?highlight=compute_at#tvm.te.Stage

    output_node: *DataNode,
    compute_instr: ops.Instruction,
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

    pub fn compute(graph: *Graph, tensor: *const AnyTensor, instr: ops.Instruction) !void {
        if (!graph.compute_nodes.contains(tensor)) {
            try graph.data(tensor);
            try graph.compute_nodes.put(tensor, .{
                .output_node = graph.data_nodes.getPtr(tensor).?,
                .compute_instr = instr,
            });
            std.debug.print("{s: <32} %{x} = {any}\n", .{
                tensor.meta.label orelse " ",
                @intFromPtr(tensor),
                tensor.meta.instr,
            });
        }
    }

    pub fn data(graph: *Graph, tensor: *const AnyTensor) !void {
        if (!graph.data_nodes.contains(tensor)) {
            try graph.data_nodes.put(tensor, .{ .tensor = tensor });
        }
    }

    pub fn traceForward(graph: *Graph, comptime out: anytype) !void {
        const forwardImpl: *const fn (comptime anytype, *Graph) anyerror!void = comptime @ptrCast(out.meta.forward);
        try forwardImpl(out, graph);
    }

    pub fn traceBackward(graph: *Graph, comptime grad_out: anytype) !void {
        const getBackwardReturnType: *const fn () type = comptime @ptrCast(grad_out.meta.backwardReturnType);
        const backwardImpl: *const fn (comptime anytype) getBackwardReturnType() = comptime @ptrCast(grad_out.meta.backward);
        try backwardImpl(grad_out, graph);
    }
};
