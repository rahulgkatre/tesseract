const ops = @import("ops.zig");
const std = @import("std");
const codegen = @import("codegen.zig");
const Graph = @import("Graph.zig");
const Program = @This();

var arena: std.heap.ArenaAllocator = undefined;
var allocator: std.mem.Allocator = undefined;
var loop_hash_table: std.AutoArrayHashMap(usize, *Loop) = undefined;
var subprograms: std.ArrayList(*Program) = undefined;

body: Body = undefined,

pub fn init(backing_allocator: std.mem.Allocator) void {
    arena = std.heap.ArenaAllocator.init(backing_allocator);
    allocator = arena.allocator();
    loop_hash_table = std.AutoArrayHashMap(usize, *Loop).init(allocator);
    subprograms = std.ArrayList(*Program).init(allocator);
}

pub fn fromGraph() !void {
    var p = try new();
    p.loops(Graph.entrypoint.?);
}

fn new() !*Program {
    const program = try allocator.create(Program);
    program.body = .{
        .contents = std.MultiArrayList(Body.Content){},
    };
    return program;
}

pub fn deinit() void {
    loop_hash_table.deinit();
    loop_hash_table = undefined;
    allocator = undefined;
    arena.deinit();
    arena = undefined;
}

/// Lower the node (and any nodes fused with it)
/// to a loop nest representation
pub fn loops(program: *Program, v: *Graph.Vertex) !*Loop {
    if (loop_hash_table.contains(v.id)) {
        return loop_hash_table.get(v.id).?;
    }
    const statement: Statement = switch (v.edge) {
        .InitOp => |edge| .{ .InitOp = .{ .id = 0, .op = edge.op, .out = v, .init = edge.value } },
        .ZipOp => |edge| .{ .ZipOp = .{ .id = 0, .op = edge.op, .a = edge.a, .b = edge.b, .out = v } },
        .MapOp => |edge| .{ .MapOp = .{ .id = 0, .op = edge.op, .x = edge.x, .out = v } },
        .ReduceOp => |edge| .{ .ReduceOp = .{ .id = 0, .op = edge.op, .x = edge.x, .out = v } },
        .TypeOp => |edge| .{ .TypeOp = .{ .id = 0, .op = edge.op, .x = edge.x, .out = v } },
    };
    var loop: *Loop = build_loop: {
        var reg_loops = std.ArrayList(*Loop).init(allocator);
        var acc_loops = std.ArrayList(*Loop).init(allocator);
        var curr_loop = try allocator.create(Loop);
        for (0..v.tensor.ndims) |d| {
            curr_loop.* = .{
                .upper_bound = switch (v.edge) {
                    .ReduceOp => |edge| edge.x.tensor.shape[d],
                    else => v.tensor.shape[d],
                },
                .node = v,
                .dim = d,
                .acc = switch (v.edge) {
                    .ReduceOp => |edge| edge.dims[d],
                    else => false,
                },
                .body = .{
                    .contents = std.MultiArrayList(Body.Content){},
                },
                .prev = null,
            };
            if (curr_loop.acc) {
                try acc_loops.append(curr_loop);
            } else {
                try reg_loops.append(curr_loop);
            }
            if (d != v.tensor.ndims - 1) {
                const inner_loop: *Loop = try allocator.create(Loop);
                curr_loop = inner_loop;
            }
        }

        // try curr_loop.body.contents.append(allocator, .{ .Loop = inner_loop });
        const root_loop: *Loop = reg_loops.items[0];
        curr_loop = root_loop;
        for (reg_loops.items[1..]) |reg_loop| {
            try curr_loop.body.contents.append(allocator, .{ .Loop = reg_loop });
            curr_loop = reg_loop;
        }
        for (acc_loops.items) |acc_loop| {
            try curr_loop.body.contents.append(allocator, .{ .Loop = acc_loop });
            curr_loop = acc_loop;
        }
        try curr_loop.body.contents.append(allocator, .{ .Statement = statement });
        break :build_loop root_loop;
    };
    switch (v.edge) {
        .InitOp => {},
        .ZipOp => |edge| {
            if (edge.a.group_id == edge.b.group_id and edge.b.group_id == v.group_id) {
                var a_loop = program.loops(edge.a) catch unreachable;
                a_loop.prev = program.loops(edge.b) catch unreachable;
                loop.prev = a_loop;
            } else {
                if (edge.a.group_id == v.group_id) {
                    loop.prev = program.loops(edge.a) catch unreachable;
                } else if (edge.b.group_id == v.group_id) {
                    loop.prev = program.loops(edge.b) catch unreachable;
                }
            }
        },
        .TypeOp => |edge| {
            switch (edge.op) {
                .AsType => {
                    loop.prev = program.loops(edge.x) catch unreachable;
                },
                else => {
                    loop.* = (program.loops(edge.x) catch unreachable).*;
                },
            }
        },
        inline else => |edge| {
            if (edge.x.group_id == v.group_id) {
                loop.prev = program.loops(edge.x) catch unreachable;
            }
        },
    }
    if (!loop_hash_table.contains(loop.node.id)) {
        try program.body.contents.append(allocator, .{ .Loop = loop });
    }

    try loop_hash_table.put(loop.node.id, loop);
    return loop;
}

/// Statement of the form y = f(x)
/// y can either be a value in an array or a variable
/// f(x) is an arithmetic operation on a value in an array or a variable
pub const Statement = union(ops.GraphOps) {
    MapOp: struct {
        id: usize,
        op: ops.MapOp,
        x: *Graph.Vertex,
        out: *Graph.Vertex,
    },
    ZipOp: struct {
        id: usize,
        op: ops.ZipOp,
        a: *Graph.Vertex,
        b: *Graph.Vertex,
        out: *Graph.Vertex,
    },
    ReduceOp: struct {
        id: usize,
        op: ops.ReduceOp,
        x: *Graph.Vertex,
        out: *Graph.Vertex,
    },
    TypeOp: struct {
        id: usize,
        op: ops.TypeOp,
        x: *Graph.Vertex,
        out: *Graph.Vertex,
    },
    InitOp: struct {
        id: usize,
        op: ops.InitOp,
        out: *Graph.Vertex,
        init: Graph.Edge.InitOp.InitValue,
    },
};

/// Abstractions for lowering Graph.Node into a loop which can be codegened
/// loop structs will be stored in a list (program) where order is exact order of code
/// loops are defined as a grammar, every loop has a header and a body
pub const Loop = struct {
    upper_bound: usize,
    node: *Graph.Vertex,
    dim: usize,
    acc: bool = false,
    body: Body,
    prev: ?*Loop,
};

pub const Body = struct {
    pub const Content = union(enum) {
        Loop: *Loop,
        Statement: Statement,
    };
    contents: std.MultiArrayList(Content),
};

test "codegen" {
    const tensor = @import("tensor.zig");
    const Zig = @import("codegen/Zig.zig");
    const out = comptime blk: {
        const a = tensor.InferredStrides(.f32, .{ 1024, 2048 }).full(2);
        const b = tensor.InferredStrides(.f32, .{ 2048, 4096 }).full(3);
        break :blk a.matmul(b);
    };
    Graph.init(std.testing.allocator);
    defer Graph.deinit();
    Graph.trace(out);
    Graph.applyGreedyFusion();
    std.debug.print("\n", .{});

    Program.init(std.testing.allocator);
    defer Program.deinit();
    var program = try Program.new();
    _ = try program.loops(Graph.entrypoint.?);

    Zig.init(std.testing.allocator);
    defer Zig.deinit();
    try Zig.bodyCode(program.body, std.debug);
    std.debug.print("\n", .{});
}
