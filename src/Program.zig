const ops = @import("ops.zig");
const std = @import("std");
const codegen = @import("codegen.zig");
const Graph = @import("Graph.zig");
const Program = @This();

var arena: std.heap.ArenaAllocator = undefined;
var allocator: std.mem.Allocator = undefined;
var loop_hash_table: std.AutoArrayHashMap(usize, *Loop) = undefined;
var statement_hash_table: std.AutoArrayHashMap(*Graph.Vertex, *Statement) = undefined;
var subprograms: std.ArrayList(*Program) = undefined;

group_id: ?usize = null,
body: Body = undefined,

pub fn init(backing_allocator: std.mem.Allocator) void {
    arena = std.heap.ArenaAllocator.init(backing_allocator);
    allocator = arena.allocator();
    loop_hash_table = std.AutoArrayHashMap(usize, *Loop).init(allocator);
    statement_hash_table = std.AutoArrayHashMap(*Graph.Vertex, *Statement).init(allocator);
    subprograms = std.ArrayList(*Program).init(allocator);
}

pub fn fromGraph() !*Program {
    // TODO: A graph can have multiple groups and each group needs to correspond to a separate program
    // A global arraylist will need to be maintained to add new programs to
    // Similarly codegen will need to provide a top level function for generating code for multiple programs
    // where each program corresponds to a function or a kernel
    var p = try new();
    const last_loop = try p.loops(Graph.entrypoint.?);
    try p.body.contents.append(allocator, .{ .Loop = last_loop });
    p.group_id = last_loop.node.group_id;
    return p;
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
    if (loop_hash_table.contains(v.tensorId())) {
        return loop_hash_table.get(v.tensorId()).?;
    }

    return build_loop: {
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

        // Reorder loops such that accumulation happens innermost but with same dimensional order
        const outer_loop: *Loop = reg_loops.items[0];
        curr_loop = outer_loop;
        for (reg_loops.items[1..]) |reg_loop| {
            try curr_loop.body.contents.append(allocator, .{ .Loop = reg_loop });
            curr_loop = reg_loop;
        }
        for (acc_loops.items) |acc_loop| {
            try curr_loop.body.contents.append(allocator, .{ .Loop = acc_loop });
            curr_loop = acc_loop;
        }
        switch (v.edge) {
            .InitOp => {},
            .ZipOp => |edge| {
                var a_loop = program.loops(edge.a) catch unreachable;
                const b_loop = program.loops(edge.b) catch unreachable;
                if (edge.a.group_id == edge.b.group_id and edge.b.group_id == v.group_id and !edge.fused_a and !edge.fused_b) {
                    a_loop.prev = b_loop;
                    outer_loop.prev = a_loop;
                    if (!loop_hash_table.contains(a_loop.node.tensorId())) {
                        try program.body.contents.append(allocator, .{ .Loop = a_loop });
                    }
                    if (!loop_hash_table.contains(b_loop.node.tensorId())) {
                        try program.body.contents.append(allocator, .{ .Loop = b_loop });
                    }
                } else {
                    if (edge.a.group_id == v.group_id and !edge.fused_a) {
                        outer_loop.prev = a_loop;
                        if (!loop_hash_table.contains(a_loop.node.tensorId())) {
                            try program.body.contents.append(allocator, .{ .Loop = a_loop });
                        }
                    } else if (edge.b.group_id == v.group_id and !edge.fused_b) {
                        outer_loop.prev = b_loop;
                        if (!loop_hash_table.contains(b_loop.node.tensorId())) {
                            try program.body.contents.append(allocator, .{ .Loop = b_loop });
                        }
                    }
                }
                try loop_hash_table.put(a_loop.node.tensorId(), a_loop);
                try loop_hash_table.put(b_loop.node.tensorId(), b_loop);
            },
            .TypeOp => |edge| {
                const x_loop = program.loops(edge.x) catch unreachable;
                switch (edge.op) {
                    .AsType => {
                        if (edge.x.group_id == v.group_id and !edge.fused_x) {
                            outer_loop.prev = x_loop;
                            if (!loop_hash_table.contains(x_loop.node.tensorId())) {
                                try program.body.contents.append(allocator, .{ .Loop = x_loop });
                            }
                        }
                    },
                    else => {
                        if (!loop_hash_table.contains(x_loop.node.tensorId())) {
                            try program.body.contents.append(allocator, .{ .Loop = x_loop });
                        }
                    },
                }
                try loop_hash_table.put(x_loop.node.tensorId(), x_loop);
            },
            inline else => |edge| {
                const x_loop = program.loops(edge.x) catch unreachable;
                if (edge.x.group_id == v.group_id and !edge.fused_x) {
                    outer_loop.prev = x_loop;
                    if (!loop_hash_table.contains(x_loop.node.tensorId())) {
                        try program.body.contents.append(allocator, .{ .Loop = x_loop });
                    }
                }
                try loop_hash_table.put(x_loop.node.tensorId(), x_loop);
            },
        }

        const statement: *Statement = try allocator.create(Statement);
        statement.* = switch (v.edge) {
            .InitOp => |edge| .{ .InitOp = .{ .op = edge.op, .out = v, .init = edge.value } },
            .ZipOp => |edge| .{ .ZipOp = .{
                .op = edge.op,
                .a = edge.a,
                .a_statement = if (edge.fused_a) statement_hash_table.get(edge.a) else null,
                .b = edge.b,
                .b_statement = if (edge.fused_b) statement_hash_table.get(edge.b) else null,
                .out = v,
            } },
            .MapOp => |edge| .{ .MapOp = .{
                .op = edge.op,
                .x = edge.x,
                .x_statement = if (edge.fused_x) statement_hash_table.get(edge.x) else null,
                .out = v,
            } },
            .ReduceOp => |edge| .{ .ReduceOp = .{
                .op = edge.op,
                .x = edge.x,
                .x_statement = if (edge.fused_x) statement_hash_table.get(edge.x) else null,
                .out = v,
            } },
            .TypeOp => |edge| .{
                .TypeOp = .{
                    .op = edge.op,
                    .x = edge.x,
                    .x_statement = if (edge.op == .AsType) statement_hash_table.get(edge.x) else {
                        // Escape hatch, only AsType corresponds to a statement inside a loop
                        outer_loop.* = loop_hash_table.get(edge.x.tensorId()).?.*;
                        return outer_loop;
                    },
                    .out = v,
                },
            },
        };

        try statement_hash_table.put(v, statement);
        try curr_loop.body.contents.append(allocator, .{ .Statement = statement });
        break :build_loop outer_loop;
    };
}

/// Statement of the form y = f(x)
/// y can either be a value in an array or a variable
/// f(x) is an arithmetic operation on a value in an array or a variable
pub const Statement = union(ops.GraphOps) {
    MapOp: struct {
        op: ops.MapOp,
        x: *const Graph.Vertex,
        x_statement: ?*const Statement,
        out: *const Graph.Vertex,
    },
    ZipOp: struct {
        op: ops.ZipOp,
        a: *const Graph.Vertex,
        a_statement: ?*const Statement,
        b: *const Graph.Vertex,
        b_statement: ?*const Statement,
        out: *const Graph.Vertex,
    },
    ReduceOp: struct {
        op: ops.ReduceOp,
        x: *const Graph.Vertex,
        x_statement: ?*const Statement,
        out: *Graph.Vertex,
    },
    TypeOp: struct {
        op: ops.TypeOp,
        x: *const Graph.Vertex,
        x_statement: ?*const Statement,
        out: *const Graph.Vertex,
    },
    InitOp: struct {
        op: ops.InitOp,
        out: *const Graph.Vertex,
        init: ops.InitValue,
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
        Statement: *Statement,
    };
    contents: std.MultiArrayList(Content),
};

pub fn code(program: *const Program, comptime Generator: type, writer: anytype) !void {
    Generator.init(allocator);
    try Generator.code(program, writer);
}

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
    Graph.trace(&out);
    // Graph.applyGreedyFusion();
    Program.init(std.testing.allocator);
    defer Program.deinit();
    const program = try Program.fromGraph();
    Zig.init(std.testing.allocator);
    defer Zig.deinit();

    const Logger = struct {
        pub fn print(comptime fmt: anytype, args: anytype) void {
            std.log.debug(fmt, args);
        }
    };
    try Zig.code(program, Logger);
}
