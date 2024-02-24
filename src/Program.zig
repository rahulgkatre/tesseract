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
    _ = try p.loops(Graph.entrypoint.?);
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
    if (loop_hash_table.contains(v.id)) {
        return loop_hash_table.get(v.id).?;
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
                if (edge.a.group_id == edge.b.group_id and edge.b.group_id == v.group_id) { // and !edge.fused_a and !edge.fused_b) {
                    var a_loop = program.loops(edge.a) catch unreachable;
                    a_loop.prev = program.loops(edge.b) catch unreachable;
                    outer_loop.prev = a_loop;
                } else {
                    if (edge.a.group_id == v.group_id) { // and !edge.fused_a) {
                        outer_loop.prev = program.loops(edge.a) catch unreachable;
                    } else if (edge.b.group_id == v.group_id) { // and !edge.fused_b) {
                        outer_loop.prev = program.loops(edge.b) catch unreachable;
                    }
                }
            },
            .TypeOp => |edge| {
                switch (edge.op) {
                    .AsType => {
                        outer_loop.prev = program.loops(edge.x) catch unreachable;
                    },
                    else => {
                        outer_loop.* = (program.loops(edge.x) catch unreachable).*;
                    },
                }
            },
            inline else => |edge| {
                if (edge.x.group_id == v.group_id) { // and !edge.fused_x) {
                    outer_loop.prev = program.loops(edge.x) catch unreachable;
                }
            },
        }
        const statement: *Statement = try allocator.create(Statement);
        statement.* = switch (v.edge) {
            .InitOp => |edge| .{ .InitOp = .{ .id = 0, .op = edge.op, .out = v, .init = edge.value } },
            .ZipOp => |edge| .{ .ZipOp = .{
                .id = 0,
                .op = edge.op,
                .a = edge.a,
                .a_statement = if (edge.fused_a) statement_hash_table.get(edge.a) else null,
                .b = edge.b,
                .b_statement = if (edge.fused_b) statement_hash_table.get(edge.b) else null,
                .out = v,
            } },
            .MapOp => |edge| .{ .MapOp = .{
                .id = 0,
                .op = edge.op,
                .x = edge.x,
                .x_statement = if (edge.fused_x) statement_hash_table.get(edge.x) else null,
                .out = v,
            } },
            .ReduceOp => |edge| .{ .ReduceOp = .{
                .id = 0,
                .op = edge.op,
                .x = edge.x,
                .x_statement = if (edge.fused_x) statement_hash_table.get(edge.x) else null,
                .out = v,
            } },
            .TypeOp => |edge| .{
                .TypeOp = .{
                    .id = 0,
                    .op = edge.op,
                    .x = edge.x,
                    .x_statement = if (edge.op == .AsType) statement_hash_table.get(edge.x) else {
                        // Escape hatch, only AsType corresponds to a statement inside a loop
                        outer_loop.* = loop_hash_table.get(edge.x.id).?.*;
                        return outer_loop;
                    },
                    .out = v,
                },
            },
        };
        try statement_hash_table.put(v, statement);
        try curr_loop.body.contents.append(allocator, .{ .Statement = statement });
        if (!loop_hash_table.contains(outer_loop.node.tensor.id)) {
            try program.body.contents.append(allocator, .{ .Loop = outer_loop });
        }
        try loop_hash_table.put(outer_loop.node.tensor.id, outer_loop);
        break :build_loop outer_loop;
    };
}

/// Statement of the form y = f(x)
/// y can either be a value in an array or a variable
/// f(x) is an arithmetic operation on a value in an array or a variable
pub const Statement = union(ops.GraphOps) {
    MapOp: struct {
        id: usize,
        op: ops.MapOp,
        x: *const Graph.Vertex,
        x_statement: ?*const Statement,
        out: *const Graph.Vertex,
    },
    ZipOp: struct {
        id: usize,
        op: ops.ZipOp,
        a: *const Graph.Vertex,
        a_statement: ?*const Statement,
        b: *const Graph.Vertex,
        b_statement: ?*const Statement,
        out: *const Graph.Vertex,
    },
    ReduceOp: struct {
        id: usize,
        op: ops.ReduceOp,
        x: *const Graph.Vertex,
        x_statement: ?*const Statement,
        out: *Graph.Vertex,
    },
    TypeOp: struct {
        id: usize,
        op: ops.TypeOp,
        x: *const Graph.Vertex,
        x_statement: ?*const Statement,
        out: *const Graph.Vertex,
    },
    InitOp: struct {
        id: usize,
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
    Graph.trace(out);
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
