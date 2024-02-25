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

// Null group id corresponds to global scope
group_id: ?usize = null,
body: Body,

pub fn init(backing_allocator: std.mem.Allocator) void {
    arena = std.heap.ArenaAllocator.init(backing_allocator);
    allocator = arena.allocator();
    loop_hash_table = std.AutoArrayHashMap(usize, *Loop).init(allocator);
    statement_hash_table = std.AutoArrayHashMap(*Graph.Vertex, *Statement).init(allocator);
    subprograms = std.ArrayList(*Program).init(allocator);
}

fn new() !*Program {
    const program = try allocator.create(Program);
    program.body = .{
        .contents = std.MultiArrayList(Body.Content){},
    };
    return program;
}

pub fn code(comptime Generator: type, writer: anytype) !void {
    var program = try new();
    const last_loop = try program.loops(Graph.entry());
    // Last loop will not get added to the program as it is the caller's responsibility to do so
    // This is because only the caller knows if the Graph node corresponding to the loop is fused
    // with the node corresponding to the caller
    try program.body.contents.append(allocator, .{ .Loop = last_loop });
    try loop_hash_table.put(last_loop.node.tensorId(), last_loop);
    // Keep track of the group (kernel/function) that the program corresponds to
    program.group_id = last_loop.node.group_id;
    // Call the code generator
    // TODO: Support for a list of programs to codegen together
    Generator.init(allocator);
    try Generator.code(program, writer);
}

pub fn deinit() void {
    loop_hash_table.deinit();
    loop_hash_table = undefined;
    allocator = undefined;
    arena.deinit();
    arena = undefined;
}

/// Lower the node to a loop nest representation
pub fn loops(program: *Program, node: *Graph.Vertex) !*Loop {
    if (loop_hash_table.contains(node.tensorId())) {
        return loop_hash_table.get(node.tensorId()).?;
    }

    var reg_loops = std.ArrayList(*Loop).init(allocator);
    var acc_loops = std.ArrayList(*Loop).init(allocator);
    var curr_loop = try allocator.create(Loop);
    for (0..node.tensor.ndims) |d| {
        curr_loop.* = .{
            .upper_bound = switch (node.edge) {
                .ReduceOp => |edge| edge.x.tensor.shape[d],
                else => node.tensor.shape[d],
            },
            .node = node,
            .dim = d,
            .acc = switch (node.edge) {
                // If the op is a reduce, check if current dim is being reduced
                .ReduceOp => |edge| edge.dims[d],
                else => false,
            },
            .body = .{
                // MultiArrayList list for body of Statements and Loops
                .contents = std.MultiArrayList(Body.Content){},
            },
        };
        // Add the loop to the correct list based on whether it is accumulating or not
        if (curr_loop.acc) {
            try acc_loops.append(curr_loop);
        } else {
            try reg_loops.append(curr_loop);
        }
        // Move 1 more loop into the nest
        if (d != node.tensor.ndims - 1) {
            const inner_loop: *Loop = try allocator.create(Loop);
            curr_loop = inner_loop;
        }
    }
    // Reorder loops such that accumulation happens innermost while preserving same dimensional order
    const outer_loop: *Loop = if (reg_loops.items.len > 0) reg_loops.items[0] else acc_loops.items[0];
    curr_loop = outer_loop;
    if (reg_loops.items.len > 1) {
        for (reg_loops.items[1..]) |reg_loop| {
            try curr_loop.body.contents.append(allocator, .{ .Loop = reg_loop });
            curr_loop = reg_loop;
        }
    }
    for (acc_loops.items) |acc_loop| {
        try curr_loop.body.contents.append(allocator, .{ .Loop = acc_loop });
        curr_loop = acc_loop;
    }
    // Recursive calls to lower the preceding loop
    switch (node.edge) {
        .InitOp => {},
        .ZipOp => |edge| {
            const a_loop = try program.loops(edge.a);
            const b_loop = try program.loops(edge.b);
            // Only add the preceding loop to the program if we know its statements are not fused
            // and the node is associated with the same kernel group
            if (edge.a.group_id == node.group_id and !edge.fused_a) {
                if (!loop_hash_table.contains(a_loop.node.tensorId())) {
                    try program.body.contents.append(allocator, .{ .Loop = a_loop });
                }
            } else if (edge.b.group_id == node.group_id and !edge.fused_b) {
                if (!loop_hash_table.contains(b_loop.node.tensorId())) {
                    try program.body.contents.append(allocator, .{ .Loop = b_loop });
                }
            }
            try loop_hash_table.put(a_loop.node.tensorId(), a_loop);
            try loop_hash_table.put(b_loop.node.tensorId(), b_loop);
        },
        .TypeOp => |edge| {
            const x_loop = try program.loops(edge.x);
            switch (edge.op) {
                // Only the AsType type op corresponds to a loop action
                // All other type ops are symbolic / logic happens in the statement itself
                .AsType => {
                    if (edge.x.group_id == node.group_id) {
                        if (!loop_hash_table.contains(x_loop.node.tensorId())) {
                            try program.body.contents.append(allocator, .{ .Loop = x_loop });
                        }
                    }
                },
                else => {
                    // Don't add the loop to the program because the next loop will automatically
                    // pass through the statement from this loop.
                },
            }
            try loop_hash_table.put(x_loop.node.tensorId(), x_loop);
        },
        // Applies to map and reduce ops
        inline else => |edge| {
            const x_loop = try program.loops(edge.x);
            if (edge.x.group_id == node.group_id and !edge.fused_x) {
                if (!loop_hash_table.contains(x_loop.node.tensorId())) {
                    try program.body.contents.append(allocator, .{ .Loop = x_loop });
                }
            }
            try loop_hash_table.put(x_loop.node.tensorId(), x_loop);
        },
    }

    // Create the statement that goes in the innermost loop based on the op
    // When fusion is happening, statements are nested inside each other
    // TODO: Create a "Read statement" for reading a tensor rather than using null to indicate end of nesting
    const statement: *Statement = try allocator.create(Statement);
    statement.* = switch (node.edge) {
        .InitOp => |edge| .{ .InitOp = .{ .op = edge.op, .out = node, .init = edge.value } },
        .ZipOp => |edge| .{ .ZipOp = .{
            .op = edge.op,
            .a = edge.a,
            .a_statement = if (edge.fused_a) statement_hash_table.get(edge.a) else null,
            .b = edge.b,
            .b_statement = if (edge.fused_b) statement_hash_table.get(edge.b) else null,
            .out = node,
        } },
        .MapOp => |edge| .{ .MapOp = .{
            .op = edge.op,
            .x = edge.x,
            .x_statement = if (edge.fused_x) statement_hash_table.get(edge.x) else null,
            .out = node,
        } },
        .ReduceOp => |edge| .{ .ReduceOp = .{
            .op = edge.op,
            .x = edge.x,
            .x_statement = if (edge.fused_x) statement_hash_table.get(edge.x) else null,
            .out = node,
        } },
        .TypeOp => |edge| .{
            .TypeOp = .{
                .op = edge.op,
                .x = edge.x,
                .x_statement = statement_hash_table.get(edge.x),
                .out = node,
            },
        },
    };
    try statement_hash_table.putNoClobber(node, statement);
    try curr_loop.body.contents.append(allocator, .{ .Statement = statement });
    return switch (node.edge) {
        .TypeOp => |edge| loop_hash_table.get(edge.x.tensorId()).?,
        inline else => outer_loop,
    };
    // TODO: Global statements need to be pushed to the global program
    // This includes initializations that happen outside the loop
    // TODO: Array declarations need to be pushed into the program body
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
};

pub const Body = struct {
    pub const Content = union(enum) {
        Loop: *Loop,
        Statement: *Statement,
    };
    contents: std.MultiArrayList(Content),
};
