const ops = @import("ops.zig");
const std = @import("std");
const codegen = @import("codegen.zig");
const Graph = @import("Graph.zig");
const Program = @This();

var arena: std.heap.ArenaAllocator = undefined;
var allocator: std.mem.Allocator = undefined;
var loop_hash_table: std.AutoArrayHashMap(usize, *Loop) = undefined;
var statement_hash_table: std.AutoArrayHashMap(*Graph.TensorNode, *Statement) = undefined;
var subprograms: std.ArrayList(*Program) = undefined;

// Null group id corresponds to global scope
group: ?usize = null,
body: Body,

pub fn init(backing_allocator: std.mem.Allocator) void {
    arena = std.heap.ArenaAllocator.init(backing_allocator);
    allocator = arena.allocator();
    loop_hash_table = std.AutoArrayHashMap(usize, *Loop).init(allocator);
    statement_hash_table = std.AutoArrayHashMap(*Graph.TensorNode, *Statement).init(allocator);
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
    const last_loop = try program.loops(.{ .node = Graph.entry() });
    // Last loop will not get added to the program as it is the caller's responsibility to do so
    // This is because only the caller knows if the Graph node corresponding to the loop is fused
    // with the node corresponding to the caller
    try program.body.contents.append(allocator, .{ .Loop = last_loop });
    try loop_hash_table.put(last_loop.tensor_node.tensor.id(), last_loop);
    // Keep track of the group (kernel/function) that the program corresponds to
    program.group = last_loop.tensor_node.group;
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
pub fn loops(program: *Program, input: Graph.OpNode.Input) !*Loop {
    if (loop_hash_table.contains(input.node.tensor.id())) {
        return loop_hash_table.get(input.node.tensor.id()).?;
    }

    var reg_loops = std.ArrayList(*Loop).init(allocator);
    var acc_loops = std.ArrayList(*Loop).init(allocator);
    var curr_loop = try allocator.create(Loop);
    for (0..input.node.tensor.ndims) |d| {
        curr_loop.* = .{
            .upper_bound = switch (input.node.opNode().*) {
                .ReduceOp => |op_node| op_node.x.node.tensor.shape[d],
                else => input.node.tensor.shape[d],
            },
            .tensor_node = input.node,
            .dim = d,
            .acc = switch (input.node.opNode().*) {
                // If the op is a reduce, check if current dim is being reduced
                .ReduceOp => |op_node| op_node.dims[d],
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
        if (d != input.node.tensor.ndims - 1) {
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
    switch (input.node.opNode().*) {
        .InitOp => {},
        .ZipOp => |op_node| {
            const a_loop = try program.loops(op_node.a);
            const b_loop = try program.loops(op_node.b);
            // Only add the preceding loop to the program if we know its statements are not fused
            // and the node is associated with the same kernel group
            if (op_node.a.node.group == input.node.group and !op_node.a.fused) {
                if (!loop_hash_table.contains(a_loop.tensor_node.tensor.id())) {
                    try program.body.contents.append(allocator, .{ .Loop = a_loop });
                }
            } else if (op_node.b.node.group == input.node.group and !op_node.b.fused) {
                if (!loop_hash_table.contains(b_loop.tensor_node.tensor.id())) {
                    try program.body.contents.append(allocator, .{ .Loop = b_loop });
                }
            }
            try loop_hash_table.put(a_loop.tensor_node.tensor.id(), a_loop);
            try loop_hash_table.put(b_loop.tensor_node.tensor.id(), b_loop);
        },
        .TypeOp => |op_node| {
            const x_loop = try program.loops(op_node.x);
            switch (op_node.op) {
                // Only the AsType type op corresponds to a loop action
                // All other type ops are symbolic / logic happens in the statement itself
                .AsType => {
                    if (op_node.x.node.group == input.node.group) {
                        if (!loop_hash_table.contains(x_loop.tensor_node.tensor.id())) {
                            try program.body.contents.append(allocator, .{ .Loop = x_loop });
                        }
                    }
                },
                else => {
                    // Don't add the loop to the program because the next loop will automatically
                    // pass through the statement from this loop.
                },
            }
            try loop_hash_table.put(x_loop.tensor_node.tensor.id(), x_loop);
        },
        // Applies to map and reduce ops
        inline else => |op_node| {
            const x_loop = try program.loops(op_node.x);
            if (op_node.x.node.group == input.node.group and !op_node.x.fused) {
                if (!loop_hash_table.contains(x_loop.tensor_node.tensor.id())) {
                    try program.body.contents.append(allocator, .{ .Loop = x_loop });
                }
            }
            try loop_hash_table.put(x_loop.tensor_node.tensor.id(), x_loop);
        },
    }

    // Create the statement that goes in the innermost loop based on the op
    // When fusion is happening, statements are nested inside each other
    // TODO: Create a "Read statement" for reading a tensor rather than using null to indicate end of nesting
    const statement: *Statement = try allocator.create(Statement);
    statement.* = switch (input.node.opNode().*) {
        .InitOp => |op_node| .{ .InitOp = .{ .op = op_node.op, .out = input.node, .init = op_node.value } },
        .ZipOp => |op_node| .{ .ZipOp = .{
            .op = op_node.op,
            .a = op_node.a.node,
            .a_statement = if (op_node.a.fused) statement_hash_table.get(op_node.a.node) else null,
            .b = op_node.b.node,
            .b_statement = if (op_node.b.fused) statement_hash_table.get(op_node.b.node) else null,
            .out = input.node,
        } },
        .MapOp => |op_node| .{ .MapOp = .{
            .op = op_node.op,
            .x = op_node.x.node,
            .x_statement = if (op_node.x.fused) statement_hash_table.get(op_node.x.node) else null,
            .out = input.node,
        } },
        .ReduceOp => |op_node| .{ .ReduceOp = .{
            .op = op_node.op,
            .x = op_node.x.node,
            .x_statement = if (op_node.x.fused) statement_hash_table.get(op_node.x.node) else null,
            .out = input.node,
        } },
        .TypeOp => |op_node| .{
            .TypeOp = .{
                .op = op_node.op,
                .x = op_node.x.node,
                .x_statement = statement_hash_table.get(op_node.x.node),
                .out = input.node,
            },
        },
    };
    try statement_hash_table.putNoClobber(input.node, statement);
    try curr_loop.body.contents.append(allocator, .{ .Statement = statement });
    return switch (input.node.opNode().*) {
        .TypeOp => |op_node| loop_hash_table.get(op_node.x.node.tensor.id()).?,
        inline else => outer_loop,
    };
    // TODO: Global statements need to be pushed to the global program
    // This includes initializations that happen outside the loop
    // TODO: Array declarations need to be pushed into the program body
}

/// Statement of the form y = f(x)
/// y can either be a value in an array or a variable
/// f(x) is an arithmetic operation on a value in an array or a variable
pub const Statement = union(ops.OpTypes) {
    MapOp: struct {
        op: ops.MapOp,
        x: *const Graph.TensorNode,
        x_statement: ?*const Statement,
        out: *const Graph.TensorNode,
    },
    ZipOp: struct {
        op: ops.ZipOp,
        a: *const Graph.TensorNode,
        a_statement: ?*const Statement,
        b: *const Graph.TensorNode,
        b_statement: ?*const Statement,
        out: *const Graph.TensorNode,
    },
    ReduceOp: struct {
        op: ops.ReduceOp,
        x: *const Graph.TensorNode,
        x_statement: ?*const Statement,
        out: *Graph.TensorNode,
    },
    TypeOp: struct {
        op: ops.TypeOp,
        x: *const Graph.TensorNode,
        x_statement: ?*const Statement,
        out: *const Graph.TensorNode,
    },
    InitOp: struct {
        op: ops.InitOp,
        out: *const Graph.TensorNode,
        init: ops.InitValue,
    },
};

/// Abstractions for lowering Graph.Node into a loop which can be codegened
/// loop structs will be stored in a list (program) where order is exact order of code
/// loops are defined as a grammar, every loop has a header and a body
pub const Loop = struct {
    upper_bound: usize,
    tensor_node: *Graph.TensorNode,
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
