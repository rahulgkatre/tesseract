const std = @import("std");
const ops = @import("ops.zig");
const Graph = @import("Graph.zig");
const dtypes = @import("dtypes.zig");

var gpa: std.heap.GeneralPurposeAllocator(.{}) = undefined;

var arena: std.heap.ArenaAllocator = undefined;
var loops: std.AutoHashMap(usize, *Loop) = undefined;
var statements: std.AutoHashMap(usize, *Statement) = undefined;
var operands: std.AutoHashMap(usize, Statement.Expression.Operand) = undefined;

var global_body: std.MultiArrayList(ScheduleItem) = undefined;

pub fn init() void {
    gpa = .{};
    arena = std.heap.ArenaAllocator.init(gpa.allocator());
    loops = std.AutoHashMap(usize, *Loop).init(arena.allocator());
    statements = std.AutoHashMap(usize, *Statement).init(arena.allocator());
    operands = std.AutoHashMap(usize, Statement.Expression.Operand).init(arena.allocator());

    global_body = .{};
}

pub fn create() !void {
    init();
    defer deinit();
    for (Graph.dagSinks()) |entry| {
        Loop.create(.{ .tensor = entry }) catch unreachable;
    }

    const slice = try Json.toJsonCompatibleSlice(global_body);
    defer Json.deinitJsonCompatibleSlice(slice);
    const str = try std.json.stringifyAlloc(gpa.allocator(), slice, .{
        .whitespace = .indent_2,
    });
    defer gpa.allocator().free(str);
    std.debug.print("\n{s}\n", .{str});
}

pub fn deinit() void {
    arena.deinit();
    _ = gpa.deinit();
}

const ScheduleItemEnum = enum {
    loop,
    statement,
};
const ScheduleItem = union(ScheduleItemEnum) {
    loop: *Loop,
    statement: *Statement,
};

// TODO: When parsing a JSON, either recreate the nodes too
const Json = struct {
    /// MultiArrayList does not automatically stringify so the default jsonStringify function
    /// does not work. To make loops JSON compatible, the body is replaced with a slice of unions.
    /// Deinit functions are provided as JsonCompatibleLoops are created with a gpa.allocator().
    const JsonCompatibleItem = union(ScheduleItemEnum) {
        loop: *JsonCompatibleLoop,
        statement: *Statement,

        fn deinit(self: *JsonCompatibleItem) void {
            switch (self.*) {
                .loop => |loop| loop.deinit(),
                else => {},
            }
        }
    };
    const JsonCompatibleLoop = struct {
        dim: u8,
        group: u64,
        bound: u64,
        reduce: bool,
        body: []JsonCompatibleItem,

        fn deinit(self: *JsonCompatibleLoop) void {
            deinitJsonCompatibleSlice(self.body);
            gpa.allocator().destroy(self);
        }
    };

    fn toJsonCompatibleSlice(body: std.MultiArrayList(ScheduleItem)) std.mem.Allocator.Error![]JsonCompatibleItem {
        const slice = try gpa.allocator().alloc(JsonCompatibleItem, body.len);
        for (body.items(.tags), body.items(.data), 0..) |tag, data, i| {
            switch (tag) {
                .loop => slice[i] = @unionInit(JsonCompatibleItem, @tagName(ScheduleItemEnum.loop), try toJsonCompatibleLoop(data.loop)),
                inline else => |active| slice[i] = @unionInit(
                    JsonCompatibleItem,
                    @tagName(active),
                    @field(data, @tagName(active)),
                ),
            }
        }
        return slice;
    }

    fn deinitJsonCompatibleSlice(slice: []JsonCompatibleItem) void {
        for (slice) |*item| {
            item.deinit();
        }
        gpa.allocator().free(slice);
    }

    fn toJsonCompatibleLoop(loop: *const Loop) !*JsonCompatibleLoop {
        const compatible_loop = try gpa.allocator().create(JsonCompatibleLoop);
        compatible_loop.* = .{
            .dim = loop.dim,
            .group = loop.group,
            .bound = loop.bound,
            .reduce = loop.reduce,
            .body = try toJsonCompatibleSlice(loop.body),
        };
        return compatible_loop;
    }

    fn fromJsonCompatibleLoop(parsed_loop: *const JsonCompatibleLoop) !*Loop {
        // Use the arena allocator because the parsed loop will join the other loops of the schedule
        var loop: *Loop = try arena.allocator().create(Loop);
        loop.dim = parsed_loop.dim;
        loop.group = parsed_loop.group;
        loop.reduce = parsed_loop.reduce;
        loop.bound = parsed_loop.bound;
        loop.body = .{};
        try loop.body.ensureTotalCapacity(arena.allocator(), parsed_loop.body.len);
        for (parsed_loop.body) |item| {
            const converted_item = switch (item) {
                .loop => |inner_loop| @unionInit(ScheduleItem, @tagName(ScheduleItemEnum.loop), try fromJsonCompatibleLoop(inner_loop)),
                .statement => |inner_stmt| @unionInit(ScheduleItem, @tagName(ScheduleItemEnum.statement), inner_stmt),
            };
            loop.body.appendAssumeCapacity(converted_item);
        }
        return loop;
    }
};

pub const Loop = struct {
    dim: u8,
    group: u64,
    reduce: bool,
    bound: u64,
    tensor: *Graph.TensorNode,
    body: std.MultiArrayList(ScheduleItem),

    /// Calculate an ordinal value used for sorting a loop nest
    /// Reduce loops are given a larger number so that they are the innermost loops
    /// Original nesting order is preserved as loops are ordered by tensor dimension
    fn ordinal(self: *const Loop) u16 {
        const ndims = self.tensor.ndims;
        if (self.reduce) {
            return ndims + self.dim;
        } else {
            return self.dim;
        }
    }

    /// Compare 2 loops, used for loop nest sorting
    fn loopCompare(_: void, lhs_loop: *const Loop, rhs_loop: *const Loop) bool {
        return lhs_loop.ordinal() < rhs_loop.ordinal();
    }

    fn create(target: Graph.OpNode.Input) !void {
        if (loops.contains(target.tensor.uid)) {
            return;
        }
        switch (target.tensor.op_node) {
            .InitOp => if (target.tensor.op_node.InitOp.op == .Input) return,
            .BinaryOp => |op_node| {
                // Generate the loop that is temporally closer first
                if (op_node.a.tensor.uid < op_node.b.tensor.uid) {
                    try Loop.create(op_node.a);
                    try Loop.create(op_node.b);
                } else {
                    try Loop.create(op_node.b);
                    try Loop.create(op_node.a);
                }
                if (target.fused) {
                    return;
                }
            },
            .ReduceOp => |op_node| {
                // Reductions happen in a separate loop that may be fused in a later step
                // Keep making the current loop even if the op is fused
                try Loop.create(op_node.x);
            },
            inline else => |op_node| {
                try Loop.create(op_node.x);
                if (target.fused) {
                    return;
                }
            },
        }
        const ndims = target.tensor.ndims;
        var loop_nest: []*Loop = try gpa.allocator().alloc(*Loop, ndims);
        defer gpa.allocator().free(loop_nest);
        for (loop_nest, 0..) |*loop, dim| {
            loop.* = try arena.allocator().create(Loop);
            loop.*.* = .{
                .dim = @intCast(dim),
                .group = target.tensor.group.?,
                .bound = target.tensor.shape[dim],
                .reduce = switch (target.tensor.op_node) {
                    .ReduceOp => |op_node| op_node.dims[dim],
                    else => false,
                },
                .body = .{},
                .tensor = target.tensor,
            };
        }
        // Sort the loops such that all reduced dims correspond to innermost loops
        std.sort.block(*const Loop, loop_nest, {}, loopCompare);
        // Nest the loops
        const outermost_loop = loop_nest[0];
        for (loop_nest[0 .. ndims - 1], loop_nest[1..]) |outer, inner| {
            try outer.body.append(arena.allocator(), .{ .loop = inner });
        }

        try loop_nest[ndims - 1].body.append(arena.allocator(), .{ .statement = try Statement.getOrInit(target) });
        try global_body.append(arena.allocator(), .{ .loop = outermost_loop });
        // TODO: For a reduce op, fuse this loop with the preceding loop
        try loops.putNoClobber(target.tensor.uid, outermost_loop);
    }

    /// Custom jsonStringify implementation that internally converts a Loop to a JsonCompatibleLoop and stringifies that
    /// and generates the Json string from the new loop
    pub fn jsonStringify(self: *const Loop, write_stream: anytype) !void {
        const compatible_loop = try Json.toJsonCompatibleLoop(self);
        defer compatible_loop.deinit();
        try write_stream.write(compatible_loop);
    }

    /// Custom jsonStringify implementation that internally converts a parsed JsonCompatibleLoop into a regular loop
    pub fn jsonParse(allocator: std.mem.Allocator, source: *std.json.Scanner, options: std.json.ParseOptions) !Loop {
        const parsed = try std.json.parseFromTokenSource(*const Json.JsonCompatibleLoop, allocator, source, options);
        defer parsed.deinit();
        const loop = try Json.fromJsonCompatibleLoop(parsed.value);
        return loop.*;
    }
};

pub const Statement = struct {
    pub const Expression = union(ops.OpTypes) {
        const Operand = union(enum) {
            global: *const Graph.TensorNode,
            expression: *const Expression,
            // TODO: Local will need to be a separate from the tensor node it is caching
            // as it will have a different size and may not be a scalar
            // Its size will be determine by depth it is used in a loop nest
            local: *const Graph.TensorNode,

            fn init(target: Graph.OpNode.Input, consumer_group: ?u64) std.mem.Allocator.Error!Operand {
                if (target.fused and !target.tensor.isCached()) {
                    switch (target.tensor.op_node) {
                        .InitOp => |op_node| {
                            if (op_node.op == .Input) {
                                // Input is always global
                                return .{ .global = target.tensor };
                            } else {
                                return .{ .expression = &(try Statement.getOrInit(target)).expression };
                            }
                        },
                        .ReduceOp => {
                            return .{ .local = target.tensor };
                        },
                        .DataOp => |op_node| {
                            if (op_node.op == .AsType) {
                                return .{ .expression = &(try Statement.getOrInit(target)).expression };
                            } else {
                                switch (op_node.x.tensor.op_node) {
                                    .InitOp => |x_op_node| if (x_op_node.op == .Input) {
                                        // Input is always global
                                        return .{ .global = x_op_node.out.tensor };
                                    },
                                    else => {},
                                }
                                return .{ .expression = &(try Statement.getOrInit(op_node.x)).expression };
                            }
                        },
                        else => return .{ .expression = &(try Statement.getOrInit(target)).expression },
                    }
                } else {
                    if (consumer_group) |group| {
                        if (target.tensor.isCached() and target.tensor.group == group) {
                            return .{ .local = target.tensor };
                        }
                    }
                    // Otherwise the operand tensor is a global access
                    return .{ .global = target.tensor };
                }
            }
        };
        UnaryOp: struct {
            op: ops.UnaryOp,
            x: Operand,
        },
        BinaryOp: struct {
            op: ops.BinaryOp,
            a: Operand,
            b: Operand,
        },
        ReduceOp: struct {
            op: ops.ReduceOp,
            x: Operand,
        },
        DataOp: struct {
            op: ops.DataOp,
            x: Operand,
        },
        InitOp: struct {
            op: ops.InitOp,
            args: ops.InitOp.Args,
        },
    };
    const Output = union(enum) {
        global: *Graph.TensorNode,
        accumulator: *Graph.TensorNode,
        local: *Graph.TensorNode,
    };
    group: ?usize,
    expression: Expression,
    out: Output,

    fn getOrInit(target: Graph.OpNode.Input) std.mem.Allocator.Error!*Statement {
        if (statements.get(target.tensor.uid)) |stmt| {
            return stmt;
        } else {
            const tmp_statement: Statement = .{
                .group = target.tensor.group,
                .expression = switch (target.tensor.op_node) {
                    // Workaround: since we don't know what the statement is being used for
                    // just put an identity mapping to the input
                    // TODO: Fix this so that it can be replaced with unreachable
                    .InitOp => |op_node| if (op_node.op != .Input) .{ .InitOp = .{
                        .op = op_node.op,
                        .args = op_node.args,
                    } } else .{ .UnaryOp = .{
                        .op = .Copy,
                        .x = .{ .global = op_node.out.tensor },
                    } },
                    .BinaryOp => |op_node| .{ .BinaryOp = .{
                        .op = op_node.op,
                        .a = try Expression.Operand.init(op_node.a, target.tensor.group),
                        .b = try Expression.Operand.init(op_node.b, target.tensor.group),
                    } },
                    .UnaryOp => |op_node| .{ .UnaryOp = .{
                        .op = op_node.op,
                        .x = try Expression.Operand.init(op_node.x, target.tensor.group),
                    } },
                    .ReduceOp => |op_node| .{
                        .BinaryOp = .{
                            .op = switch (op_node.op) {
                                .Sum => .Add,
                                .Max => .Maximum,
                            },
                            .a = try Expression.Operand.init(op_node.x, target.tensor.group),
                            .b = .{ .local = target.tensor },
                        },
                    },
                    .DataOp => |op_node| if (op_node.op == .AsType) .{ .DataOp = .{
                        .op = op_node.op,
                        .x = try Expression.Operand.init(op_node.x, target.tensor.group),
                    } } else {
                        // Get the statement for the input to the view-changing operation
                        const statement = try Statement.getOrInit(op_node.x);
                        // The statement tensor needs to reflect the target's view of the memory
                        switch (statement.out) {
                            inline else => |out| {
                                out.ndims = target.tensor.ndims;
                                out.shape = target.tensor.shape;
                                out.strides = target.tensor.strides;
                            },
                        }

                        return statement;
                    },
                },
                .out = switch (target.tensor.op_node) {
                    // The output of a reduce op is always an accumulator
                    // A unaryFn op will be pushed in after the reduce loop to assign the accumulator to the global tensor
                    .ReduceOp => .{ .local = target.tensor },
                    else => if (target.tensor.isCached()) .{ .local = target.tensor } else .{ .global = target.tensor },
                },
            };
            const statement = try arena.allocator().create(Statement);
            statement.* = tmp_statement;
            try statements.putNoClobber(target.tensor.uid, statement);
            return statement;
        }
    }
};

test "single loop deserialization" {
    init();
    defer deinit();
    const str =
        \\{"dim":0,"bound":2,"reduce":false,"body":[]}
    ;
    const parsed = try std.json.parseFromSlice(Loop, std.testing.allocator, str, .{});
    defer parsed.deinit();
    const new_str = try std.json.stringifyAlloc(std.testing.allocator, parsed.value, .{});
    defer std.testing.allocator.free(new_str);
    std.debug.assert(std.mem.eql(u8, str, new_str));
}

test "double nested loop deserialization" {
    init();
    defer deinit();
    const str =
        \\{"dim":0,"bound":2,"reduce":false,"body":[{"loop":{"dim":0,"bound":2,"reduce":false,"body":[]}}]}
    ;
    const parsed = try std.json.parseFromSlice(Loop, std.testing.allocator, str, .{});
    defer parsed.deinit();
    const new_str = try std.json.stringifyAlloc(std.testing.allocator, parsed.value, .{});
    defer std.testing.allocator.free(new_str);
    std.debug.assert(std.mem.eql(u8, str, new_str));
}

test "triple nested loop deserialization" {
    init();
    defer deinit();
    const str =
        \\{"dim":0,"bound":2,"reduce":false,"body":[{"loop":{"dim":0,"bound":2,"reduce":false,"body":[{"loop":{"dim":0,"bound":2,"reduce":false,"body":[]}}]}}]}
    ;
    const parsed = try std.json.parseFromSlice(Loop, std.testing.allocator, str, .{});
    defer parsed.deinit();
    const new_str = try std.json.stringifyAlloc(std.testing.allocator, parsed.value, .{});
    defer std.testing.allocator.free(new_str);
    std.debug.assert(std.mem.eql(u8, str, new_str));
}
