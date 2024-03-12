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

pub fn deinit() void {
    arena.deinit();
    _ = gpa.deinit();
}

// TODO: When parsing a JSON, either recreate the nodes too

const ScheduleItemEnum = enum {
    loop,
    statement,
};
const ScheduleItem = union(ScheduleItemEnum) {
    loop: *Loop,
    statement: *Statement,
};

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
        loop.reduce = parsed_loop.reduce;
        loop.bound = parsed_loop.bound;
        loop.from_json = true;
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
    reduce: bool,
    bound: u64,
    tensor: *Graph.TensorNode,
    body: std.MultiArrayList(ScheduleItem),
    from_json: bool = false,

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
        if (loops.contains(target.tensor.uniqueId())) {
            return;
        }
        switch (target.tensor.op_node) {
            .InitOp => if (target.tensor.op_node.InitOp.op == .Input) return,
            else => {},
        }

        const statement = if (try Statement.getOrInit(target)) |stmt| stmt else return;

        switch (target.tensor.op_node) {
            .InitOp => {},
            .ZipOp => |op_node| {
                try Loop.create(op_node.a);
                try Loop.create(op_node.b);
            },
            inline else => |op_node| {
                try Loop.create(op_node.x);
            },
        }

        if (target.fused) {
            return;
        }

        const ndims = target.tensor.ndims;
        var loop_nest: []*Loop = try gpa.allocator().alloc(*Loop, ndims);
        defer gpa.allocator().free(loop_nest);
        for (loop_nest, 0..) |*loop, dim| {
            loop.* = try arena.allocator().create(Loop);
            loop.*.* = .{
                .dim = @intCast(dim),
                .bound = target.tensor.shape[dim],
                .reduce = switch (target.tensor.op_node) {
                    .ReduceOp => |op_node| op_node.dims[dim],
                    else => false,
                },
                .body = .{},
                .tensor = target.tensor,
            };
        }
        // Sort the loops such that all
        std.sort.block(*const Loop, loop_nest, {}, loopCompare);
        // Nest the loops
        const outermost_loop = loop_nest[0];
        for (loop_nest[0 .. ndims - 1], loop_nest[1..]) |outer, inner| {
            try outer.body.append(arena.allocator(), .{ .loop = inner });
        }

        try loop_nest[ndims - 1].body.append(arena.allocator(), .{ .statement = statement });
        try global_body.append(arena.allocator(), .{ .loop = outermost_loop });
        try loops.putNoClobber(target.tensor.uniqueId(), outermost_loop);
        std.debug.print("created loop for {d}\n", .{target.tensor.uniqueId()});
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
        const OperandEnum = enum { tensor, expression };
        const Operand = union(OperandEnum) {
            tensor: *Graph.TensorNode,
            expression: *const Expression,
            fn get(target: Graph.OpNode.Input) std.mem.Allocator.Error!Operand {
                if (operands.get(target.tensor.uniqueId())) |operand| {
                    return operand;
                }

                // TODO: Fix this and Statement so that statement is never null
                // Statement is only null when InitOp == Input
                const operand: Statement.Expression.Operand = if (target.fused) make_operand: {
                    const expr = switch (target.tensor.op_node) {
                        .InitOp => |op_node| make_init_expression: {
                            if (op_node.op == .Input) {
                                break :make_operand .{ .tensor = target.tensor };
                            } else if (try Statement.getOrInit(target)) |stmt| {
                                break :make_init_expression &stmt.expr;
                            }
                            break :make_init_expression null;
                        },
                        .TypeOp => if (try Statement.getOrInit(target)) |stmt| &stmt.expr else null,
                        else => blk: {
                            break :blk if (try Statement.getOrInit(target)) |stmt| &stmt.expr else null;
                        },
                    };
                    if (expr) |expression| {
                        break :make_operand .{ .expression = expression };
                    } else {
                        break :make_operand .{ .tensor = target.tensor };
                    }
                } else .{ .tensor = target.tensor };
                try operands.put(target.tensor.uniqueId(), operand);
                return operand;
            }
        };
        const MapOp = struct {
            op: ops.MapOp,
            x: Operand,
        };
        const ZipOp = struct {
            op: ops.ZipOp,
            a: Operand,
            b: Operand,
        };
        const ReduceOp = struct {
            op: ops.ReduceOp,
            x: Operand,
        };
        const TypeOp = struct {
            op: ops.TypeOp,
            x: Operand,
        };
        const InitOp = struct {
            op: ops.InitOp,
            options: ops.InitOp.Options,
        };
        MapOp: MapOp,
        ZipOp: ZipOp,
        ReduceOp: ReduceOp,
        TypeOp: TypeOp,
        InitOp: InitOp,
    };
    const Output = struct {
        tensor: *Graph.TensorNode,
    };
    group: ?usize,
    expr: Expression,
    out: Output,

    fn getOrInit(target: Graph.OpNode.Input) std.mem.Allocator.Error!?*Statement {
        if (statements.get(target.tensor.uniqueId())) |stmt| {
            return stmt;
        } else {
            const tmp_statement: Statement = .{
                .group = target.tensor.group,
                .expr = switch (target.tensor.op_node) {
                    .InitOp => |op_node| if (op_node.op != .Input) .{ .InitOp = .{
                        .op = op_node.op,
                        .options = op_node.options,
                    } } else return null,
                    .ZipOp => |op_node| .{ .ZipOp = .{
                        .op = op_node.op,
                        .a = try Expression.Operand.get(op_node.a),
                        .b = try Expression.Operand.get(op_node.b),
                    } },
                    .MapOp => |op_node| .{ .MapOp = .{
                        .op = op_node.op,
                        .x = try Expression.Operand.get(op_node.x),
                    } },
                    .ReduceOp => |op_node| .{ .ReduceOp = .{
                        .op = op_node.op,
                        .x = try Expression.Operand.get(op_node.x),
                    } },
                    .TypeOp => |op_node| if (op_node.op == .AsType) .{ .TypeOp = .{
                        .op = op_node.op,
                        .x = try Expression.Operand.get(op_node.x),
                    } } else {
                        std.debug.print("{any}\n", .{op_node.x.tensor.uniqueId()});
                        const statement = try Statement.getOrInit(op_node.x);
                        if (statement) |stmt| {
                            stmt.out.tensor = target.tensor;
                            return stmt;
                        }
                        return statement;
                    },
                },
                .out = .{ .tensor = target.tensor },
            };
            const statement = try arena.allocator().create(Statement);
            statement.* = tmp_statement;
            try statements.putNoClobber(target.tensor.uniqueId(), statement);
            return statement;
        }
    }
};

pub fn create() !void {
    init();
    defer deinit();
    Loop.create(.{ .tensor = Graph.entry() }) catch unreachable;

    const slice = try Json.toJsonCompatibleSlice(global_body);
    defer Json.deinitJsonCompatibleSlice(slice);
    const str = try std.json.stringifyAlloc(gpa.allocator(), slice, .{
        .whitespace = .indent_2,
    });
    defer gpa.allocator().free(str);
    std.debug.print("\n{s}\n", .{str});
}

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
