const std = @import("std");
const ops = @import("ops.zig");
const Graph = @import("Graph.zig");

var gpa: std.heap.GeneralPurposeAllocator(.{}) = undefined;
var allocator: std.mem.Allocator = undefined;

var arena: std.heap.ArenaAllocator = undefined;
var loops: std.AutoHashMap(usize, *Loop) = undefined;
var statements: std.AutoHashMap(usize, *Statement) = undefined;
var body: std.MultiArrayList(ScheduleItem) = undefined;

pub fn init(backing_allocator: std.mem.Allocator) void {
    gpa = .{};
    allocator = gpa.allocator();
    arena = std.heap.ArenaAllocator.init(backing_allocator);
    loops = std.AutoHashMap(usize, *Loop).init(arena.allocator());
    statements = std.AutoHashMap(usize, *Statement).init(arena.allocator());
    body = std.MultiArrayList(ScheduleItem){};
}

pub fn deinit() void {
    _ = gpa.deinit();
    arena.deinit();
}

const ScheduleItemEnum = enum {
    loop,
    statement,
};
const ScheduleItem = union(ScheduleItemEnum) {
    loop: *Loop,
    statement: *Statement,
};
pub const Loop = struct {
    dim: u8,
    reduce: bool,
    ptr: usize,
    body: std.MultiArrayList(ScheduleItem),

    fn node(self: *const Loop) *Graph.TensorNode {
        return Graph.TensorNode.get(self.ptr);
    }

    /// Calculate an ordinal value used for sorting a loop nest
    /// Reduce loops are given a larger number so that they are the innermost loops
    /// Original nesting order is preserved as loops are ordered by tensor dimension
    fn ordinal(self: *const Loop) u16 {
        const ndims = self.node().tensor.ndims;
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

    fn create(tensor_node: *const Graph.TensorNode) !*Loop {
        const ndims = tensor_node.tensor.ndims;
        var loop_nest: []*Loop = try allocator.alloc(*Loop, ndims);
        defer allocator.free(loop_nest);
        for (loop_nest, 0..) |*loop, dim| {
            loop.* = try arena.allocator().create(Loop);
            loop.*.* = .{
                .dim = @intCast(dim),
                .reduce = switch (tensor_node.opNode().*) {
                    .ReduceOp => |op_node| op_node.dims[dim],
                    else => false,
                },
                .body = std.MultiArrayList(ScheduleItem){},
                .ptr = tensor_node.tensor.ptr,
            };
        }
        std.sort.block(*const Loop, loop_nest, {}, loopCompare);
        const outermost_loop = loop_nest[0];
        var current_loop = outermost_loop;
        for (loop_nest[1..]) |loop| {
            const pre_length = current_loop.body.len;
            try current_loop.body.append(arena.allocator(), .{ .loop = loop });
            std.debug.assert(pre_length + 1 == current_loop.body.len);
            current_loop = loop;
        }
        try loops.put(tensor_node.tensor.ptr, outermost_loop);
        const str = try std.json.stringifyAlloc(allocator, outermost_loop, .{});
        defer allocator.free(str);
        std.debug.print("{s}\n", .{str});
        const parsed = try std.json.parseFromSlice(Loop, allocator, str, .{});
        defer parsed.deinit();
        std.debug.print("{any}\n", .{parsed.value});
        return outermost_loop;
    }

    /// MultiArrayList does not automatically stringify so the default jsonStringify function
    /// does not work. To make loops JSON compatible, the body is replaced with a slice of unions.
    /// Deinit functions are provided as JsonCompatibleLoops are created with an allocator.
    const JsonComptatibleItem = union(ScheduleItemEnum) {
        loop: *JsonCompatibleLoop,
        statement: *Statement,

        fn deinit(self: *JsonComptatibleItem) void {
            switch (self.*) {
                .loop => |loop| loop.deinit(),
                else => {},
            }
            allocator.destroy(self);
        }
    };
    const JsonCompatibleLoop = struct {
        dim: u8,
        reduce: bool,
        ptr: usize,
        body: []JsonComptatibleItem,

        fn deinit(self: *JsonCompatibleLoop) void {
            for (self.body) |*item| {
                item.deinit();
            }
            allocator.destroy(self);
        }
    };

    fn toJsonCompatible(self: *const Loop) !*JsonCompatibleLoop {
        const body_as_slice = try allocator.alloc(JsonComptatibleItem, self.body.len);
        for (self.body.items(.tags), self.body.items(.data), 0..) |tag, data, i| {
            switch (tag) {
                .loop => body_as_slice[i] = @unionInit(JsonComptatibleItem, @tagName(ScheduleItemEnum.loop), try toJsonCompatible(data.loop)),
                inline else => |active| body_as_slice[i] = @unionInit(
                    JsonComptatibleItem,
                    @tagName(active),
                    @field(data, @tagName(active)),
                ),
            }
        }

        const compatible_loop = try allocator.create(JsonCompatibleLoop);
        compatible_loop.* = .{
            .dim = self.dim,
            .reduce = self.reduce,
            .ptr = self.ptr,
            .body = body_as_slice,
        };
        return compatible_loop;
    }

    /// Custom jsonStringify implementation that converts a Loop to a JsonCompatibleLoop
    /// and generates the Json string from the new loop
    pub fn jsonStringify(self: *const Loop, write_stream: anytype) !void {
        const compatible_loop = try toJsonCompatible(self);
        defer compatible_loop.deinit();
        try write_stream.write(compatible_loop);
    }

    fn fromJsonCompatible(parsed_loop: *const JsonCompatibleLoop) !*Loop {
        // Use the arena allocator because the parsed loop will join the other loops of the schedule
        var loop: *Loop = try arena.allocator().create(Loop);
        loop.dim = parsed_loop.dim;
        loop.reduce = parsed_loop.reduce;
        loop.ptr = parsed_loop.ptr;
        loop.body = .{};
        try loop.body.ensureTotalCapacity(arena.allocator(), parsed_loop.body.len);
        for (parsed_loop.body) |item| {
            const converted_item = switch (item) {
                .loop => |inner_loop| @unionInit(ScheduleItem, @tagName(ScheduleItemEnum.loop), try fromJsonCompatible(inner_loop)),
                .statement => |inner_stmt| @unionInit(ScheduleItem, @tagName(ScheduleItemEnum.statement), inner_stmt),
            };
            loop.body.appendAssumeCapacity(converted_item);
        }
        return loop;
    }

    /// Custom jsonStringify implementation that converts a parsed JsonCompatibleLoop
    /// into a regular loop
    pub fn jsonParse(parse_allocator: std.mem.Allocator, source: *std.json.Scanner, options: std.json.ParseOptions) !Loop {
        const parsed = try std.json.parseFromTokenSource(JsonCompatibleLoop, parse_allocator, source, options);
        defer parsed.deinit();
        const loop = try fromJsonCompatible(&parsed.value);
        return loop.*;
    }
};

const Statement = struct {
    const Input = struct {
        ptr: usize,
        inner_expression: ?*Expression,
        fn init(target: Graph.OpNode.Input) Input {
            return .{
                .ptr = target.ptr,
                .inner_expression = if (target.fused) &statements.get(target.ptr).?.expr else null,
            };
        }
        fn node(self: *const Input) *Graph.TensorNode {
            return Graph.TensorNode.get(self.ptr);
        }
    };
    const Expression = union(ops.OpTypes) {
        const InitExpression = struct { op: ops.InitOp, value: ops.InitValue };
        fn UnaryExpression(comptime Op: type) type {
            return struct { op: Op, x: Input };
        }
        fn BinaryExpression(comptime Op: type) type {
            return struct { op: Op, a: Input, b: Input };
        }

        MapOp: UnaryExpression(ops.MapOp),
        ZipOp: BinaryExpression(ops.ZipOp),
        ReduceOp: UnaryExpression(ops.ReduceOp),
        TypeOp: UnaryExpression(ops.TypeOp),
        InitOp: InitExpression,
    };
    const Output = struct {
        ptr: usize,
        fn node(self: *const Input) *Graph.TensorNode {
            return Graph.TensorNode.get(self.ptr);
        }
    };
    group: usize,
    expr: Expression,
    out: Output,
};

pub fn create() void {
    init(std.heap.page_allocator);
    defer deinit();
    _ = Loop.create(Graph.entry()) catch unreachable;
}

test "double nested loop deserialization" {
    init(std.testing.allocator);
    defer deinit();
    const str =
        \\{"dim":0,"reduce":false,"ptr":0,"body":[{"loop":{"dim":1,"reduce":false,"ptr":0,"body":[]}}]}
    ;
    const parsed = try std.json.parseFromSlice(Loop, std.testing.allocator, str, .{});
    defer parsed.deinit();
    const new_str = try std.json.stringifyAlloc(std.testing.allocator, parsed.value, .{});
    defer std.testing.allocator.free(new_str);
    std.debug.assert(std.mem.eql(u8, str, new_str));
}

test "triple nested loop deserialization" {
    init(std.testing.allocator);
    defer deinit();
    const str =
        \\{"dim":0,"reduce":false,"ptr":0,"body":[{"loop":{"dim":1,"reduce":false,"ptr":0,"body":[{"loop":{"dim":2,"reduce":false,"ptr":0,"body":[]}}]}}]}
    ;
    const parsed = try std.json.parseFromSlice(Loop, std.testing.allocator, str, .{});
    defer parsed.deinit();
    const new_str = try std.json.stringifyAlloc(std.testing.allocator, parsed.value, .{});
    defer std.testing.allocator.free(new_str);
    std.debug.assert(std.mem.eql(u8, str, new_str));
}
