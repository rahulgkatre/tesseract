const std = @import("std");
const Program = @import("../Program.zig");
const ops = @import("../ops.zig");
const dtypes = @import("../dtypes.zig");
const codegen = @import("../codegen.zig");
const Graph = @import("../Graph.zig");

var arena: std.heap.ArenaAllocator = undefined;
var allocator: std.mem.Allocator = undefined;

pub fn init(backing_allocator: std.mem.Allocator) void {
    arena = std.heap.ArenaAllocator.init(backing_allocator);
    allocator = arena.allocator();
}

pub fn deinit() void {
    arena.deinit();
}

pub fn code(program: *const Program, writer: anytype) !void {
    return try bodyCode(program, program.body, writer);
}

const acc_loop_header_fmt =
    \\var acc{d} = {s};
    \\for (0..{d}) |{s}| {{
    \\
;
const reg_loop_header_fmt =
    \\for (0..{d}) |{s}| {{
    \\
;
fn headerCode(loop: *Program.Loop, writer: anytype) std.mem.Allocator.Error!void {
    const loop_var_code = try codegen.loopVarCode(allocator, loop);
    if (loop.acc) {
        return writer.print(acc_loop_header_fmt, .{ loop.tensor_node.tensor.id(), "0", loop.upper_bound, loop_var_code });
    } else {
        return writer.print(reg_loop_header_fmt, .{ loop.upper_bound, loop_var_code });
    }
}
fn bodyCode(program: *const Program, body: Program.Body, writer: anytype) std.mem.Allocator.Error!void {
    const slice = body.contents.slice();
    for (slice.items(.tags), slice.items(.data)) |tag, data| {
        switch (tag) {
            .Loop => try loopCode(program, data.Loop, writer),
            .Statement => {
                writer.print("{s} = {s};\n", .{
                    switch (data.Statement.*) {
                        .ReduceOp => |reduce| try std.fmt.allocPrint(allocator, "acc{d}", .{reduce.out.tensor.id()}),
                        inline else => |stmt| try std.fmt.allocPrint(allocator, "T{d}[{s}]", .{
                            stmt.out.tensor.id(),
                            try codegen.unravelCode(allocator, stmt.out),
                        }),
                    },
                    (try statementCode(data.Statement)).?,
                });
            },
        }
    }
}
const acc_loop_footer_fmt =
    \\}}
    \\T{d}[{s}] = acc{d};
    \\
;
const reg_loop_footer_fmt =
    \\}}
    \\
;
fn footerCode(loop: *Program.Loop, writer: anytype) !void {
    if (loop.acc) {
        const unravel_code = try codegen.unravelCode(allocator, loop.tensor_node);
        return writer.print(acc_loop_footer_fmt, .{ loop.tensor_node.tensor.id(), unravel_code, loop.tensor_node.tensor.id() });
    } else {
        return writer.print(reg_loop_footer_fmt, .{});
    }
}

fn loopCode(program: *const Program, loop: *Program.Loop, writer: anytype) std.mem.Allocator.Error!void {
    if (loop.tensor_node.group == program.group) {
        try headerCode(loop, writer);
        try bodyCode(program, loop.body, writer);
        try footerCode(loop, writer);
    }
}

fn mapOpCode(op: ops.MapOp, dtype: dtypes.DType, x: []const u8) ![]const u8 {
    return try switch (op) {
        .Copy => std.fmt.allocPrint(allocator, "{s}", .{x}),
        .Neg => if (dtypes.isBool(dtype)) std.fmt.allocPrint(allocator, "!({s})", .{x}) else std.fmt.allocPrint(allocator, "-({s})", .{x}),
        .Log => std.fmt.allocPrint(allocator, "@log({s})", .{x}),
        .Exp => std.fmt.allocPrint(allocator, "@exp({s})", .{x}),
        .Sqrt => std.fmt.allocPrint(allocator, "@sqrt({s})", .{x}),
        .Recip => if (dtypes.isFloat(dtype)) std.fmt.allocPrint(allocator, "1.0 / ({s})", .{x}) else if (dtypes.isInt(dtype)) std.fmt.allocPrint(allocator, "@divFloor(1, {s})", .{x}) else unreachable,
        else => unreachable,
    };
}

fn zipOpCode(
    op: ops.ZipOp,
    a: []const u8,
    b: []const u8,
) ![]const u8 {
    return try switch (op) {
        .Add => std.fmt.allocPrint(allocator, "({s}) + ({s})", .{ a, b }),
        .Mul => std.fmt.allocPrint(allocator, "({s}) * ({s})", .{ a, b }),
        .Maximum => std.fmt.allocPrint(allocator, "@max({s}, {s})", .{ a, b }),
        .LessThan => std.fmt.allocPrint(allocator, "({s}) < ({s})", .{ a, b }),
        .Equals => std.fmt.allocPrint(allocator, "({s}) == ({s})", .{ a, b }),
        .Xor => std.fmt.allocPrint(allocator, "({s}) ^ ({s})", .{ a, b }),
        else => unreachable,
    };
}

fn reduceOpCode(op: ops.ReduceOp, x: []const u8, out_id: usize) ![]const u8 {
    const zip_op: ops.ZipOp = switch (op) {
        .Sum => .Add,
        .Max => .Maximum,
    };
    const acc = try std.fmt.allocPrint(allocator, "acc{d}", .{out_id});
    return zipOpCode(
        zip_op,
        acc,
        x,
    );
}

fn statementCode(statement: ?*const Program.Statement) std.mem.Allocator.Error!?[]const u8 {
    if (statement == null) {
        return null;
    }
    switch (statement.?.*) {
        .MapOp => |map| {
            const inner_x = try statementCode(map.x_statement) orelse try std.fmt.allocPrint(allocator, "T{d}[{s}]", .{
                map.x.tensor.id(),
                try codegen.unravelCode(allocator, map.x),
            });
            return try mapOpCode(map.op, map.x.tensor.dtype, inner_x);
        },
        .ZipOp => |zip| {
            const inner_a = try statementCode(zip.a_statement) orelse try std.fmt.allocPrint(allocator, "T{d}[{s}]", .{
                zip.a.tensor.id(),
                try codegen.broadcastedUnravelCode(allocator, zip.a, zip.out),
            });
            const inner_b = try statementCode(zip.b_statement) orelse try std.fmt.allocPrint(allocator, "T{d}[{s}]", .{
                zip.b.tensor.id(),
                try codegen.broadcastedUnravelCode(allocator, zip.b, zip.out),
            });
            return try zipOpCode(zip.op, inner_a, inner_b);
        },
        .ReduceOp => |reduce| {
            const inner_x = try statementCode(reduce.x_statement) orelse try std.fmt.allocPrint(allocator, "T{d}[{s}]", .{
                reduce.x.tensor.id(),
                try codegen.unravelCode(allocator, reduce.x),
            });
            return try reduceOpCode(reduce.op, inner_x, reduce.out.tensor.id());
        },
        .InitOp => |initialize| {
            if (initialize.op != .Input) {
                return switch (initialize.init) {
                    .Full => |value| try std.fmt.allocPrint(allocator, "{s}", .{value}),
                    .Range => |range| try std.fmt.allocPrint(allocator, "{s}+d0", .{range.start}),
                    .Rand => |dtype| try std.fmt.allocPrint(allocator, "std.rand.random.floatNorm({s})", .{@tagName(dtype)}),
                    else => unreachable,
                };
            } else {
                return null;
            }
        },
        .TypeOp => |typing| {
            if (typing.op == .AsType) {
                const inner_x = try statementCode(typing.x_statement.?) orelse try std.fmt.allocPrint(allocator, "T{d}[{s}]", .{
                    typing.x.tensor.id(),
                    try codegen.unravelCode(allocator, typing.x),
                });
                defer allocator.free(inner_x);
                return "TODO";
            } else {
                return try statementCode(typing.x_statement.?);
            }
        },
    }
}
