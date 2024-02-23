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
    defer allocator.free(loop_var_code);
    if (loop.acc) {
        return writer.print(acc_loop_header_fmt, .{ loop.node.id, "0", loop.upper_bound, loop_var_code });
    } else {
        return writer.print(reg_loop_header_fmt, .{ loop.upper_bound, loop_var_code });
    }
}
pub fn bodyCode(body: Program.Body, writer: anytype) std.mem.Allocator.Error!void {
    const slice = body.contents.slice();
    for (slice.items(.tags), slice.items(.data)) |tag, data| {
        switch (tag) {
            .Loop => try loopCode(data.Loop, writer),
            .Statement => writer.print("{s}", .{try statementCode(data.Statement)}),
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
        const unravel_code = try codegen.unravelCode(allocator, loop.node);
        defer allocator.free(unravel_code);
        return writer.print(acc_loop_footer_fmt, .{ loop.node.id, unravel_code, loop.node.id });
    } else {
        return writer.print(reg_loop_footer_fmt, .{});
    }
}

pub fn loopCode(loop: *Program.Loop, writer: anytype) std.mem.Allocator.Error!void {
    try headerCode(loop, writer);
    try bodyCode(loop.body, writer);
    try footerCode(loop, writer);
}

fn mapOpCode(op: ops.MapOp, dtype: dtypes.DType, x: []const u8) ![]const u8 {
    return try switch (op) {
        .Id => std.fmt.allocPrint(allocator, "{s}", .{x}),
        .Neg => if (dtypes.isBool(dtype)) std.fmt.allocPrint(allocator, "!({s})", .{x}) else std.fmt.allocPrint(allocator, "-({s})", .{x}),
        .Log2 => std.fmt.allocPrint(allocator, "@log2({s})", .{x}),
        .Exp2 => std.fmt.allocPrint(allocator, "@exp2({s})", .{x}),
        .Sqrt => std.fmt.allocPrint(allocator, "@sqrt({s})", .{x}),
        .Recip => if (dtypes.isFloat(dtype)) std.fmt.allocPrint(allocator, "1.0 / ({s})", .{x}) else if (dtypes.isInt(dtype)) std.fmt.allocPrint(allocator, "@divFloor(1, {s})", .{x}) else unreachable,
        else => unreachable,
    };
}
fn mapOpCodeLen(op: ops.MapOp, dtype: dtypes.DType) u64 {
    return (switch (op) {
        .Id => "{s}",
        .Neg => if (dtypes.isBool(dtype)) "!({s})" else "-({s})",
        .Log2 => "@log2({s})",
        .Exp2 => "@exp2({s})",
        .Sqrt => "@sqrt({s})",
        .Recip => if (dtypes.isFloat(dtype)) "1.0 / ({s})" else if (dtypes.isInt(dtype)) "@divFloor(1, {s})" else unreachable,
        else => unreachable,
    }).len;
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
    defer allocator.free(acc);
    return zipOpCode(
        zip_op,
        acc,
        x,
    );
}

pub fn statementCode(statement: Program.Statement) std.mem.Allocator.Error![]const u8 {
    switch (statement) {
        .MapOp => |map| {
            const unravel_x = try codegen.unravelCode(allocator, map.x);
            defer allocator.free(unravel_x);
            const read_x = try std.fmt.allocPrint(allocator, "T{d}[{s}]", .{
                map.x.id,
                unravel_x,
            });
            defer allocator.free(read_x);
            const rhs = try mapOpCode(map.op, map.x.tensor.dtype, read_x);
            defer allocator.free(rhs);
            return try std.fmt.allocPrint(allocator, "T{d}[{s}] = {s};", .{
                map.out.id,
                try codegen.unravelCode(allocator, map.out),
                rhs,
            });
        },
        .ZipOp => |zip| {
            const unravel_a = try codegen.broadcastedUnravelCode(allocator, zip.a, zip.out);
            defer allocator.free(unravel_a);
            const read_a = try std.fmt.allocPrint(allocator, "T{d}[{s}]", .{
                zip.a.id,
                unravel_a,
            });
            defer allocator.free(read_a);
            const unravel_b = try codegen.broadcastedUnravelCode(allocator, zip.b, zip.out);
            defer allocator.free(unravel_b);
            const read_b = try std.fmt.allocPrint(allocator, "T{d}[{s}]", .{
                zip.b.id,
                unravel_b,
            });
            defer allocator.free(read_b);
            const rhs = try zipOpCode(zip.op, read_a, read_b);
            defer allocator.free(rhs);
            return try std.fmt.allocPrint(allocator, "T{d}[{s}] = {s};", .{
                zip.out.id,
                try codegen.unravelCode(allocator, zip.out),
                rhs,
            });
        },
        .ReduceOp => |reduce| {
            const unravel_x = try codegen.unravelCode(allocator, reduce.x);
            defer allocator.free(unravel_x);
            const read_x = try std.fmt.allocPrint(allocator, "T{d}[{s}]", .{
                reduce.x.id,
                unravel_x,
            });
            defer allocator.free(read_x);
            const rhs = try reduceOpCode(reduce.op, read_x, reduce.out.id);
            defer allocator.free(rhs);
            return try std.fmt.allocPrint(allocator, "acc{d} = {s};", .{
                reduce.out.id,
                rhs,
            });
        },
        .InitOp => |initialize| {
            if (initialize.op != .Input) {
                const rhs = switch (initialize.init) {
                    .Full => |value| try std.fmt.allocPrint(allocator, "{s}", .{value}),
                    .Range => |range| try std.fmt.allocPrint(allocator, "{s}+d0", .{range.start}),
                    .Rand => "random()",
                    else => "",
                };
                return try std.fmt.allocPrint(allocator, "T{d}[{s}] = {s};", .{
                    initialize.out.id,
                    try codegen.unravelCode(allocator, initialize.out),
                    rhs,
                });
            } else {
                return "";
            }
        },
        else => return "",
    }
}
