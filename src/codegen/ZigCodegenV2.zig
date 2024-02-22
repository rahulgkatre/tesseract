const std = @import("std");
const Program = @import("../Program.zig");
const ops = @import("../ops.zig");
const dtypes = @import("../dtypes.zig");
const codegen = @import("../codegen.zig");
const Graph = @import("../Graph.zig");

fn bodyCodegenCount(body: Program.Body) u64 {
    const slice = body.contents.slice();
    var total_len: u64 = 0;
    for (slice.items(.tags), slice.items(.data)) |tag, data| {
        total_len += switch (tag) {
            .Loop => loopCodegenCount(data.Loop),
            .Statement => statementCodegenCount(data.Statement),
        };
    }
    return total_len;
}

pub fn bodyCodegen(allocator: std.mem.Allocator, body: Program.Body) []const u8 {
    const buf: []u8 = allocator.alloc(u8, bodyCodegenCount(body)) catch unreachable;
    var offset: u64 = 0;
    const slice = body.contents.slice();
    for (slice.items(.tags), slice.items(.data)) |tag, data| {
        const body_codegen = switch (tag) {
            .Loop => loopCodegen(allocator, data.Loop),
            .Statement => statementCodegen(allocator, data.Statement),
        };
        defer allocator.free(body_codegen);
        @memcpy(buf[offset .. offset + body_codegen.len], body_codegen);
        offset += body_codegen.len;
    }
    return buf[0..offset];
}

const acc_loop_fmt =
    \\var acc{d} = {s};
    \\for (0..{d}) |{s}| {{
    \\{s}
    \\}}
    \\T{d}[{s}] = acc{d};
;
const loop_fmt =
    \\for (0..{d}) |{s}| {{
    \\{s}
    \\}}
;
fn loopCodegenCount(loop: *Program.Loop) u64 {
    const loop_var_codegen_count = codegen.loopVarCodegenCount(loop);
    const body_codegen_count = bodyCodegenCount(loop.body);
    if (loop.acc) {
        return std.fmt.count(acc_loop_fmt, .{
            loop.node.id,
            "0; // TODO: Determine accumulator start value when lowering to acc loop",
            loop.upper_bound,
            "",
            "",
            loop.node.id,
            "",
            loop.node.id,
        }) + loop_var_codegen_count + body_codegen_count + codegen.unravelCodegenCount(loop.node);
    } else {
        return std.fmt.count(loop_fmt, .{
            loop.upper_bound,
            "",
            "",
        }) + loop_var_codegen_count + body_codegen_count;
    }
}
pub fn loopCodegen(allocator: std.mem.Allocator, loop: *Program.Loop) []const u8 {
    const buf: []u8 = allocator.alloc(u8, loopCodegenCount(loop)) catch unreachable;
    const loop_var_codegen = codegen.loopVarCodegen(allocator, loop);
    defer allocator.free(loop_var_codegen);

    const body_codegen = bodyCodegen(allocator, loop.body);
    defer allocator.free(body_codegen);

    if (loop.acc) {
        return std.fmt.bufPrint(buf, acc_loop_fmt, .{
            loop.node.id,
            "0; // TODO: Determine accumulator start value when lowering to acc loop",
            loop.upper_bound,
            loop_var_codegen,
            body_codegen,
            loop.node.id,
            codegen.unravelCodegen(allocator, loop.node) catch unreachable,
            loop.node.id,
        }) catch unreachable;
    } else {
        return std.fmt.bufPrint(buf, loop_fmt, .{
            loop.upper_bound,
            loop_var_codegen,
            body_codegen,
        }) catch unreachable;
    }
}

fn mapOpCodegen(allocator: std.mem.Allocator, op: ops.MapOp, dtype: dtypes.DType, x: []const u8) []const u8 {
    return switch (op) {
        .Neg => if (dtypes.isBool(dtype)) std.fmt.allocPrint(allocator, "!({s})", .{x}) else std.fmt.allocPrint(allocator, "-({s})", .{x}),
        .Log2 => std.fmt.allocPrint(allocator, "@log2({s})", .{x}),
        .Exp2 => std.fmt.allocPrint(allocator, "@exp2({s})", .{x}),
        .Sqrt => std.fmt.allocPrint(allocator, "@sqrt({s})", .{x}),
        .Recip => if (dtypes.isFloat(dtype)) std.fmt.allocPrint(allocator, "1.0 / ({s})", .{x}) else if (dtypes.isInt(dtype)) std.fmt.allocPrint(allocator, "@divFloor(1, {s})", .{x}) else unreachable,
        else => unreachable,
    } catch unreachable;
}
fn mapOpCodegenCount(op: ops.MapOp, dtype: dtypes.DType) u64 {
    return (switch (op) {
        .Neg => if (dtypes.isBool(dtype)) "!({s})" else "-({s})",
        .Log2 => "@log2({s})",
        .Exp2 => "@exp2({s})",
        .Sqrt => "@sqrt({s})",
        .Recip => if (dtypes.isFloat(dtype)) "1.0 / ({s})" else if (dtypes.isInt(dtype)) "@divFloor(1, {s})" else unreachable,
        else => unreachable,
    }).len;
}

fn zipOpCodegen(
    allocator: std.mem.Allocator,
    op: ops.ZipOp,
    a: []const u8,
    b: []const u8,
) []const u8 {
    return switch (op) {
        .Add => std.fmt.allocPrint(allocator, "({s}) + ({s})", .{ a, b }),
        .Mul => std.fmt.allocPrint(allocator, "({s}) * ({s})", .{ a, b }),
        .Maximum => std.fmt.allocPrint(allocator, "@max({s}, {s})", .{ a, b }),
        .LessThan => std.fmt.allocPrint(allocator, "({s}) < ({s})", .{ a, b }),
        .Equals => std.fmt.allocPrint(allocator, "({s}) == ({s})", .{ a, b }),
        .Xor => std.fmt.allocPrint(allocator, "({s}) ^ ({s})", .{ a, b }),
        else => unreachable,
    } catch unreachable;
}
fn zipOpCodegenCount(op: ops.ZipOp) u64 {
    return (switch (op) {
        .Add => "({s}) + ({s})",
        .Mul => "({s}) * ({s})",
        .Maximum => "@max({s}, {s})",
        .LessThan => "({s}) < ({s})",
        .Equals => "({s}) == ({s})",
        .Xor => "({s}) ^ ({s})",
        else => unreachable,
    }).len;
}

fn reduceOpCodegen(allocator: std.mem.Allocator, op: ops.ReduceOp, x: []const u8, out_id: usize) []const u8 {
    const zip_op: ops.ZipOp = switch (op) {
        .Sum => .Add,
        .Max => .Maximum,
    };
    const acc = std.fmt.allocPrint(allocator, "acc{d}", .{out_id}) catch unreachable;
    defer allocator.free(acc);
    return zipOpCodegen(
        allocator,
        zip_op,
        acc,
        x,
    );
}

fn reduceOpCodegenCount(op: ops.ReduceOp) u64 {
    const zip_op: ops.ZipOp = switch (op) {
        .Sum => .Add,
        .Max => .Maximum,
    };
    return zipOpCodegenCount(zip_op);
}

pub fn statementCodegenCount(statement: Program.Statement) u64 {
    return switch (statement) {
        .MapOp => |map| std.fmt.count("T{d}[{s}] = {s}", .{ map.out.id, "", "" }) +
            mapOpCodegenCount(map.op, map.x.tensor.dtype) +
            codegen.unravelCodegenCount(map.x) +
            codegen.unravelCodegenCount(map.out),
        .ZipOp => |zip| std.fmt.count("T{d}[{s}] = {s}", .{ zip.out.id, "", "" }) +
            zipOpCodegenCount(zip.op) +
            codegen.broadcastedUnravelCodegenCount(zip.a, zip.out) +
            codegen.broadcastedUnravelCodegenCount(zip.b, zip.out) +
            codegen.unravelCodegenCount(zip.out),
        .ReduceOp => |red| std.fmt.count("acc{d} = {s}", .{ red.out.id, "" }) +
            reduceOpCodegenCount(red.op) +
            codegen.broadcastedUnravelCodegenCount(red.out, red.x) +
            codegen.unravelCodegenCount(red.out),
        else => 0,
    };
}

pub fn statementCodegen(allocator: std.mem.Allocator, statement: Program.Statement) []const u8 {
    switch (statement) {
        .MapOp => |map| {
            const unravel_x = codegen.unravelCodegen(allocator, map.x) catch unreachable;
            defer allocator.free(unravel_x);
            const read_x = std.fmt.allocPrint(allocator, "T{d}[{s}]", .{
                map.x.id,
                unravel_x,
            }) catch unreachable;
            defer allocator.free(read_x);
            const rhs = mapOpCodegen(allocator, map.op, map.x.tensor.dtype, read_x);
            defer allocator.free(rhs);
            return std.fmt.allocPrint(allocator, "T{d}[{s}] = {s};", .{
                map.out.id,
                rhs,
                codegen.unravelCodegen(allocator, map.out) catch unreachable,
            }) catch unreachable;
        },
        .ZipOp => |zip| {
            const unravel_a = codegen.broadcastedUnravelCodegen(allocator, zip.a, zip.out) catch unreachable;
            defer allocator.free(unravel_a);
            const read_a = std.fmt.allocPrint(allocator, "T{d}[{s}]", .{
                zip.a.id,
                unravel_a,
            }) catch unreachable;
            defer allocator.free(read_a);
            const unravel_b = codegen.broadcastedUnravelCodegen(allocator, zip.b, zip.out) catch unreachable;
            defer allocator.free(unravel_b);
            const read_b = std.fmt.allocPrint(allocator, "T{d}[{s}]", .{
                zip.b.id,
                unravel_b,
            }) catch unreachable;
            defer allocator.free(read_b);
            const rhs = zipOpCodegen(allocator, zip.op, read_a, read_b);
            defer allocator.free(rhs);
            return std.fmt.allocPrint(allocator, "T{d}[{s}] = {s};", .{
                zip.out.id,
                rhs,
                codegen.unravelCodegen(allocator, zip.out) catch unreachable,
            }) catch unreachable;
        },
        .ReduceOp => |reduce| {
            const unravel_x = codegen.unravelCodegen(allocator, reduce.x) catch unreachable;
            defer allocator.free(unravel_x);
            const read_x = std.fmt.allocPrint(allocator, "T{d}[{s}]", .{
                reduce.x.id,
                unravel_x,
            }) catch unreachable;
            defer allocator.free(read_x);
            const rhs = reduceOpCodegen(allocator, reduce.op, read_x, reduce.out.id);
            defer allocator.free(rhs);
            return std.fmt.allocPrint(allocator, "acc{d} = {s};", .{
                reduce.out.id,
                rhs,
                // codegen.unravelCodegen(allocator, reduce.out) catch unreachable,
            }) catch unreachable;
        },
        else => return "",
    }
}
