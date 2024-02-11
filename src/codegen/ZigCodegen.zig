const std = @import("std");
const tensor = @import("../tensor.zig");
const ops = @import("../ops.zig");
const comptimePrint = std.fmt.comptimePrint;
const codegen = @import("../codegen.zig");
const ZigCodegen = @This();
const Graph = @import("../Graph.zig");
const dtypes = @import("../dtypes.zig");

pub fn header(_: *const ZigCodegen, writer: anytype) !void {
    const header_fmt =
        \\const std = @import("std");
        \\pub fn main({s}) !void {{
        \\
    ;
    try writer.print(header_fmt, .{""});
}

pub fn footer(_: *const ZigCodegen, writer: anytype) !void {
    const footer_code =
        \\    std.debug.print("{any}\n", .{tensor_0});
        \\}
        \\
    ;
    try writer.write(footer_code);
}

pub fn storage(_: *const ZigCodegen, writer: anytype, id: usize, comptime dtype: dtypes.DType, size: usize) void {
    const storage_fmt =
        \\var @"%{d}": [{d}]{s} = undefined;
        \\
    ;
    writer.print(storage_fmt, .{ id, size, @typeName(dtype) }) catch unreachable;
}

pub fn memset(writer: anytype, node: *Graph.Node) void {
    const memset_fmt =
        \\@memset(&@"%{d}", {any});
        \\
        \\
    ;
    writer.print(memset_fmt, .{
        node.id,
    }) catch unreachable;
}

const data_read = "@\"%{d}\"[{s}]";
const no_broadcast_loop =
    \\for (0..{{d}}) |i| {{{{
    \\    @"%{{d}}"[i] = {s};  
    \\}}}}
    \\
;

pub fn as_type(
    writer: anytype,
    node_out: *Graph.Node,
) !void {
    const node_x = node_out.link.TypeOp.x;
    const old_dtype = node_x.dtype;
    const new_dtype = node_out.dtype;
    if (old_dtype == new_dtype) {
        return;
    }

    const err_msg = comptimePrint("Cannot cast dtype {} to {}", .{ old_dtype, new_dtype });
    const fmt = std.fmt.allocPrint(no_broadcast_loop, .{
        comptimePrint(switch (new_dtype) {
            dtypes.isFloat(new_dtype) => switch (@typeInfo(old_dtype)) {
                dtypes.isInt(old_dtype) => "@floatFromInt({s})",
                dtypes.isFloat(old_dtype) => "@floatCast({s})",
                else => @compileError(err_msg),
            },
            dtypes.isInt(new_dtype) => switch (@typeInfo(old_dtype)) {
                dtypes.isFloat(old_dtype) => "@intFromFloat({s})",
                .bool => "@intFromBool({s})",
                dtypes.isInt(old_dtype) => "@intCast({s})",
                else => @compileError(err_msg),
            },
            else => @compileError(err_msg),
        }, .{data_read}),
    });
    try writer.print(fmt, .{
        node_out.size,
        node_out.id,
        node_x.id,
        "i",
    });
}

pub fn map(
    writer: anytype,
    node_out: *Graph.Node,
) !void {
    const link = node_out.link.MapOp;
    const node_x = link.x;
    const fmt = comptimePrint(no_broadcast_loop, .{
        comptimePrint(switch (link.op) {
            .Neg => if (node_x.dtype == .bool) "!({s})" else "-({s})",
            .Log2 => "@log2({s})",
            .Exp2 => "@exp2({s})",
            .Sqrt => "@sqrt({s})",
            .Recip => "1.0 / ({s})",
            else => @compileError("Not implemented"),
        }, .{data_read}),
    });
    try writer.print(fmt, .{
        node_out.size,
        node_out.id,
        node_x.id,
        "i",
    });
}

fn zipOpFmt(op: ops.ZipOp) []const u8 {
    return switch (op) {
        .Add => "({s}) + ({s})",
        .Mul => "({s}) * ({s})",
        .Maximum => "@max({s}, {s})",
        .LessThan => "({s}) < ({s})",
        .Equals => "({s}) == ({s})",
        .Xor => "({s}) ^ ({s})",
        else => @compileError("Not implemented"),
    };
}
pub fn zip(
    writer: anytype,
    out: *Graph.Node,
) !void {
    const link = out.link.ZipOp;
    const a = link.a;
    const b = link.b;
    const op_fmt = comptime zipOpFmt(link.op);
    if (A == B) {
        const fmt = comptimePrint(no_broadcast_loop, .{
            comptimePrint(op_fmt, .{
                data_read,
                data_read,
            }),
        });
        try writer.print(fmt, .{ Out.size, out.id, a.id, "i", b.id, "i" });
    } else {
        const loop_fmt =
            \\for (0..{d}) |i_{d}| {{{{{{{{
            \\    {s}
            \\}}}}}}}}
        ;
        const nested_loop_fmt = comptime gen: {
            var tmp: []const u8 = "{s}";
            for (0..Out.ndims) |d| {
                tmp = comptimePrint(loop_fmt, .{ Out.shape[Out.ndims - 1 - d], Out.ndims - 1 - d, tmp });
            }
            break :gen tmp;
        };
        const inner_expression_fmt = comptimePrint("tensor_{{d}}[{{s}}] = {s};", .{
            comptimePrint(op_fmt, .{
                data_read,
                data_read,
            }),
        });
        const final_fmt = comptimePrint(nested_loop_fmt, .{inner_expression_fmt});
        try writer.print(final_fmt ++ "\n", .{
            out.id,
            codegen.unravelMultiIndex(Out, "i_"),
            a.id,
            codegen.broadcastedUnravelMultiIndex(A, Out, "i_"),
            b.id,
            codegen.broadcastedUnravelMultiIndex(B, Out, "i_"),
        });
    }
}

pub fn reduce(
    writer: anytype,
    node_out: *Graph.Node,
) !void {
    const node_x = node_out.link.ReduceOp.x;
    const op_fmt = comptimePrint(switch (node_x.op) {
        .Sum => zipOpFmt(.add),
        .Max => zipOpFmt(.maximum),
    }, .{ "acc", data_read });
    if (node_out.dim == null) {
        const reduce_all_loop =
            \\{{{{
            \\    var acc = @"%{{d}}"[0];
            \\    for (1..{{d}}) |i| {{{{
            \\        acc = {s};
            \\    }}}}
            \\    @"%{{d}}"[0] = acc;
            \\}}}}
            \\
        ;
        const fmt = comptimePrint(reduce_all_loop, .{
            op_fmt,
        });
        try writer.print(fmt, .{
            x.id,
            X.size,
            x.id,
            "i",
            out.id,
        });
    } else {
        const no_acc_loop_fmt =
            \\for (0..{d}) |i_{d}| {{{{{{{{
            \\    {s}
            \\}}}}}}}}
        ;
        const acc_loop_fmt =
            \\const pos = {{{{s}}}};
            \\var acc = @"%{{{{d}}}}"[pos];
            \\for (1..{d}) |i_{d}| {{{{{{{{
            \\    {s}
            \\}}}}}}}}
            \\@"%{{{{d}}}}"[pos] = acc;
            \\
        ;
        const nested_loop_fmt = comptime gen: {
            var tmp: []const u8 = "{s}";
            for (0..X.ndims) |d| {
                if (X.ndims - 1 - d == dim.?) {
                    tmp = comptimePrint(acc_loop_fmt, .{ X.shape[X.ndims - 1 - d], X.ndims - 1 - d, tmp });
                }
            }
            for (0..X.ndims) |d| {
                if (X.ndims - 1 - d != dim.?) {
                    tmp = comptimePrint(no_acc_loop_fmt, .{ X.shape[X.ndims - 1 - d], X.ndims - 1 - d, tmp });
                }
            }
            break :gen tmp;
        };
        const inner_expression_fmt = comptimePrint("acc = {s};", .{op_fmt});
        const final_fmt = comptimePrint(nested_loop_fmt, .{inner_expression_fmt});
        try writer.print(final_fmt, .{
            codegen.unravelMultiIndex(Out, "i_"),
            x.id,
            x.id,
            comptimePrint("pos + i_{d} * {d}", .{ dim.?, X.strides[dim.?] }),
            out.id,
        });
    }
}

// TODO: Test the codegen
// test "cast codegen" {
//     const actual = comptime gen: {
//         var t1 = tensor.range(TestBackend, f32, 4.0, 16.0);
//         var t2 = t1.cast(i32);
//         const cg = @import("codegen/ZigCodegen.zig"){};
//         break :gen cg.cast(i32, &t1, 1, @constCast(&t2), 0);
//     };
//     _ = actual;
//     const expected =
//         \\for (0..12) |i| {
//         \\    t0[i] = @intFromFloat(t1[i]);
//         \\}
//     ;
//     _ = expected;
//     // try expectEqual(expected[0..expected.len].*, actual[0..actual.len].*);
// }

// test "map codegen" {
//     const actual = comptime gen: {
//         var t1 = tensor.range(TestBackend, f32, 4.0, 16.0);
//         var t2 = t1.neg();
//         const cg = @import("codegen/ZigCodegen.zig"){};
//         break :gen cg.map(.neg, &t1, 1, @constCast(&t2), 0);
//     };
//     _ = actual;
//     const expected =
//         \\for (0..12) |i| {
//         \\    t0[i] = -(t1[i]);
//         \\}
//     ;
//     _ = expected;
//     // try expectEqual(expected[0..expected.len].*, actual[0..actual.len].*);
// }

// test "zip no broadcast codegen" {
//     const actual = comptime gen: {
//         var t1 = tensor.range(TestBackend, f32, 4.0, 16.0);
//         var t2 = tensor.range(TestBackend, f32, 4.0, 16.0);
//         var t3 = t1.add(t2);
//         const cg = @import("codegen/ZigCodegen.zig"){};
//         break :gen cg.zip(.add, &t1, &t2, @constCast(&t3));
//     };
//     _ = actual;
//     const expected =
//         \\for (0..12) |i| {
//         \\    t0[i] = (t1[i]) + (t2[i]);
//         \\}
//     ;
//     _ = expected;
//     // try expectEqual(expected[0..expected.len].*, actual[0..actual.len].*);
// }

// test "reduce all codegen" {
//     const actual = comptime gen: {
//         var t1 = tensor.range(TestBackend, f32, 4.0, 16.0);
//         var t2 = t1.sum(null);
//         const cg = @import("codegen/ZigCodegen.zig"){};
//         break :gen cg.reduce(.sum, &t1, 1, null, @constCast(&t2), 0);
//     };
//     _ = actual;
//     const expected =
//         \\{
//         \\    var acc = t1[0];
//         \\    for (1..12) |i| {
//         \\        acc = (acc) + (t1[i]);
//         \\    }
//         \\    t0[0] = acc;
//         \\}
//     ;
//     _ = expected;
//     // try expectEqual(expected[0..expected.len].*, actual[0..actual.len].*);
// }
