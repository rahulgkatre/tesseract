const std = @import("std");
const tensor = @import("../tensor.zig");
const ops = @import("../ops.zig");
const Backend = @import("../backend.zig").Backend;
const comptimePrint = std.fmt.comptimePrint;
const codegen = @import("../codegen.zig");
const ZigCodegen = @This();

pub fn header(_: *const ZigCodegen, writer: anytype) void {
    const header_fmt =
        \\const std = @import("std");
        \\pub fn main({s}) !void {{
        \\
    ;
    _ = writer.print(header_fmt, .{""}) catch unreachable;
}

pub fn footer(_: *const ZigCodegen, writer: anytype) void {
    const footer_code =
        \\    //std.debug.print("{any}\n", .{tensor_0});
        \\}
        \\
    ;
    writer.write(footer_code) catch unreachable;
}

pub fn alloc(_: *const ZigCodegen, writer: anytype, id: usize, comptime dtype: type, size: usize) void {
    const alloc_fmt =
        \\var tensor_{d}: [{d}]{s} = undefined;
        \\
    ;
    writer.print(alloc_fmt, .{ id, size, @typeName(dtype) }) catch unreachable;
}

pub fn memset(_: *const ZigCodegen, writer: anytype, id: usize, comptime dtype: type, value: dtype) void {
    const memset_fmt =
        \\@memset(&tensor_{d}, {any});
        \\
        \\
    ;
    writer.print(memset_fmt, .{ id, value }) catch unreachable;
}

const data_read = "tensor_{d}[{s}]";
const no_broadcast_loop =
    \\for (0..{s}) |i| {{{{
    \\    tensor_{s}[i] = {s};  
    \\}}}}
    \\
;

pub fn cast(
    _: *const ZigCodegen,
    writer: anytype,
    comptime new_dtype: type,
    x: anytype,
    out: *@TypeOf(x.*).Cast(new_dtype),
) !void {
    const Out: type = @TypeOf(out.*);
    const out_dtype: type = @TypeOf(x.*).dtype;
    const err_msg = comptimePrint("Cannot cast dtype {} to {}", .{ out_dtype, new_dtype });
    const fmt = comptimePrint(no_broadcast_loop, .{
        "{d}",
        "{d}",
        comptimePrint(switch (@typeInfo(new_dtype)) {
            .Float => switch (@typeInfo(out_dtype)) {
                .Int => "@floatFromInt({s})",
                .Float => "@floatCast({s})",
                else => @compileError(err_msg),
            },
            .Int => switch (@typeInfo(out_dtype)) {
                .Float => "@intFromFloat({s})",
                .Bool => "@intFromBool({s})",
                .Int => "@intCast({s})",
                else => @compileError(err_msg),
            },
            else => @compileError(err_msg),
        }, .{data_read}),
    });
    try writer.print(fmt, .{
        Out.size,
        out.id,
        x.id,
        "i",
    });
}

pub fn map(
    _: *const ZigCodegen,
    writer: anytype,
    comptime op: ops.MapOp,
    x: anytype,
    out: *@TypeOf(x.*),
) !void {
    const Out: type = @TypeOf(out.*);
    const fmt = comptimePrint(no_broadcast_loop, .{
        "{d}",
        "{d}",
        comptimePrint(switch (op) {
            .Neg => if (@typeInfo(Out.dtype) == .Bool) "!({s})" else "-({s})",
            .Log2 => "@log2({s})",
            .Exp2 => "@exp2({s})",
            .Sqrt => "@sqrt({s})",
            .Recip => "1.0 / ({s})",
            else => @compileError("Not implemented"),
        }, .{data_read}),
    });
    // @compileLog(fmt);
    try writer.print(fmt, .{
        Out.size,
        out.id,
        x.id,
        "i",
    });
}

fn zipOpFmt(comptime op: ops.ZipOp) []const u8 {
    return comptime switch (op) {
        .Add => "({s}) + ({s})",
        .Mul => "({s}) * ({s})",
        .Maximum => "@max({s}, {s})",
        .Lt => "({s}) < ({s})",
        .Eq => "({s}) == ({s})",
        .Xor => "({s}) ^ ({s})",
        else => @compileError("Not implemented"),
    };
}
pub fn zip(
    _: *const ZigCodegen,
    writer: anytype,
    comptime op: ops.ZipOp,
    a: anytype,
    b: anytype,
    out: *@TypeOf(a.*).Broadcast(@TypeOf(b.*)),
) !void {
    const Out: type = @TypeOf(out.*);
    const A: type = @TypeOf(a.*);
    const B: type = @TypeOf(b.*);
    const op_fmt = comptime zipOpFmt(op);
    if (A == B) {
        const fmt = comptimePrint(no_broadcast_loop, .{
            "{d}",
            "{d}",
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
            codegen.idxToPos(Out, "i_"),
            a.id,
            codegen.broadcastIdxToPos(A, Out, "i_"),
            b.id,
            codegen.broadcastIdxToPos(B, Out, "i_"),
        });
    }
}

pub fn reduce(
    _: *const ZigCodegen,
    writer: anytype,
    comptime op: ops.ReduceOp,
    x: anytype,
    comptime dim: ?u8,
    out: *@TypeOf(x.*).Reduce(dim),
) !void {
    const Out: type = @TypeOf(out.*);
    const X: type = @TypeOf(x.*);
    const op_fmt = comptimePrint(switch (op) {
        .Sum => zipOpFmt(.Add),
        .Max => zipOpFmt(.Maximum),
    }, .{ "acc", data_read });
    if (dim == null) {
        const reduce_all_loop =
            \\{{{{
            \\    var acc = tensor_{{d}}[0];
            \\    for (1..{{d}}) |i| {{{{
            \\        acc = {s};
            \\    }}}}
            \\    tensor_{{d}}[0] = acc;
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
            \\var acc = tensor_{{{{d}}}}[pos];
            \\for (1..{d}) |i_{d}| {{{{{{{{
            \\    {s}
            \\}}}}}}}}
            \\tensor_{{{{d}}}}[pos] = acc;
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
            codegen.idxToPos(Out, "i_"),
            x.id,
            x.id,
            comptimePrint("pos + i_{d} * {d}", .{ dim.?, X.strides[dim.?] }),
            out.id,
        });
    }
}
