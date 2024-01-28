const std = @import("std");
const tensor = @import("../tensor.zig");
const ops = @import("../ops.zig");
const Backend = @import("../backend.zig").Backend;
const comptimePrint = std.fmt.comptimePrint;
const codegen = @import("../codegen.zig");
const ZigCodegen = @This();

const data_read = "t{d}[{s}]";
const no_broadcast_loop =
    \\for (0..{d}) |i| {{
    \\    t{d}[i] = {s};  
    \\}}
    \\
;

pub fn cast(
    _: *const ZigCodegen,
    comptime new_dtype: type,
    x_ptr: anytype,
    comptime x_id: usize,
    out: *@TypeOf(x_ptr.*).Cast(new_dtype),
    comptime out_id: usize,
) []const u8 {
    const Out: type = @TypeOf(out.*);
    const out_dtype: type = @TypeOf(x_ptr.*).dtype;
    const err_msg = comptimePrint("Cannot cast dtype {} to {}", .{ out_dtype, new_dtype });
    const op_fmt = switch (@typeInfo(new_dtype)) {
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
    };
    return comptimePrint(no_broadcast_loop, .{
        Out.size,
        out_id,
        comptimePrint(op_fmt, .{comptimePrint(data_read, .{ x_id, "i" })}),
    });
}

pub fn map(
    _: *const ZigCodegen,
    comptime op: ops.MapOp,
    x_ptr: anytype,
    x_id: usize,
    out: *@TypeOf(x_ptr.*),
    out_id: usize,
) []const u8 {
    const Out: type = @TypeOf(out.*);
    const op_fmt = switch (op) {
        .Neg => if (@typeInfo(Out.dtype) == .Bool) "!({s})" else "-({s})",
        .Log2 => "@log2({s})",
        .Exp2 => "@exp2({s})",
        .Sqrt => "@sqrt({s})",
        .Recip => "1.0 / ({s})",
        else => @compileError("Not implemented"),
    };
    return comptimePrint(no_broadcast_loop, .{
        Out.size,
        out_id,
        comptimePrint(op_fmt, .{comptimePrint(data_read, .{ x_id, "i" })}),
    });
}

const broadcast_loop =
    \\for (0..{d}) |i| {{
    \\    const idx = {s};
    \\    const pos = {s};
    \\    t{d}[i] = {s};
    \\}}
    \\
;
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
    comptime op: ops.ZipOp,
    a_ptr: anytype,
    comptime a_id: usize,
    b_ptr: anytype,
    comptime b_id: usize,
    out: *@TypeOf(a_ptr.*).Broadcast(@TypeOf(b_ptr.*)),
    comptime out_id: usize,
) []const u8 {
    const Out: type = @TypeOf(out.*);
    const A: type = @TypeOf(a_ptr.*);
    const B: type = @TypeOf(b_ptr.*);
    const op_fmt = zipOpFmt(op);
    if (A == B) {
        return comptimePrint(no_broadcast_loop, .{
            Out.size,
            out_id,
            comptimePrint(op_fmt, .{
                comptimePrint(data_read, .{ a_id, "i" }),
                comptimePrint(data_read, .{ b_id, "ising" }),
            }),
        });
    } else {
        return comptimePrint(broadcast_loop, .{
            Out.size,
            codegen.posToIdx(Out),
            out_id,
            comptimePrint(op_fmt, .{
                comptimePrint(data_read, .{ a_id, codegen.broadcastIdxToPos(A, Out, "idx") }),
                comptimePrint(data_read, .{ b_id, codegen.broadcastIdxToPos(B, Out, "idx") }),
            }),
        });
    }
}

const reduce_all_loop =
    \\{{
    \\    var acc = t{d}[0];
    \\    for (1..{d}) |i| {{
    \\        acc = {s};
    \\    }}
    \\    t{d}[0] = acc;
    \\}}
    \\
;
const reduce_dim_loop =
    \\for (0..{d}) |i| {{
    \\    const idx = {s};
    \\    const pos = {s};
    \\    var acc = t{d}[pos];
    \\    for (1..{d}) |j| {{
    \\        acc = {s};
    \\    }}
    \\    t{d}[i] = acc;
    \\}}
    \\
;
pub fn reduce(
    _: *const ZigCodegen,
    comptime op: ops.ReduceOp,
    x_ptr: anytype,
    comptime x_id: usize,
    comptime dim: ?u8,
    out: *@TypeOf(x_ptr.*).Reduce(dim),
    comptime out_id: usize,
) []const u8 {
    const Out: type = @TypeOf(out.*);
    const X: type = @TypeOf(x_ptr.*);
    const op_fmt = comptimePrint(switch (op) {
        .Sum => zipOpFmt(.Add),
        .Max => zipOpFmt(.Maximum),
    }, .{ "acc", "{s}" });
    if (dim == null) {
        return comptimePrint(reduce_all_loop, .{
            x_id,
            X.size,
            comptimePrint(op_fmt, .{comptimePrint(data_read, .{ x_id, "i" })}),
            out_id,
        });
    } else {
        return comptimePrint(reduce_dim_loop, .{
            Out.size,
            codegen.posToIdx(@TypeOf(out.*), "i"),
            codegen.idxToPos(@TypeOf(x_ptr.*), "idx"),
            X.shape[dim.?],
            comptimePrint(op_fmt, .{comptimePrint(data_read, .{ x_id, comptimePrint("pos + j * {d}", .{X.strides[dim.?]}) })}),
            out_id,
        });
    }
}
