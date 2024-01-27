const std = @import("std");
const tensor = @import("../tensor.zig");
const ops = @import("../ops.zig");
const Backend = @import("../backend.zig").Backend;
const comptimePrint = std.fmt.comptimePrint;
const codegen = @import("../codegen.zig");
const ZigCodegen = @This();

const DATA_CONST_INDEX = "t{d}[i]";
const DATA_VAR_INDEX = "t{d}[{s}]";

// TODO
// Codegen for allocating storage

const NO_BROADCAST_LOOP_FMT =
    \\for (0..{d}) |i| {{
    \\    t{d}[i] = {s};  
    \\}}
;
const BROADCAST_LOOP_FMT =
    \\for (0..{d}) |i| {{
    \\    const idx = {s}; 
    \\    t{d}[i] = {s};
    \\}}
;
const REDUCE_ALL_FMT =
    \\{{
    \\    var acc = t{d}[0];
    \\    for (1..{d}) |i| {{
    \\        acc = {s};
    \\    }}
    \\    t{d}[0] = acc;
    \\}}
;
const REDUCE_DIM_FMT =
    \\for (0..{d}) |i| {{
    \\    const idx = {s}
    \\    const pos = {s};
    \\    var acc = t{d}[pos];
    \\    for (1..{d}) |j| {{
    \\    acc = {s};
    \\    }}
    \\    t{d}[i] = acc;
    \\}}
;

fn getCastFmt(comptime new_dtype: type, comptime old_dtype: type) []const u8 {
    const err_msg = comptimePrint("Cannot cast dtype {} to {}", .{ old_dtype, new_dtype });
    return comptime switch (@typeInfo(new_dtype)) {
        .Float => switch (@typeInfo(old_dtype)) {
            .Int => "@floatFromInt({s})",
            .Float => "@floatCast({s})",
            else => @compileError(err_msg),
        },
        .Int => switch (@typeInfo(old_dtype)) {
            .Float => "@intFromFloat({s})",
            .Bool => "@intFromBool({s})",
            .Int => "@intCast({s})",
            else => @compileError(err_msg),
        },
        else => @compileError(err_msg),
    };
}

fn getMapOpFmt(comptime op: ops.MapOp, comptime dtype: type) []const u8 {
    return comptime switch (op) {
        .Neg => if (@typeInfo(dtype) == .Bool) "!({s})" else "-({s})",
        .Log2 => "@log2({s})",
        .Exp2 => "@exp2({s})",
        .Sqrt => "@sqrt({s})",
        .Recip => "(1.0 / ({s}))",
        else => @compileError("Not implemented"),
    };
}

fn getZipOpFmt(comptime op: ops.ZipOp) []const u8 {
    return comptime switch (op) {
        .Add => "(({s}) + ({s}))",
        .Mul => "(({s}) * ({s}))",
        .Maximum => "@max({s}, {s})",
        .Lt => "(({s}) < ({s}))",
        .Eq => "(({s}) == ({s}))",
        .Xor => "(({s}) ^ ({s}))",
        else => @compileError("Not implemented"),
    };
}

fn getReduceOpFmt(comptime op: ops.ReduceOp) []const u8 {
    return comptime comptimePrint(switch (op) {
        .Sum => getZipOpFmt(.Add),
        .Max => getZipOpFmt(.Maximum),
    }, .{ "acc", "{s}" });
}

pub fn castCodegen(
    _: *const ZigCodegen,
    comptime new_dtype: type,
    x_ptr: anytype,
    comptime x_id: usize,
    out: *@TypeOf(x_ptr.*).Cast(new_dtype),
    comptime out_id: usize,
) []const u8 {
    const Out: type = @TypeOf(out.*);
    const old_dtype: type = @TypeOf(x_ptr.*).dtype;
    return comptimePrint(NO_BROADCAST_LOOP_FMT, .{
        Out.size,
        out_id,
        comptimePrint(getCastFmt(new_dtype, old_dtype), .{comptimePrint(DATA_CONST_INDEX, .{x_id})}),
    });
}

pub fn mapCodegen(
    _: *const ZigCodegen,
    comptime op: ops.MapOp,
    x_ptr: anytype,
    x_id: usize,
    out: *@TypeOf(x_ptr.*),
    out_id: usize,
) []const u8 {
    const Out: type = @TypeOf(out.*);
    return comptimePrint(NO_BROADCAST_LOOP_FMT, .{
        Out.size,
        out_id,
        comptimePrint(getMapOpFmt(op, Out.dtype), .{comptimePrint(DATA_CONST_INDEX, .{x_id})}),
    });
}

pub fn zipCodegen(
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
    if (A == B) {
        return comptimePrint(NO_BROADCAST_LOOP_FMT, .{
            Out.size,
            out_id,
            comptimePrint(getZipOpFmt(op), .{
                comptimePrint(DATA_CONST_INDEX, .{a_id}),
                comptimePrint(DATA_CONST_INDEX, .{b_id}),
            }),
        });
    } else {
        return comptimePrint(BROADCAST_LOOP_FMT, .{
            Out.size,
            codegen.posToIdx(Out),
            out_id,
            comptimePrint(getZipOpFmt(op), .{
                comptimePrint(DATA_VAR_INDEX, .{ a_id, codegen.broadcastIdxToPos(A, Out, "idx") }),
                comptimePrint(DATA_VAR_INDEX, .{ b_id, codegen.broadcastIdxToPos(B, Out, "idx") }),
            }),
        });
    }
}

pub fn reduceCodegen(
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
    if (dim == null) {
        return comptimePrint(REDUCE_ALL_FMT, .{
            x_id,
            X.size,
            comptimePrint(getReduceOpFmt(op), .{comptimePrint(DATA_CONST_INDEX, .{x_id})}),
            out_id,
        });
    } else {
        return comptimePrint(REDUCE_DIM_FMT, .{
            Out.size,
            codegen.posToIdx(@TypeOf(out.*), "i"),
            codegen.idxToPos(@TypeOf(x_ptr.*), "idx"),
            X.shape[dim.?],
            comptimePrint(getReduceOpFmt(op), .{comptimePrint(DATA_VAR_INDEX, .{ x_id, comptimePrint("pos + j * {d}", .{X.strides[dim.?]}) })}),
            out_id,
        });
    }
}
