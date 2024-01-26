const std = @import("std");
const tensor = @import("../tensor.zig");
const ops = @import("../ops.zig");
const Backend = @import("../backend.zig").Backend;
const comptimePrint = std.fmt.comptimePrint;
const codegen = @import("../codegen.zig");
const ZigCodegen = @This();

const DATA_CONST_INDEX = "t{d}[i]";
const DATA_VAR_INDEX = "t{d}[{s}]";

const SIMPLE_LOOP_FMT =
    \\for (0..{d}) |i| {
    \\    t{d}[i] = {s};  
    \\}
;
const BROADCAST_LOOP_FMT =
    \\for (0..{d}) |i| {
    \\    const idx = {s}; 
    \\    t{d}[i] = {s};
    \\}
;
const REDUCE_ALL_FMT =
    \\{
    \\    var acc = t{d}[0];
    \\    for (1..{d}) |i| {
    \\        acc = {s};
    \\    }
    \\    t{d}[0] = acc;
    \\};
;
const REDUCE_DIM_FMT =
    \\for (0..{d}) |i| {
    \\    const idx = {s}
    \\    const pos = {s};
    \\    var acc = t{d}[pos];
    \\    for (1..{d}) |j| {
    \\        acc = {s};
    \\    }
    \\    t{d}[i] = acc;
    \\}
;
const REDUCE_ZIP_ARGS = .{ "acc", "{s}" };

const FLOAT_INT = "@floatFromInt({s})";
const FLOAT_FLOAT = "   @floatCast({s})";
const INT_FLOAT = "@intFromFloat({s})";
const INT_BOOL = " @intFromBool({s})";
const INT_INT = "     @intCast({s})";

comptime {
    if (!(FLOAT_INT.len == FLOAT_FLOAT.len and FLOAT_FLOAT.len == INT_FLOAT.len and INT_FLOAT.len == INT_BOOL.len and INT_BOOL.len == INT_INT.len)) {
        @compileError("Map operation template strings have inequal lengths");
    }
}

fn getCastFmt(comptime new_dtype: type, comptime old_dtype: type) @TypeOf(FLOAT_INT) {
    const err_msg = comptimePrint("Cannot cast dtype {} to {}", .{ old_dtype, new_dtype });
    comptime switch (@typeInfo(new_dtype)) {
        .Float => switch (@typeInfo(old_dtype)) {
            .Int => FLOAT_INT,
            .Float => FLOAT_FLOAT,
            else => @compileError(err_msg),
        },
        .Int => switch (@typeInfo(old_dtype)) {
            .Float => INT_FLOAT,
            .Bool => INT_BOOL,
            .Int => INT_INT,
            else => @compileError(err_msg),
        },
        else => @compileError(err_msg),
    };
}

const NOT = "  (!({s}))";
const NEG = "  (-({s}))";
const LOG2 = "@log2({s})";
const EXP2 = "@exp2({s})";
const SQRT = "@sqrt({s})";
const RECIP = "(1./({s}))";

comptime {
    if (!(NOT.len == NEG.len and NEG.len == LOG2.len and LOG2.len == EXP2.len and EXP2.len == SQRT.len and SQRT.len == RECIP.len)) {
        @compileError("Map operation template strings have inequal lengths");
    }
}

fn getMapOpFmt(comptime op: ops.MapOp, comptime dtype: type) @TypeOf(NOT) {
    return comptime switch (op) {
        .Neg => if (@typeInfo(dtype) == .Bool) NOT else NEG,
        .Log2 => LOG2,
        .Exp2 => EXP2,
        .Sqrt => SQRT,
        .Recip => RECIP,
        else => @compileError("Not implemented"),
    };
}

const ADD = " (({s})+({s}))";
const MUL = " (({s})*({s}))";
const MAXIMUM = " @max({s},{s})";
const LT = " (({s})<({s}))";
const EQ = "(({s})==({s}))";
const XOR = " (({s})^({s}))";

comptime {
    if (!(ADD.len == MUL.len and MUL.len == MAXIMUM.len and MAXIMUM.len == LT.len and LT.len == EQ.len and EQ.len == XOR.len)) {
        @compileError("Zip operation template strings have inequal lengths");
    }
}

fn getZipOpFmt(comptime op: ops.ZipOp) @TypeOf(ADD) {
    return comptime switch (op) {
        .Add => ADD,
        .Mul => MUL,
        .Maximum => MAXIMUM,
        .Lt => LT,
        .Eq => EQ,
        .Xor => XOR,
        else => @compileError("Not implemented"),
    };
}

fn getReduceOpFmt(comptime op: ops.ReduceOp) @TypeOf(comptimePrint(ADD, REDUCE_ZIP_ARGS)) {
    return comptime comptimePrint(switch (op) {
        .Sum => ADD,
        .Max => MAXIMUM,
    }, REDUCE_ZIP_ARGS);
}

pub fn castCodegen(
    _: *const ZigCodegen,
    comptime new_dtype: type,
    x_ptr: anytype,
    comptime x_id: usize,
    out: *@TypeOf(x_ptr.*).Cast(new_dtype),
    comptime out_id: usize,
) @TypeOf(comptimePrint(SIMPLE_LOOP_FMT, .{
    @TypeOf(out.*).size,
    out_id,
    comptimePrint(getCastFmt(new_dtype, @TypeOf(out.*).dtype), .{comptimePrint(DATA_CONST_INDEX, .{x_id})}),
})) {
    const Out: type = @TypeOf(out.*);
    const old_dtype: type = @Type(x_ptr).dtype;
    return comptimePrint(SIMPLE_LOOP_FMT, .{
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
) @TypeOf(comptimePrint(SIMPLE_LOOP_FMT, .{
    @TypeOf(out.*).size,
    out_id,
    comptimePrint(NOT, .{comptimePrint(DATA_CONST_INDEX, .{x_id})}),
})) {
    const Out: type = @TypeOf(out.*);
    return comptimePrint(SIMPLE_LOOP_FMT, .{
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
) @TypeOf(if (@TypeOf(a_ptr.*) == @TypeOf(b_ptr.*))
{
    comptimePrint(SIMPLE_LOOP_FMT, .{
        @TypeOf(out.*).size,
        out_id,
        comptimePrint(getZipOpFmt(op), .{
            comptimePrint(DATA_CONST_INDEX, .{a_id}),
            comptimePrint(DATA_CONST_INDEX, .{b_id}),
        }),
    });
} else {
    comptimePrint(BROADCAST_LOOP_FMT, .{
        @TypeOf(out.*).size,
        codegen.posToIdx(@TypeOf(out.*)),
        out_id,
        comptimePrint(getZipOpFmt(op), .{
            comptimePrint(DATA_VAR_INDEX, .{ a_id, codegen.broadcastIdxToPos(@TypeOf(a_ptr.*), @TypeOf(out.*), "idx") }),
            comptimePrint(DATA_VAR_INDEX, .{ b_id, codegen.broadcastIdxToPos(@TypeOf(b_ptr.*), @TypeOf(out.*), "idx") }),
        }),
    });
}) {
    const Out: type = @TypeOf(out.*);
    const A: type = @TypeOf(a_ptr.*);
    const B: type = @TypeOf(b_ptr.*);
    return if (A == B) {
        comptimePrint(SIMPLE_LOOP_FMT, .{
            Out.size,
            out_id,
            comptimePrint(getZipOpFmt(op), .{
                comptimePrint(DATA_CONST_INDEX, .{a_id}),
                comptimePrint(DATA_CONST_INDEX, .{b_id}),
            }),
        });
    } else {
        comptimePrint(BROADCAST_LOOP_FMT, .{
            Out.size,
            codegen.posToIdx(Out),
            out_id,
            comptimePrint(getZipOpFmt(op), .{
                comptimePrint(DATA_VAR_INDEX, .{ a_id, codegen.broadcastIdxToPos(A, Out, "idx") }),
                comptimePrint(DATA_VAR_INDEX, .{ b_id, codegen.broadcastIdxToPos(B, Out, "idx") }),
            }),
        });
    };
}

pub fn reduceCodegen(
    _: *const ZigCodegen,
    comptime op: ops.ReduceOp,
    x_ptr: anytype,
    comptime x_id: usize,
    comptime dim: ?u8,
    out: *@TypeOf(x_ptr.*).Reduce(dim),
    comptime out_id: usize,
) @TypeOf(if (dim == null)
{
    comptimePrint(REDUCE_ALL_FMT, .{
        out_id,
        @TypeOf(out.*).size,
        comptimePrint(getReduceOpFmt(op), .{comptimePrint(DATA_CONST_INDEX, .{x_id})}),
        out_id,
    });
} else {
    comptimePrint(REDUCE_DIM_FMT, .{
        @TypeOf(out.*).size,
        codegen.posToIdx(@TypeOf(out.*), "i"),
        codegen.idxToPos(@TypeOf(x_ptr.*), "idx"),
        @TypeOf(x_ptr.*).shape[dim.?],
        comptimePrint(getReduceOpFmt(op), .{comptimePrint(DATA_VAR_INDEX, .{ x_id, comptimePrint("base + j * {d}", .{@TypeOf(x_ptr.*).strides[dim.?]}) })}),
        out_id,
    });
}) {
    const Out: type = @TypeOf(out.*);
    const X: type = @TypeOf(x_ptr.*);
    if (dim == null) {
        comptimePrint(REDUCE_ALL_FMT, .{
            out_id,
            Out.size,
            comptimePrint(getReduceOpFmt(op), .{comptimePrint(DATA_CONST_INDEX, .{x_id})}),
            out_id,
        });
    } else {
        comptimePrint(REDUCE_DIM_FMT, .{
            Out.size,
            codegen.posToIdx(@TypeOf(out.*), "i"),
            codegen.idxToPos(@TypeOf(x_ptr.*), "idx"),
            X.shape[dim.?],
            comptimePrint(getReduceOpFmt(op), .{comptimePrint(DATA_VAR_INDEX, .{ x_id, comptimePrint("pos + j * {d}", .{X.strides[dim.?]}) })}),
            out_id,
        });
    }
}
