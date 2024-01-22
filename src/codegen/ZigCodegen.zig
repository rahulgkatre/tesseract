// TODO: Write a backend that emits a zig file for all the code
// https://www.youtube.com/watch?v=iWIuaUmMhbI
// The purpose of this is to inline function calls in source code
// which will hopefully make it easier for LLVM to optimize them with loop transforms
const std = @import("std");
const tensor = @import("../tensor.zig");
const ops = @import("../ops.zig");
const Backend = @import("../backend.zig").Backend;
const comptimePrint = std.fmt.comptimePrint;
const codegen = @import("../codegen.zig");

const ZigCodegen = @This();

pub fn cast(_: *const ZigCodegen, comptime new_dtype: type, x_ptr: anytype, comptime x_id: usize, out: *@TypeOf(x_ptr.*).Cast(new_dtype), comptime out_id: usize) void {
    const Out: type = @TypeOf(out.*);
    const old_dtype: type = @Type(x_ptr).dtype;
    const err_msg = comptimePrint("Cannot cast dtype {} to {}", .{ old_dtype, new_dtype });
    const fmt =
        \\
        \\for (0..{d}) |i| {
        \\    t{d}[i] = {s};  
        \\}
        \\
    ;
    return comptimePrint(fmt, .{
        Out.size,
        out_id,
        comptimePrint(comptime switch (@typeInfo(new_dtype)) {
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
        }, .{
            comptimePrint("t{d}[i]", .{x_id}),
        }),
    });
}

pub fn map(
    _: *const ZigCodegen,
    comptime op: ops.MapOp,
    x_ptr: anytype,
    x_id: usize,
    out: *@TypeOf(x_ptr.*),
    out_id: usize,
) t: {
    const Out: type = @TypeOf(out.*);
    const fmt =
        \\
        \\for (0..{d}) |i| {
        \\    t{d}[i] = {s};  
        \\}
        \\
    ;
    break :t @TypeOf(comptimePrint(fmt, .{
        Out.size,
        out_id,
        comptimePrint(comptime switch (op) {
            .Neg => if (@typeInfo(@TypeOf(x_ptr).dtype) == .Bool) "(!{s})" else "(-{s})",
            .Log2 => "@log2({s})",
            .Exp2 => "@exp2({s})",
            .Sqrt => "@sqrt({s})",
            .Recip => "(1.0 / ({s}))",
            else => @compileError("Not implemented"),
        }, .{
            comptimePrint("t{d}[i]", .{x_id}),
        }),
    }));
} {
    const Out: type = @TypeOf(out.*);
    const fmt =
        \\
        \\for (0..{d}) |i| {
        \\    t{d}[i] = {s};  
        \\}
        \\
    ;
    return comptimePrint(fmt, .{
        Out.size,
        out_id,
        comptimePrint(comptime switch (op) {
            .Neg => if (@typeInfo(@TypeOf(x_ptr).dtype) == .Bool) "(!{s})" else "(-{s})",
            .Log2 => "@log2({s})",
            .Exp2 => "@exp2({s})",
            .Sqrt => "@sqrt({s})",
            .Recip => "(1.0 / ({s}))",
            else => @compileError("Not implemented"),
        }, .{
            comptimePrint("t{d}[i]", .{x_id}),
        }),
    });
}

const zipUnicastBaseFmt =
    \\
    \\for (0..{s}) |i| {
    \\    t{s}[i] = {s};
    \\}
    \\
;
fn zipUnicastFmt(comptime op: ops.ZipOp) @TypeOf(comptimePrint(zipUnicastBaseFmt, .{
    "{d}",
    "{d}",
    comptimePrint(comptime switch (op) {
        .Add => "({s} + {s})",
        .Mul => "({s} * {s})",
        .Maximum => "@max({s}, {s})",
        .Lt => "({s} < {s})",
        .Eq => "({s} == {s})",
        .Xor => "({s} ^ {s})",
        else => @compileError("Not implemented"),
    }, .{
        "t{d}[i]",
        "t{d}[i]",
    }),
})) {
    return comptimePrint(zipUnicastBaseFmt, .{
        "{d}",
        "{d}",
        comptimePrint(comptime switch (op) {
            .Add => "({s} + {s})",
            .Mul => "({s} * {s})",
            .Maximum => "@max({s}, {s})",
            .Lt => "({s} < {s})",
            .Eq => "({s} == {s})",
            .Xor => "({s} ^ {s})",
            else => @compileError("Not implemented"),
        }, .{
            "t{d}[i]",
            "t{d}[i]",
        }),
    });
}

const zipBroadcastBaseFmt =
    \\
    \\for (0..{s}) |i| {
    \\    const idx = {s}; 
    \\    t{s}[i] = {s};
    \\}
    \\
;
fn zipBroadcastFmt(comptime op: ops.ZipOp) @TypeOf(comptimePrint(zipUnicastBaseFmt, .{
    "{d}", "{d}", "{d}", comptimePrint(comptime switch (op) {
        .Add => "({s} + {s})",
        .Mul => "({s} * {s})",
        .Maximum => "@max({s}, {s})",
        .Lt => "({s} < {s})",
        .Eq => "({s} == {s})",
        .Xor => "({s} ^ {s})",
        else => @compileError("Not implemented"),
    }, .{ "t{d}[{s}]", "t{d}[{s}]" }),
})) {
    return comptimePrint(zipUnicastBaseFmt, .{
        "{d}",
        "{d}",
        "{d}",
        comptimePrint(comptime switch (op) {
            .Add => "({s} + {s})",
            .Mul => "({s} * {s})",
            .Maximum => "@max({s}, {s})",
            .Lt => "({s} < {s})",
            .Eq => "({s} == {s})",
            .Xor => "({s} ^ {s})",
            else => @compileError("Not implemented"),
        }, .{ "t{d}[{s}]", "t{d}[{s}]" }),
    });
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
) t: {
    const Out: type = @TypeOf(out.*);
    const A: type = @TypeOf(a_ptr.*);
    const B: type = @TypeOf(b_ptr.*);
    if (A == B) {
        break :t @TypeOf(comptimePrint(zipUnicastFmt(op), .{ Out.size, out_id, a_id, b_id }));
    } else {
        break :t @TypeOf(comptimePrint(zipBroadcastFmt(op), .{
            Out.size,
            codegen.posToIdx(Out),
            out_id,
            a_id,
            codegen.broadcastIdxToPos(A, Out, "idx"),
            b_id,
            codegen.broadcastIdxToPos(B, Out, "idx"),
        }));
    }
} {
    const Out: type = @TypeOf(out.*);
    const A: type = @TypeOf(a_ptr.*);
    const B: type = @TypeOf(b_ptr.*);
    if (A == B) {
        return comptimePrint(zipUnicastFmt(op), .{ Out.size, out_id, a_id, b_id });
    } else {
        return comptimePrint(zipBroadcastFmt(op), .{
            Out.size,
            codegen.posToIdx(Out),
            out_id,
            a_id,
            codegen.broadcastIdxToPos(A, Out, "idx"),
            b_id,
            codegen.broadcastIdxToPos(B, Out, "idx"),
        });
    }
}

const reduceAllBaseFmt =
    \\
    \\var acc{s} = t{s}[0];
    \\for (1..{s}) |i| {
    \\    acc{s} = {s};
    \\}
    \\t{s}[0] = acc{s};
    \\
;
fn reduceAllFmt(comptime op: ops.ReduceOp) @TypeOf(comptimePrint(reduceAllBaseFmt, .{
    "{d}", "{d}", "{d}", "{d}",
    comptimePrint(comptime switch (op) {
        .Sum => "(acc{s} + {s})",
        .Max => "@max(acc{s}, {s})",
        else => @compileError("Not implemented"),
    }, .{ "{d}", "t{d}[i]" }),
    "{d}",
    "{d}",
})) {
    return comptimePrint(reduceAllBaseFmt, .{
        "{d}", "{d}", "{d}", "{d}",
        comptimePrint(comptime switch (op) {
            .Sum => "(acc{s} + {s})",
            .Max => "@max(acc{s}, {s})",
            else => @compileError("Not implemented"),
        }, .{ "{d}", "t{d}[i]" }),
        "{d}",
        "{d}",
    });
}

const reduceDimBaseFmt =
    \\
    \\for (0..{s}) |i| {
    \\    var acc = t{s}[{s}];
    \\    for (1..{d}) |j| {
    \\        acc = {s};
    \\    }
    \\    t{d}[i] = acc;
    \\}
    \\
;
fn reduceDimFmt(comptime op: ops.ReduceOp) @TypeOf(comptimePrint(reduceDimBaseFmt, .{
    "{d}", "{d}", "{s}", "{d}",
    comptimePrint(comptime switch (op) {
        .Sum => "(acc{s} + {s})",
        .Max => "@max(acc{s}, {s})",
        else => @compileError("Not implemented"),
    }, .{ "{d}", "t{d}[i]" }),
    "{d}",
})) {
    return comptimePrint(reduceAllBaseFmt, .{
        "{d}", "{d}", "{s}", "{d}",
        comptimePrint(comptime switch (op) {
            .Sum => "(acc{s} + {s})",
            .Max => "@max(acc{s}, {s})",
            else => @compileError("Not implemented"),
        }, .{ "{d}", "t{d}[i]" }),
        "{d}",
    });
}

pub fn reduce(
    _: *const ZigCodegen,
    comptime op: ops.ReduceOp,
    x_ptr: anytype,
    comptime x_id: usize,
    comptime dim: ?u8,
    out: *@TypeOf(x_ptr.*).Reduce(dim),
    comptime out_id: usize,
) t: {
    const Out: type = @TypeOf(out.*);
    const X: type = @TypeOf(x_ptr.*);
    if (dim == null) {
        break :t @TypeOf(comptimePrint(reduceAllFmt(op), .{
            out_id,
            out_id,
            Out.size,
            out_id,
            comptimePrint(comptime switch (op) {
                .Sum => "(acc{d} + {s})",
                .Max => "@max(acc{d}, {s})",
                else => @compileError("Not implemented"),
            }, .{
                out_id,
                comptimePrint("t{d}[i]", .{x_id}),
            }),
            out_id,
            out_id,
        }));
    } else {
        break :t @TypeOf(comptimePrint(reduceDimFmt(op), .{
            Out.size,
            x_id,
            codegen.posToIdx(Out, "i"),
            X.shape[dim.?],
            comptimePrint(comptime switch (op) {
                .Sum => "(acc + {s})",
                .Max => "@max(acc, {s})",
                else => @compileError("Not implemented"),
            }, .{
                comptimePrint("t{d}[offset + j * {d}]", .{
                    x_id,
                    X.strides[dim.?],
                }),
            }),
            out_id,
        }));
    }
} {
    const Out: type = @TypeOf(out.*);
    const X: type = @TypeOf(x_ptr.*);
    if (dim == null) {
        return comptimePrint(reduceAllFmt(op), .{
            out_id,
            out_id,
            Out.size,
            out_id,
            comptimePrint(comptime switch (op) {
                .Sum => "(acc{d} + {s})",
                .Max => "@max(acc{d}, {s})",
                else => @compileError("Not implemented"),
            }, .{
                out_id,
                comptimePrint("t{d}[i]", .{x_id}),
            }),
            out_id,
            out_id,
        });
    } else {
        return comptimePrint(reduceDimFmt(op), .{
            Out.size,
            x_id,
            codegen.posToIdx(Out, "i"),
            X.shape[dim.?],
            comptimePrint(comptime switch (op) {
                .Sum => "(acc + {s})",
                .Max => "@max(acc, {s})",
                else => @compileError("Not implemented"),
            }, .{
                comptimePrint("t{d}[offset + j * {d}]", .{
                    x_id,
                    X.strides[dim.?],
                }),
            }),
            out_id,
        });
    }
}
