// TODO: Write a backend that emits a zig file for all the code
// https://www.youtube.com/watch?v=iWIuaUmMhbI
// The purpose of this is to inline function calls in source code
// which will hopefully make it easier for LLVM to optimize them
const std = @import("std");
const tensor = @import("../tensor.zig");
const ops = @import("../ops.zig");
const Backend = @import("../backend.zig").Backend;

const ZigCGBackend = @This();

fn idxToPosFmt(comptime tensor_type: type) void {
    _ = tensor_type; // autofix
}

pub fn asType(_: *const ZigCGBackend, comptime new_dtype: type, x: anytype, out: *@TypeOf(x).AsType(new_dtype)) void {
    const old_dtype: type = @Type(x).dtype;
    const err_msg = std.fmt.comptimePrint("Cannot cast dtype {} to {}", .{ old_dtype, new_dtype });
    const code = std.fmt.comptimePrint(
        \\
        \\for (0..{d}) |i| {
        \\    dst[i] = {s};  
        \\}
        \\
    , .{
        @TypeOf(out.*).size,
        std.fmt.comptimePrint(comptime switch (@typeInfo(new_dtype)) {
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
            "src[i]",
        }),
    });
    _ = code; // autofix
}

pub fn map(_: *const ZigCGBackend, comptime map_op: ops.MapOp, x: anytype, out: *@TypeOf(x)) void {
    const code = std.fmt.comptimePrint(
        \\
        \\for (0..{d}) |i| {
        \\    dst[i] = {s};  
        \\}
        \\
    , .{
        @TypeOf(out.*).size,
        std.fmt.comptimePrint(comptime switch (map_op) {
            .Neg => if (@typeInfo(@TypeOf(x).dtype) == .Bool) "(!{s})" else "(-{s})",
            .Log2 => "@log2({s})",
            .Exp2 => "@exp2({s})",
            .Sqrt => "@sqrt({s})",
            .Recip => "(1.0 / ({s}))",
            else => @compileError("Not implemented"),
        }, .{
            "src[i]",
        }),
    });
    _ = code; // autofix
}

pub fn zip(_: *const ZigCGBackend, comptime zip_op: ops.ZipOp, a: anytype, b: anytype, out: *@TypeOf(a).Broadcast(@TypeOf(b))) void {
    const code = std.fmt.comptimePrint(
        \\
        \\for (0..{d}) |i| {
        \\    const dst_idx = {s}; 
        \\    dst[i] = {s};
        \\}
        \\
    , .{
        @TypeOf(out.*).size,
        "TBA",
        std.fmt.comptimePrint(comptime switch (zip_op) {
            .Add => "({s} + {s})",
            .Mul => "({s} * {s})",
            .Maximum => "@max({s}, {s})",
            .Lt => "({s} < {s})",
            .Eq => "({s} == {s})",
            .Xor => "({s} ^ {s})",
            else => @compileError("Not implemented"),
        }, .{
            "src1[{s}]",
            "src2[{s}]",
        }),
    });
    _ = code; // autofix
}
