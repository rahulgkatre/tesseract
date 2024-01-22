// TODO: Write a backend that emits a zig file for all the code
// https://www.youtube.com/watch?v=iWIuaUmMhbI
// The purpose of this is to inline function calls in source code
// which will hopefully make it easier for LLVM to optimize them with loop transforms
const std = @import("std");
const tensor = @import("../tensor.zig");
const ops = @import("../ops.zig");
const Backend = @import("../backend.zig").Backend;
const comptimePrint = std.fmt.comptimePrint;

const ZigCodegen = @This();

fn idxToPosCodegen(comptime tensor_type: type) void {
    _ = tensor_type;
}

fn broadcastIdxToPosCodegen(comptime tensor_type1: type, comptime tensor_type2: type) void {
    _ = tensor_type2;
    _ = tensor_type1;
}

fn posToIdxCodegen(comptime tensor_type: type) void {
    _ = tensor_type;
}

pub fn castCodegen(_: *const ZigCodegen, comptime new_dtype: type, x_ptr: anytype, x_id: usize, out: *@TypeOf(x_ptr.*).Cast(new_dtype), out_id: usize) void {
    const old_dtype: type = @Type(x_ptr).dtype;
    const err_msg = comptimePrint("Cannot cast dtype {} to {}", .{ old_dtype, new_dtype });
    const fmt =
        \\
        \\for (0..{d}) |i| {
        \\    tensor{d}[i] = {s};  
        \\}
        \\
    ;
    return comptimePrint(fmt, .{
        @TypeOf(out.*).size,
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
            comptimePrint("tensor{d}[i]", .{x_id}),
        }),
    });
}

pub fn mapCodegen(_: *const ZigCodegen, comptime op: ops.MapOp, x_ptr: anytype, x_id: usize, out: *@TypeOf(x_ptr.*), out_id: usize) void {
    const fmt =
        \\
        \\for (0..{d}) |i| {
        \\    tensor{d}[i] = {s};  
        \\}
        \\
    ;
    return comptimePrint(fmt, .{
        @TypeOf(out.*).size,
        out_id,
        comptimePrint(comptime switch (op) {
            .Neg => if (@typeInfo(@TypeOf(x_ptr).dtype) == .Bool) "(!{s})" else "(-{s})",
            .Log2 => "@log2({s})",
            .Exp2 => "@exp2({s})",
            .Sqrt => "@sqrt({s})",
            .Recip => "(1.0 / ({s}))",
            else => @compileError("Not implemented"),
        }, .{
            comptimePrint("tensor{d}[i]", .{x_id}),
        }),
    });
}

pub fn zipCodegen(_: *const ZigCodegen, comptime op: ops.ZipOp, a_ptr: anytype, a_id: usize, b_ptr: anytype, b_id: usize, out: *@TypeOf(a_ptr.*).Broadcast(@TypeOf(b_ptr.*)), out_id: usize) void {
    if (@TypeOf(a_ptr.*) == @TypeOf(b_ptr.*)) {
        const fmt =
            \\
            \\for (0..{d}) |i| {
            \\    tensor{d}[i] = {s};
            \\}
            \\
        ;
        return comptimePrint(fmt, .{
            @TypeOf(out.*).size,
            out_id,
            comptimePrint(comptime switch (op) {
                .Add => "({s} + {s})",
                .Mul => "({s} * {s})",
                .Maximum => "@max({s}, {s})",
                .Lt => "({s} < {s})",
                .Eq => "({s} == {s})",
                .Xor => "({s} ^ {s})",
                else => @compileError("Not implemented"),
            }, .{
                comptimePrint("tensor{d}[i]", .{a_id}),
                comptimePrint("tensor{d}[i]", .{b_id}),
            }),
        });
    } else {
        const fmt =
            \\
            \\for (0..{d}) |i| {
            \\    const idx = {s}; 
            \\    tensor{d}[i] = {s};
            \\}
            \\
        ;
        return comptimePrint(fmt, .{
            @TypeOf(out.*).size,
            posToIdxCodegen(@TypeOf(out.*)),
            out_id,
            comptimePrint(comptime switch (op) {
                .Add => "({s} + {s})",
                .Mul => "({s} * {s})",
                .Maximum => "@max({s}, {s})",
                .Lt => "({s} < {s})",
                .Eq => "({s} == {s})",
                .Xor => "({s} ^ {s})",
                else => @compileError("Not implemented"),
            }, .{
                comptimePrint("tensor{d}[{s}]", .{
                    a_id,
                    broadcastIdxToPosCodegen(@TypeOf(a_ptr.*), @TypeOf(out.*)),
                }),
                comptimePrint("tensor{d}[{s}]", .{
                    b_id,
                    broadcastIdxToPosCodegen(@TypeOf(a_ptr.*), @TypeOf(out.*)),
                }),
            }),
        });
    }
}

pub fn reduce(_: *const ZigCodegen, comptime op: ops.ReduceOp, x_ptr: anytype, x_id: usize, comptime dim: ?u8, out: *@TypeOf(x_ptr.*).Reduce(dim), out_id: usize) void {
    if (dim == null) {
        const fmt =
            \\
            \\var acc{d} = tensor{d}[0];
            \\for (1..{d}) |i| {
            \\    acc{d} = {s};
            \\}
            \\tensor{d}[0] = acc{d};
            \\
        ;
        return comptimePrint(fmt, .{
            out_id,
            out_id,
            @TypeOf(out.*).size,
            out_id,
            comptimePrint(comptime switch (op) {
                .Sum => "(acc{d} + {s})",
                .Max => "@max(acc{d}, {s})",
                else => @compileError("Not implemented"),
            }, .{
                out_id,
                comptimePrint("tensor{d}[i]", .{x_id}),
            }),
        });
    } else {
        const fmt =
            \\
            \\for (0..{d}) |i| {
            \\    var acc = tensor{d}[{s}];
            \\    for (1..{d}) |j| {
            \\        acc = {s};
            \\    }
            \\    tensor{d}[i] = acc;
            \\}
            \\
        ;
        return comptimePrint(fmt, .{
            @TypeOf(out.*).size,
            x_id,
            posToIdxCodegen(@TypeOf(out.*)),
            out.shape[dim.?],
            comptimePrint(comptime switch (op) {
                .Sum => "(acc + {s})",
                .Max => "@max(acc, {s})",
                else => @compileError("Not implemented"),
            }, .{
                comptimePrint("tensor{d}[offset + i * {d}]", .{
                    x_id,
                    out.strides[dim.?],
                }),
            }),
        });
    }
}
