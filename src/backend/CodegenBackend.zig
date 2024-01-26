// TODO: Write a backend that emits a zig file for all the code
// https://www.youtube.com/watch?v=iWIuaUmMhbI
// The purpose of this is to inline function calls in source code
// which will hopefully make it easier for LLVM to optimize them with loop transforms

const std = @import("std");
const ops = @import("../ops.zig");
const CodegenBackend = @This();
const codegen = @import("../codegen.zig");

const CodegenLanguage = struct {
    var generator: ?codegen.CodegenTypes = null;
    fn init(code_generator: codegen.CodegenTypes) void {
        generator = code_generator;
    }
    fn gen() codegen.CodegenTypes {
        return generator.?;
    }
};

pub inline fn map(_: *const CodegenBackend, comptime op: ops.MapOp, x_ptr: anytype, comptime x_id: usize, out_ptr: *@TypeOf(x_ptr.*), comptime out_id: usize) void {
    switch (CodegenLanguage.gen()) {
        inline else => |cg| std.debug.print("{s}", .{cg.mapCodegen(op, x_ptr, x_id, out_ptr, out_id)}),
    }
}

pub inline fn cast(_: *const CodegenBackend, comptime new_dtype: type, x_ptr: anytype, x_id: usize, out_ptr: *@TypeOf(x_ptr.*), out_id: usize) void {
    switch (CodegenLanguage.gen()) {
        inline else => |cg| std.debug.print("{s}", .{cg.castCodegen(new_dtype, x_ptr, x_id, out_ptr, out_id)}),
    }
}

pub inline fn zip(_: *const CodegenBackend, comptime op: ops.ZipOp, a_ptr: anytype, comptime a_id: usize, b_ptr: anytype, comptime b_id: usize, out_ptr: *@TypeOf(a_ptr.*).Broadcast(@TypeOf(b_ptr.*)), comptime out_id: usize) void {
    switch (CodegenLanguage.gen()) {
        inline else => |cg| std.debug.print("{s}", .{cg.zipCodegen(op, a_ptr, a_id, b_ptr, b_id, out_ptr, out_id)}),
    }
}

pub inline fn reduce(_: *const CodegenBackend, comptime op: ops.MapOp, x_ptr: anytype, comptime x_id: usize, comptime dim: ?u8, out_ptr: *@TypeOf(x_ptr.*), comptime out_id: usize) void {
    switch (CodegenLanguage.gen()) {
        inline else => |cg| std.debug.print("{s}", .{cg.reduceCodegen(op, x_ptr, x_id, dim, out_ptr, out_id)}),
    }
}
