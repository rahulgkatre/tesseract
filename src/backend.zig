const std = @import("std");
const Allocator = std.mem.Allocator;
const ops = @import("ops.zig");
const comptimePrint = std.fmt.comptimePrint;
const tensor = @import("tensor.zig");
const ZigBackend = @import("backend/ZigBackend.zig");

pub const BackendTypes = enum {
    Zig,
};

pub const Backend = union(BackendTypes) {
    Zig: ZigBackend,

    pub fn Storage(comptime dtype: type) type {
        return union(BackendTypes) {
            const Self = @This();
            Zig: ZigBackend.Storage(dtype),
            pub fn fill(self: *Self, value: dtype) void {
                switch (self.*) {
                    inline else => |*b| b.fill(value),
                }
            }
            pub fn init(self: *Self) void {
                switch (self.*) {
                    inline else => |*b| b.init(),
                }
            }
        };
    }

    pub fn init(self: *const Backend, args: anytype) void {
        return switch (self.*) {
            inline else => |*b| b.init(args),
        };
    }
    pub fn storage(self: *const Backend, comptime dtype: type, size: usize) Storage(dtype) {
        return switch (self.*) {
            inline else => |*b| b.storage(dtype, size),
        };
    }
    pub fn deinit(self: *const Backend) void {
        return switch (self.*) {
            inline else => |*b| b.deinit(),
        };
    }
    pub fn asType(self: *const Backend, x: anytype, comptime dtype: type) tensor.CastedTensor(@TypeOf(x), dtype) {
        var out = tensor.CastedTensor(@TypeOf(x), dtype);
        out.evalFn = struct {
            fn eval(out_ptr: *@TypeOf(out)) @TypeOf(out) {
                const eval_x = x.eval();
                out_ptr.initStorage();
                switch (self.*) {
                    inline else => |*backend| backend.asType(eval_x, out_ptr, dtype),
                }
                return out_ptr.*;
            }
        }.eval;
        out.graphFn = struct {
            fn graph(ptr: *@TypeOf(out)) void {
                if (@inComptime()) {
                    x.graph();
                    @compileLog(comptimePrint("{s} = AsType({any}) {s}", .{ ptr.str, dtype, x.str }));
                } else {
                    std.debug.print("{s}@{d} = AsType({any}) {s}@{d}\n", .{ ptr.str, @intFromPtr(ptr), dtype, x.str, @intFromPtr(&x) });
                }
            }
        }.graph;
        return out;
    }

    pub fn map(self: *const Backend, op: ops.MapOp, x: anytype) @TypeOf(x) {
        var out = @TypeOf(x).result(self, null);
        out.evalFn = struct {
            fn eval(eval_out: *@TypeOf(out)) @TypeOf(out) {
                const eval_x = x.eval();
                eval_out.initStorage();
                switch (self.*) {
                    inline else => |*backend| backend.map(op, eval_x, eval_out),
                }
                return eval_out.*;
            }
        }.eval;
        out.graphFn = struct {
            fn graph(out_ptr: *const @TypeOf(out)) void {
                if (@inComptime()) {
                    x.graph();
                    @compileLog(comptimePrint("{s} := {s} {s}", .{ out_ptr.str, @tagName(op), x.str }));
                } else {
                    std.debug.print("{s}@{d} := {s} {s}@{d}\n", .{ out_ptr.str, @intFromPtr(out_ptr), @tagName(op), x.str, @intFromPtr(&x) });
                }
            }
        }.graph;
        return out;
    }
    pub fn zip(self: *const Backend, op: ops.ZipOp, a: anytype, b: anytype) tensor.BroadcastedTensor(@TypeOf(a), @TypeOf(b)) {
        var out = tensor.BroadcastedTensor(@TypeOf(a), @TypeOf(b)).result(self, null);
        out.evalFn = struct {
            fn eval(eval_out: *@TypeOf(out)) @TypeOf(out) {
                const eval_a = a.eval();
                const eval_b = b.eval();
                eval_out.initStorage();
                switch (self.*) {
                    inline else => |*backend| backend.zip(op, eval_a, eval_b, eval_out),
                }
                return eval_out.*;
            }
        }.eval;
        out.graphFn = struct {
            fn graph(ptr: *const @TypeOf(out)) void {
                a.graph();
                b.graph();
                if (@inComptime()) {
                    @compileLog(comptimePrint("{s} := {s} {s} {s}", .{ ptr.str, a.str, @tagName(op), b.str }));
                } else {
                    std.debug.print("{s}@{d} := {s}@{d} {s} {s}@{d}\n", .{ ptr.str, @intFromPtr(ptr), a.str, @intFromPtr(&a), @tagName(op), b.str, @intFromPtr(&b) });
                }
            }
        }.graph;
        return out;
    }
    pub fn reduce(self: *const Backend, op: ops.ReduceOp, x: anytype, dim: ?u8) tensor.ReducedTensor(@TypeOf(x), dim) {
        var out = tensor.ReducedTensor(@TypeOf(x), dim).result(self, null);
        out.evalFn = struct {
            fn eval(eval_out: *@TypeOf(out)) @TypeOf(out) {
                const eval_x = x.eval();
                eval_out.initStorage();
                switch (self.*) {
                    inline else => |*backend| backend.reduce(op, eval_x, dim, eval_out),
                }
                return eval_out.*;
            }
        }.eval;
        out.graphFn = struct {
            fn graph(ptr: *const @TypeOf(out)) void {
                x.graph();
                if (@inComptime()) {
                    @compileLog(comptimePrint("{s} := {s} {s} {d}", .{ ptr.str, @tagName(op), x.str, dim orelse -1 }));
                } else {
                    std.debug.print("{s}@{d} := {s} {s}@{d} {d}\n", .{ ptr.str, @intFromPtr(ptr), @tagName(op), x.str, @intFromPtr(&x), dim orelse -1 });
                }
            }
        }.graph;
        return out;
    }
};
