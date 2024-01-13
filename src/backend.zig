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
    pub fn asType(self: *const Backend, comptime new_dtype: type, x: anytype) tensor.CastedTensor(@TypeOf(x), new_dtype) {
        const OutType = tensor.CastedTensor(@TypeOf(x), new_dtype);
        const Impl = struct {
            fn eval(out_ptr: *OutType) OutType {
                const eval_x = x.eval();
                out_ptr.initStorage();
                switch (self.*) {
                    inline else => |*backend| backend.asType(new_dtype, eval_x, out_ptr),
                }
                return out_ptr.*;
            }
            fn graph(ptr: *const OutType) void {
                if (@inComptime()) {
                    x.graph();
                    @compileLog(comptimePrint("{s} = AsType({any}) {s}", .{ ptr.str, new_dtype, x.str }));
                } else {
                    std.debug.print("{s}@{d} = AsType({any}) {s}@{d}\n", .{ ptr.str, @intFromPtr(ptr), new_dtype, x.str, @intFromPtr(&x) });
                }
            }
        };
        return OutType.result(
            self,
            null,
            Impl.eval,
            Impl.graph,
        );
    }

    pub fn map(self: *const Backend, op: ops.MapOp, x: anytype) @TypeOf(x) {
        const OutType: type = @TypeOf(x);
        const Impl = struct {
            fn eval(eval_out: *OutType) OutType {
                const eval_x = x.eval();
                eval_out.initStorage();
                switch (self.*) {
                    inline else => |*backend| backend.map(op, eval_x, eval_out),
                }
                return eval_out.*;
            }
            fn graph(out_ptr: *const OutType) void {
                if (@inComptime()) {
                    x.graph();
                    @compileLog(comptimePrint("{s} := {s} {s}", .{ out_ptr.str, @tagName(op), x.str }));
                } else {
                    std.debug.print("{s}@{d} := {s} {s}@{d}\n", .{ out_ptr.str, @intFromPtr(out_ptr), @tagName(op), x.str, @intFromPtr(&x) });
                }
            }
        };
        return OutType.result(
            self,
            null,
            Impl.eval,
            Impl.graph,
        );
    }
    pub fn zip(self: *const Backend, op: ops.ZipOp, a: anytype, b: anytype) tensor.BroadcastedTensor(@TypeOf(a), @TypeOf(b)) {
        const OutType = tensor.BroadcastedTensor(@TypeOf(a), @TypeOf(b));
        const Impl = struct {
            fn eval(eval_out: *OutType) OutType {
                const eval_a = a.eval();
                const eval_b = b.eval();
                eval_out.initStorage();
                switch (self.*) {
                    inline else => |*backend| backend.zip(op, eval_a, eval_b, eval_out),
                }
                return eval_out.*;
            }
            fn graph(ptr: *const OutType) void {
                a.graph();
                b.graph();
                if (@inComptime()) {
                    @compileLog(comptimePrint("{s} := {s} {s} {s}", .{ ptr.str, a.str, @tagName(op), b.str }));
                } else {
                    std.debug.print("{s}@{d} := {s}@{d} {s} {s}@{d}\n", .{ ptr.str, @intFromPtr(ptr), a.str, @intFromPtr(&a), @tagName(op), b.str, @intFromPtr(&b) });
                }
            }
        };
        return OutType.result(
            self,
            null,
            Impl.eval,
            Impl.graph,
        );
    }
    pub fn reduce(self: *const Backend, op: ops.ReduceOp, x: anytype, dim: ?u8) tensor.ReducedTensor(@TypeOf(x), dim) {
        const OutType = tensor.ReducedTensor(@TypeOf(x), dim);
        const Impl = struct {
            fn eval(eval_out: *OutType) OutType {
                const eval_x = x.eval();
                eval_out.initStorage();
                switch (self.*) {
                    inline else => |*backend| backend.reduce(op, eval_x, dim, eval_out),
                }
                return eval_out.*;
            }

            fn graph(ptr: *const OutType) void {
                x.graph();
                if (@inComptime()) {
                    @compileLog(comptimePrint("{s} := {s} {s} {d}", .{ ptr.str, @tagName(op), x.str, dim orelse -1 }));
                } else {
                    std.debug.print("{s}@{d} := {s} {s}@{d} {d}\n", .{ ptr.str, @intFromPtr(ptr), @tagName(op), x.str, @intFromPtr(&x), dim orelse -1 });
                }
            }
        };
        return OutType.result(
            self,
            null,
            Impl.eval,
            Impl.graph,
        );
    }
};
