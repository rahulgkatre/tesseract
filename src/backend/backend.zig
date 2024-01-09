const std = @import("std");
const Allocator = std.mem.Allocator;
const ops = @import("../ops.zig");
const comptimePrint = std.fmt.comptimePrint;
const tensor = @import("../tensor.zig");
const ZigBackend = @import("ZigBackend.zig");

pub const BackendTypes = enum {
    Zig,
};

pub const Backend = union(BackendTypes) {
    Zig: ZigBackend,

    pub fn Storage(comptime dtype: type) type {
        return union(BackendTypes) {
            const Self = @This();
            Zig: ZigBackend.ZigStorage(dtype),

            pub fn fill(self: *Self, value: dtype) void {
                switch (self.*) {
                    inline else => |*b| b.fill(value),
                }
            }
        };
    }

    pub fn init(self: *Backend, args: anytype) void {
        return switch (self.*) {
            inline else => |*b| b.init(args),
        };
    }
    pub fn alloc(self: *const Backend, comptime dtype: type, size: usize) !*Storage(dtype) {
        return switch (self.*) {
            inline else => |*b| try b.alloc(dtype, size),
        };
    }
    pub fn map(self: *const Backend, op: ops.MapOp, x: anytype) @TypeOf(x) {
        var out = @TypeOf(x).result(self);
        out.eval_fn = struct {
            var done = false;
            fn eval(ptr: *@TypeOf(out)) void {
                x.eval();
                if (!@inComptime()) {
                    if (done) {
                        return;
                    }
                    std.debug.print("\n{s}@{d} = {any} {s}@{d}", .{ ptr.str, @intFromPtr(ptr), op, x.str, @intFromPtr(&x) });
                    done = true;
                } else {
                    @compileLog(comptimePrint("{s} = {any} {s}", .{ ptr.str, op, x.str }));
                }
                switch (self.*) {
                    inline else => |*eval_backend| eval_backend.mapEval(op, x, ptr),
                }
            }
        }.eval;
        return out;
    }
    pub fn zip(self: *const Backend, op: ops.ZipOp, a: anytype, b: anytype) tensor.BroadcastedTensor(@TypeOf(a), @TypeOf(b)) {
        var out = tensor.BroadcastedTensor(@TypeOf(a), @TypeOf(b)).result(self);
        out.eval_fn = struct {
            var done = false;
            fn eval(ptr: *@TypeOf(out)) void {
                a.eval();
                b.eval();
                if (!@inComptime()) {
                    if (done) {
                        return;
                    }
                    std.debug.print("\n{s}@{d} = {any} {s}@{d} {s}@{d}", .{ ptr.str, @intFromPtr(ptr), op, a.str, @intFromPtr(&a), b.str, @intFromPtr(&b) });
                    done = true;
                } else {
                    @compileLog(comptimePrint("{s} = {any} {s} {s}", .{ ptr.str, op, a.str, b.str }));
                }
                switch (self.*) {
                    inline else => |*eval_backend| eval_backend.zipEval(op, a, b, ptr),
                }
            }
        }.eval;
        return out;
    }
    pub fn reduce(self: *const Backend, op: ops.ReduceOp, x: anytype, dim: ?u8) tensor.ReducedTensor(@TypeOf(x), dim) {
        var out = tensor.ReducedTensor(@TypeOf(x), dim).result(self);
        out.eval_fn = struct {
            var done = false;
            fn eval(ptr: *@TypeOf(out)) void {
                x.eval();
                if (!@inComptime()) {
                    if (done) {
                        return;
                    }
                    std.debug.print("\n{s}@{d} = {any} {s}@{d} {?}", .{ ptr.str, @intFromPtr(ptr), op, x.str, @intFromPtr(&x), dim });
                    done = true;
                } else {
                    @compileLog(comptimePrint("{s} = {any} {s} {?}", .{ ptr.str, op, x.str, dim }));
                }
                // TODO: Compute the start value for the accumulator based on the op, and the zip op used to accumulate
                // by switching on the reduce op
                switch (self.*) {
                    inline else => |*eval_backend| eval_backend.reduceEval(op, x, dim, ptr),
                }
            }
        }.eval;
        return out;
    }
};
