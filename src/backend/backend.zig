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
    pub fn map(self: *const Backend, op: ops.MapOp, x: anytype) @TypeOf(x) {
        var out = @TypeOf(x).result(self);
        out.evalFn = struct {
            fn eval(out_ptr: *@TypeOf(out)) @TypeOf(out) {
                const eval_x = x.eval();
                if (!@inComptime()) {
                    // std.debug.print("\n{s}@{d} = {s} {s}@{d}", .{ out_ptr.str, @intFromPtr(out_ptr), @tagName(op), x.str, @intFromPtr(&x) });
                    out_ptr.initStorage();
                    switch (self.*) {
                        inline else => |*backend| backend.map(op, eval_x, out_ptr),
                    }
                } else {
                    @compileLog(comptimePrint("{s} = {s} {s}", .{ out_ptr.str, @tagName(op), x.str }));
                }
                return out_ptr.*;
            }
        }.eval;
        return out;
    }
    pub fn zip(self: *const Backend, op: ops.ZipOp, a: anytype, b: anytype) tensor.BroadcastedTensor(@TypeOf(a), @TypeOf(b)) {
        var out = tensor.BroadcastedTensor(@TypeOf(a), @TypeOf(b)).result(self);
        out.evalFn = struct {
            fn eval(out_ptr: *@TypeOf(out)) @TypeOf(out) {
                const eval_a = a.eval();
                const eval_b = b.eval();
                if (!@inComptime()) {
                    // std.debug.print("\n{s}@{d} = {s} {s}@{d} {s}@{d}", .{ out_ptr.str, @intFromPtr(out_ptr), @tagName(op), a.str, @intFromPtr(&a), b.str, @intFromPtr(&b) });
                    out_ptr.initStorage();
                    switch (self.*) {
                        inline else => |*backend| backend.zip(op, eval_a, eval_b, out_ptr),
                    }
                } else {
                    @compileLog(comptimePrint("{s} = {s} {s} {s}", .{ out_ptr.str, @tagName(op), a.str, b.str }));
                }
                return out_ptr.*;
            }
        }.eval;
        return out;
    }
    pub fn reduce(self: *const Backend, op: ops.ReduceOp, x: anytype, dim: ?u8) tensor.ReducedTensor(@TypeOf(x), dim) {
        var out = tensor.ReducedTensor(@TypeOf(x), dim).result(self);
        out.evalFn = struct {
            fn eval(out_ptr: *@TypeOf(out)) @TypeOf(out) {
                const eval_x = x.eval();
                if (!@inComptime()) {
                    // std.debug.print("\n{s}@{d} = {s} {s}@{d} {d}", .{ out_ptr.str, @intFromPtr(out_ptr), @tagName(op), x.str, @intFromPtr(&x), dim orelse -1 });
                    out_ptr.initStorage();
                    switch (self.*) {
                        inline else => |*backend| backend.reduce(op, eval_x, dim, out_ptr),
                    }
                } else {
                    @compileLog(comptimePrint("{s} = {s} {s} {?}", .{ out_ptr.str, @tagName(op), x.str, dim orelse -1 }));
                }
                return out_ptr.*;
            }
        }.eval;
        return out;
    }
};
