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
            pub fn load(self: *Self, data: []const dtype) void {
                switch (self.*) {
                    inline else => |*b| b.load(data),
                }
            }
        };
    }

    pub fn init(self: *const Backend, args: anytype) void {
        return switch (self.*) {
            inline else => |*b| b.init(args),
        };
    }
    pub fn storage(self: *const Backend, comptime dtype: type, comptime size: usize) *Storage(dtype) {
        return switch (self.*) {
            inline else => |*b| b.storage(dtype, size),
        };
    }
    pub fn deinit(self: *const Backend) void {
        return switch (self.*) {
            inline else => |*b| b.deinit(),
        };
    }
    pub inline fn asType(self: *const Backend, comptime new_dtype: type, x: anytype) @TypeOf(x).AsType(new_dtype) {
        const Output = @TypeOf(x).AsType(new_dtype);
        const impl = struct {
            fn eval(out: *Output) Output {
                out.initStorage();
                const x_eval = @call(.always_inline, @TypeOf(x).eval, .{&x});
                switch (self.*) {
                    inline else => |*backend| backend.asType(new_dtype, x_eval, out),
                }
                return out.*;
            }
            fn graph(ptr: *const Output) void {
                x.graph();
                if (@inComptime()) {
                    @compileLog(comptimePrint("{s} = AsType({any}) {s}", .{ ptr.str, new_dtype, x.str }));
                } else {
                    std.debug.print("{s}@{d} = AsType({any}) {s}@{d}\n", .{ ptr.str, @intFromPtr(ptr), new_dtype, x.str, @intFromPtr(&x) });
                }
            }
        };
        return Output.result(self, null, impl.eval, impl.graph);
    }

    pub inline fn map(self: *const Backend, op: ops.MapOp, x: anytype) @TypeOf(x) {
        const Output: type = @TypeOf(x);
        const impl = struct {
            fn eval(out: *Output) Output {
                out.initStorage();
                const x_eval = @call(.always_inline, @TypeOf(x).eval, .{&x});
                switch (self.*) {
                    inline else => |*backend| backend.map(op, x_eval, out),
                }
                return out.*;
            }
            fn graph(out_ptr: *const Output) void {
                x.graph();
                if (@inComptime()) {
                    @compileLog(comptimePrint("{s} = {s} {s}", .{ out_ptr.str, @tagName(op), x.str }));
                } else {
                    std.debug.print("{s}@{d} = {s} {s}@{d}\n", .{ out_ptr.str, @intFromPtr(out_ptr), @tagName(op), x.str, @intFromPtr(&x) });
                }
            }
        };
        return Output.result(self, null, impl.eval, impl.graph);
    }
    pub inline fn zip(self: *const Backend, op: ops.ZipOp, a: anytype, b: anytype) @TypeOf(a).Broadcast(@TypeOf(b)) {
        const Output = @TypeOf(a).Broadcast(@TypeOf(b));
        const impl = struct {
            fn eval(out: *Output) Output {
                out.initStorage();
                const a_eval = @call(.always_inline, @TypeOf(a).eval, .{&a});
                const b_eval = @call(.always_inline, @TypeOf(b).eval, .{&b});
                switch (self.*) {
                    inline else => |*backend| backend.zip(op, a_eval, b_eval, out),
                }
                return out.*;
            }
            fn graph(ptr: *const Output) void {
                a.graph();
                b.graph();
                if (@inComptime()) {
                    @compileLog(comptimePrint("{s} = {s} {s} {s}", .{ ptr.str, a.str, @tagName(op), b.str }));
                } else {
                    std.debug.print("{s}@{d} = {s}@{d} {s} {s}@{d}\n", .{ ptr.str, @intFromPtr(ptr), a.str, @intFromPtr(&a), @tagName(op), b.str, @intFromPtr(&b) });
                }
            }
        };
        return Output.result(self, null, impl.eval, impl.graph);
    }
    pub inline fn reduce(self: *const Backend, op: ops.ReduceOp, x: anytype, comptime dim: ?u8) @TypeOf(x).Reduce(dim) {
        const Output = @TypeOf(x).Reduce(dim);
        const impl = struct {
            fn eval(out: *Output) Output {
                out.initStorage();
                const x_eval = @call(.always_inline, @TypeOf(x).eval, .{&x});
                switch (self.*) {
                    inline else => |*backend| backend.reduce(op, x_eval, dim, out),
                }
                return out.*;
            }

            fn graph(ptr: *const Output) void {
                x.graph();
                if (@inComptime()) {
                    @compileLog(comptimePrint("{s} = {s} {s} {d}", .{ ptr.str, @tagName(op), x.str, dim orelse -1 }));
                } else {
                    std.debug.print("{s}@{d} = {s} {s}@{d} {d}\n", .{ ptr.str, @intFromPtr(ptr), @tagName(op), x.str, @intFromPtr(&x), dim orelse -1 });
                }
            }
        };
        return Output.result(self, null, impl.eval, impl.graph);
    }
};
