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

    pub fn runtime(self: *const Backend, args: anytype) void {
        tensor.runtime();
        return switch (self.*) {
            inline else => |*b| b.runtime(args),
        };
    }
    pub fn storage(self: *const Backend, comptime dtype: type, comptime size: usize) *Storage(dtype) {
        return switch (self.*) {
            inline else => |*b| b.storage(dtype, size),
        };
    }
    pub fn finished(self: *const Backend) void {
        tensor.finished();
        return switch (self.*) {
            inline else => |*b| b.finished(),
        };
    }
    pub inline fn cast(self: *const Backend, comptime new_dtype: type, x_ptr: anytype) @TypeOf(x_ptr.*).Cast(new_dtype) {
        const Output = @TypeOf(x_ptr.*).Cast(new_dtype);
        const impl = struct {
            fn eval(out_ptr: *Output) *Output {
                const x_eval = @call(.always_inline, @TypeOf(x_ptr.*).eval, .{x_ptr});
                const out_eval = out_ptr.runtime();
                switch (self.*) {
                    inline else => |*backend| backend.cast(new_dtype, x_eval, out_ptr),
                }
                return out_eval;
            }
            fn graph(_: *const Output, id: usize) usize {
                const next_id = @call(.auto, x_ptr.graphFn, .{ x_ptr, id + 1 });
                if (@inComptime()) {
                    @compileLog(comptimePrint("tensor{d} = Cast({any}) tensor{d}", .{ id, new_dtype, id + 1 }));
                } else {
                    std.debug.print("tensor{d} = Cast({any}) tensor{d}\n", .{ id, new_dtype, id + 1 });
                }
                return next_id;
            }
        };
        return Output.result(self, null, impl.eval, impl.graph);
    }

    pub inline fn map(self: *const Backend, op: ops.MapOp, x_ptr: anytype) @TypeOf(x_ptr.*) {
        const Output: type = @TypeOf(x_ptr.*);
        const impl = struct {
            fn eval(out_ptr: *Output) *Output {
                const x_eval = @call(.always_inline, @TypeOf(x_ptr.*).eval, .{x_ptr});
                const out_eval = out_ptr.runtime();
                switch (self.*) {
                    inline else => |*backend| backend.map(op, x_eval, out_eval),
                }
                return out_eval;
            }
            fn graph(_: *const Output, id: usize) usize {
                const next_id = @call(.auto, x_ptr.graphFn, .{ x_ptr, id + 1 });
                if (@inComptime()) {
                    @compileLog(comptimePrint("tensor{d} = {s} tensor{d}", .{ id, @tagName(op), id + 1 }));
                } else {
                    std.debug.print("tensor{d} = {s} tensor{d}\n", .{ id, @tagName(op), id + 1 });
                }
                return next_id;
            }
        };
        return Output.result(self, null, impl.eval, impl.graph);
    }
    pub inline fn zip(self: *const Backend, op: ops.ZipOp, a_ptr: anytype, b_ptr: anytype) @TypeOf(a_ptr.*).Broadcast(@TypeOf(b_ptr.*)) {
        const Output = @TypeOf(a_ptr.*).Broadcast(@TypeOf(b_ptr.*));
        const impl = struct {
            fn eval(out_ptr: *Output) *Output {
                const a_eval = @call(.always_inline, @TypeOf(a_ptr.*).eval, .{a_ptr});
                const b_eval = @call(.always_inline, @TypeOf(b_ptr.*).eval, .{b_ptr});
                const out_eval = out_ptr.runtime();
                switch (self.*) {
                    inline else => |*backend| backend.zip(op, a_eval, b_eval, out_eval),
                }
                return out_eval;
            }
            fn graph(_: *const Output, id: usize) usize {
                const a_next_id = @call(.auto, a_ptr.graphFn, .{ a_ptr, id + 1 });
                const b_next_id = @call(.auto, b_ptr.graphFn, .{ b_ptr, a_next_id });
                if (@inComptime()) {
                    @compileLog(comptimePrint("tensor{d} = tensor{d} {s} tensor{d}", .{ id, id + 1, @tagName(op), a_next_id }));
                } else {
                    std.debug.print("tensor{d} = tensor{d} {s} tensor{d}\n", .{ id, id + 1, @tagName(op), a_next_id });
                }
                return b_next_id;
            }
        };
        return Output.result(self, null, impl.eval, impl.graph);
    }
    pub inline fn reduce(self: *const Backend, op: ops.ReduceOp, x_ptr: anytype, comptime dim: ?u8) @TypeOf(x_ptr.*).Reduce(dim) {
        const Output = @TypeOf(x_ptr.*).Reduce(dim);
        const impl = struct {
            fn eval(out_ptr: *Output) *Output {
                const x_eval = @call(.always_inline, @TypeOf(x_ptr.*).eval, .{x_ptr});
                const out_eval = out_ptr.runtime();
                switch (self.*) {
                    inline else => |*backend| backend.reduce(op, x_eval, dim, out_eval),
                }
                return out_eval;
            }
            fn graph(_: *const Output, id: usize) usize {
                const next_id = @call(.auto, x_ptr.graphFn, .{ x_ptr, id + 1 });
                if (@inComptime()) {
                    @compileLog(comptimePrint("tensor{d} = {s} tensor{d} {d}", .{ id, @tagName(op), id + 1, dim orelse -1 }));
                } else {
                    std.debug.print("tensor{d} = {s} tensor{d} {d}\n", .{ id, @tagName(op), id + 1, dim orelse -1 });
                }
                return next_id;
            }
        };
        return Output.result(self, null, impl.eval, impl.graph);
    }
};
