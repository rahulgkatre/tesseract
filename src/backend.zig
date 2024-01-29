const std = @import("std");
const Allocator = std.mem.Allocator;
const ops = @import("ops.zig");
const comptimePrint = std.fmt.comptimePrint;
const tensor = @import("tensor.zig");
const ZigBackend = @import("backend/ZigBackend.zig");
const CodegenBackend = @import("backend/CodegenBackend.zig");

pub const BackendTypes = enum {
    Zig,
    Codegen,
};

pub const Backend = union(BackendTypes) {
    Zig: ZigBackend,
    Codegen: CodegenBackend,

    pub fn Storage(comptime dtype: type) type {
        return union(BackendTypes) {
            const StorageType = @This();
            Zig: ZigBackend.Storage(dtype),
            Codegen: CodegenBackend.Storage(dtype),
            pub fn fill(store: *StorageType, value: dtype) void {
                switch (store.*) {
                    inline else => |*s| s.fill(value),
                }
            }
            pub fn load(store: *StorageType, data: []const dtype) void {
                switch (store.*) {
                    inline else => |*s| s.load(data),
                }
            }
        };
    }

    pub fn runtime(back: *const Backend, args: anytype) void {
        tensor.runtime();
        return switch (back.*) {
            inline else => |*b| b.runtime(args),
        };
    }
    pub fn storage(back: *const Backend, comptime dtype: type, comptime size: usize) *Storage(dtype) {
        return switch (back.*) {
            inline else => |*b| b.storage(dtype, size),
        };
    }
    pub fn finished(back: *const Backend) void {
        tensor.finished();
        return switch (back.*) {
            inline else => |*b| b.finished(),
        };
    }
    pub inline fn cast(back: *const Backend, comptime new_dtype: type, x_ptr: anytype) @TypeOf(x_ptr.*).Cast(new_dtype) {
        const Out: type = @TypeOf(x_ptr.*).Cast(new_dtype);
        const impl = struct {
            fn eval(out: *Out) *Out {
                if (!out.evaluated) {
                    if (x_ptr.evaluated) {
                        std.debug.print("Already evaluated {s}", .{x_ptr.str});
                    }
                    const x_eval = @call(.always_inline, x_ptr.evalFn, .{x_ptr.runtime(0)});
                    out.id = x_eval.id.? + 1;
                    if (tensor.debug) {
                        std.debug.print("t{d} = Cast({s}) t{d}\n", .{ out.id.?, @typeName(new_dtype), x_eval.id.? });
                    }
                    switch (back.*) {
                        inline else => |*backend| backend.cast(new_dtype, x_eval, out),
                    }
                    out.evaluated = true;
                }
                return out;
            }
        };
        return Out.result(back, null, impl.eval);
    }

    pub inline fn map(back: *const Backend, op: ops.MapOp, x_ptr: anytype) @TypeOf(x_ptr.*) {
        const Out: type = @TypeOf(x_ptr.*);
        const impl = struct {
            fn eval(out: *Out) *Out {
                if (!out.evaluated) {
                    const x_eval = @call(.auto, x_ptr.evalFn, .{x_ptr.runtime(0)});
                    out.id = x_eval.id.? + 1;
                    if (tensor.debug) {
                        std.debug.print("t{d} = {s} t{d}\n", .{ out.id.?, @tagName(op), x_eval.id.? });
                    }
                    switch (back.*) {
                        inline else => |*backend| backend.map(op, x_eval, out),
                    }
                    out.evaluated = true;
                }
                return out;
            }
        };
        return Out.result(back, null, impl.eval);
    }
    pub inline fn zip(back: *const Backend, op: ops.ZipOp, a_ptr: anytype, b_ptr: anytype) @TypeOf(a_ptr.*).Broadcast(@TypeOf(b_ptr.*)) {
        const Out: type = @TypeOf(a_ptr.*).Broadcast(@TypeOf(b_ptr.*));
        const impl = struct {
            fn eval(out: *Out) *Out {
                if (!out.evaluated) {
                    const a_eval = a_ptr.eval();
                    const b_eval = b_ptr.evalFn(b_ptr.runtime(a_eval.id.? + 1));
                    out.id = b_eval.id.? + 1;
                    if (tensor.debug) {
                        std.debug.print("t{d} = t{d} {s} t{d}\n", .{ out.id.?, a_eval.id.?, @tagName(op), b_eval.id.? });
                    }
                    switch (back.*) {
                        inline else => |*backend| backend.zip(op, a_eval, b_eval, out),
                    }
                    out.evaluated = true;
                }
                return out;
            }
        };
        return Out.result(back, null, impl.eval);
    }
    pub inline fn reduce(back: *const Backend, op: ops.ReduceOp, x_ptr: anytype, comptime dim: ?u8) @TypeOf(x_ptr.*).Reduce(dim) {
        const Out: type = @TypeOf(x_ptr.*).Reduce(dim);
        const impl = struct {
            fn eval(out: *Out) *Out {
                if (!out.evaluated) {
                    const x_eval = x_ptr.eval();
                    out.id = x_eval.id.? + 1;
                    if (tensor.debug) {
                        std.debug.print("t{d} = {s}{s} t{d}\n", .{ out.id.?, @tagName(op), if (dim != null) comptimePrint("({d})", .{dim.?}) else "", x_eval.id.? });
                    }
                    switch (back.*) {
                        inline else => |*backend| backend.reduce(op, x_eval, dim, out),
                    }
                    out.evaluated = true;
                }
                return out;
            }
        };
        return Out.result(back, null, impl.eval);
    }
};
