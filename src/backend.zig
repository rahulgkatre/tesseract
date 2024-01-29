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
    pub fn storage(back: *const Backend, id: usize, comptime dtype: type, comptime size: usize, constant: bool) *Storage(dtype) {
        return switch (back.*) {
            inline else => |*b| b.storage(id, dtype, size, constant),
        };
    }
    pub fn finished(back: *const Backend) void {
        tensor.finished();
        return switch (back.*) {
            inline else => |*b| b.finished(),
        };
    }
    pub inline fn cast(back: *const Backend, comptime new_dtype: type, x: anytype) @TypeOf(x.*).Cast(new_dtype) {
        const Out: type = @TypeOf(x.*).Cast(new_dtype);
        const impl = struct {
            fn eval(out: *Out) *Out {
                if (!out.evaluated) {
                    const x_done = x.eval();
                    if (out.storage == null) {
                        out.storage = out.backend.storage(out.id.?, Out.dtype, Out.size, false);
                    }
                    if (tensor.debug) {
                        std.debug.print("tensor_{d} = Cast({s}) tensor_{d}\n", .{ out.id.?, @typeName(new_dtype), x_done.id.? });
                    }
                    switch (back.*) {
                        inline else => |*backend| backend.cast(new_dtype, x_done, out),
                    }
                    out.evaluated = true;
                }
                return out;
            }
        };
        return Out.init(back, impl.eval);
    }

    pub inline fn map(back: *const Backend, op: ops.MapOp, x: anytype) @TypeOf(x.*) {
        const Out: type = @TypeOf(x.*);
        const impl = struct {
            fn eval(out: *Out) *Out {
                if (!out.evaluated) {
                    const x_done = x.eval();
                    if (out.storage == null) {
                        out.storage = out.backend.storage(out.id.?, Out.dtype, Out.size, false);
                    }
                    if (tensor.debug) {
                        std.debug.print("tensor_{d} = {s} tensor_{d}\n", .{ out.id.?, @tagName(op), x_done.id.? });
                    }
                    switch (back.*) {
                        inline else => |*backend| backend.map(op, x_done, out),
                    }
                    out.evaluated = true;
                }
                return out;
            }
        };
        return Out.init(back, impl.eval);
    }
    pub inline fn zip(back: *const Backend, op: ops.ZipOp, a: anytype, b: anytype) @TypeOf(a.*).Broadcast(@TypeOf(b.*)) {
        const Out: type = @TypeOf(a.*).Broadcast(@TypeOf(b.*));
        const impl = struct {
            fn eval(out: *Out) *Out {
                if (!out.evaluated) {
                    const a_done = a.eval();
                    const b_done = b.eval();
                    if (out.storage == null) {
                        out.storage = out.backend.storage(out.id.?, Out.dtype, Out.size, false);
                    }
                    if (tensor.debug) {
                        std.debug.print("tensor_{d} = tensor_{d} {s} tensor_{d}\n", .{ out.id.?, a_done.id.?, @tagName(op), b_done.id.? });
                    }
                    switch (back.*) {
                        inline else => |*backend| backend.zip(op, a_done, b_done, out),
                    }
                    out.evaluated = true;
                }
                return out;
            }
        };
        return Out.init(back, impl.eval);
    }
    pub inline fn reduce(back: *const Backend, op: ops.ReduceOp, x: anytype, comptime dim: ?u8) @TypeOf(x.*).Reduce(dim) {
        const Out: type = @TypeOf(x.*).Reduce(dim);
        const impl = struct {
            fn eval(out: *Out) *Out {
                if (!out.evaluated) {
                    const x_done = x.eval();
                    if (out.storage == null) {
                        out.storage = out.backend.storage(out.id.?, Out.dtype, Out.size, false);
                    }
                    if (tensor.debug) {
                        std.debug.print("tensor_{d} = {s}{s} tensor_{d}\n", .{ out.id.?, @tagName(op), if (dim != null) comptimePrint("({d})", .{dim.?}) else "", x_done.id.? });
                    }
                    switch (back.*) {
                        inline else => |*backend| backend.reduce(op, x_done, dim, out),
                    }
                    out.evaluated = true;
                }
                return out;
            }
        };
        return Out.init(back, impl.eval);
    }
};
