const Allocator = @import("std").mem.Allocator;
const tensor = @import("../tensor.zig");
const ops = @import("../ops.zig");
const Backend = @import("../backend.zig").Backend;
const ZigBackend = @This();
const std = @import("std");

const GlobalArena = struct {
    var global_arena: std.heap.ArenaAllocator = undefined;
    fn init(arena: std.heap.ArenaAllocator) void {
        global_arena = arena;
    }
    fn deinit() void {
        global_arena.deinit();
        global_arena = undefined;
    }
    fn allocator() std.mem.Allocator {
        return global_arena.allocator();
    }
};

pub fn Storage(comptime dtype: type) type {
    return struct {
        const Self = @This();
        // pub const default_vec_len = std.simd.suggestVectorLength(dtype) orelse 4;
        // pub const simd_align = @alignOf(@Vector(default_vec_len, dtype));
        data: []dtype,
        size: usize,
        pub fn fill(self: *Self, value: dtype) void {
            @memset(self.data, value);
        }
        pub fn load(self: *Self, data: []const dtype) void {
            @memcpy(self.data, data);
        }
    };
}

// TODO: Replace page allocator with a fixed buffer allocator
// Buffer size should be computed at compile time
pub fn init(_: *const ZigBackend, _: anytype) void {
    GlobalArena.init(std.heap.ArenaAllocator.init(std.heap.page_allocator));
}

pub fn storage(_: *const ZigBackend, comptime dtype: type, comptime size: usize) *Backend.Storage(dtype) {
    const store = GlobalArena.allocator().create(Backend.Storage(dtype)) catch @panic("Out of memory");
    store.* = .{
        .Zig = .{
            // .data = GlobalArena.allocator().alignedAlloc(dtype, Storage(dtype).simd_align, size) catch @panic("Unable to allocate tensor storage"),
            .data = GlobalArena.allocator().alloc(dtype, size) catch @panic("Unable to allocate tensor storage"),
            .size = size,
        },
    };
    return store;
}

pub fn deinit(_: *const ZigBackend) void {
    GlobalArena.deinit();
}

const MapFnType = @TypeOf(struct {
    inline fn f(x: anytype) @TypeOf(x) {
        unreachable;
    }
}.f);

fn MapFn(comptime map_op: ops.MapOp, comptime dtype: type) MapFnType {
    return comptime switch (map_op) {
        .Neg => switch (@typeInfo(dtype)) {
            .Bool => struct {
                inline fn negateBoolEval(x: anytype) @TypeOf(x) {
                    return !x;
                }
            }.negateBoolEval,
            else => struct {
                inline fn negateEval(x: anytype) @TypeOf(x) {
                    return -x;
                }
            }.negateEval,
        },
        .Log2 => struct {
            inline fn log2Eval(x: anytype) @TypeOf(x) {
                return @log2(x);
            }
        }.log2Eval,
        .Exp2 => struct {
            inline fn exp2Eval(x: anytype) @TypeOf(x) {
                return @exp2(x);
            }
        }.exp2Eval,
        .Sqrt => struct {
            inline fn sqrtEval(x: anytype) @TypeOf(x) {
                return @sqrt(x);
            }
        }.sqrtEval,
        .Recip => struct {
            inline fn recipEval(x: anytype) @TypeOf(x) {
                return switch (@typeInfo(@TypeOf(x))) {
                    .Vector => |v| @as(@Vector(v.len, v.child), @splat(1.0)) / x,
                    else => 1.0 / x,
                };
            }
        }.recipEval,
        else => @compileError("Not implemented"),
    };
}

fn ZipFnType(comptime zip_op: ops.ZipOp, comptime dtype: type) type {
    return @TypeOf(struct {
        inline fn f(a: anytype, _: @TypeOf(a)) switch (zip_op) {
            .Lt, .Eq => bool,
            else => dtype,
        } {
            unreachable;
        }
    }.f);
}

fn ZipFn(comptime zip_op: ops.ZipOp, comptime dtype: type) ZipFnType(zip_op, dtype) {
    return comptime switch (zip_op) {
        .Add => struct {
            inline fn addEval(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
                return a + b;
            }
        }.addEval,
        .Mul => struct {
            inline fn mulEval(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
                return a * b;
            }
        }.mulEval,
        .Maximum => struct {
            inline fn maximumEval(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
                return @max(a, b);
            }
        }.maximumEval,
        .Lt => struct {
            inline fn lessThanEval(a: anytype, b: @TypeOf(a)) bool {
                return a < b;
            }
        }.lessThanEval,
        .Eq => struct {
            inline fn equalsEval(a: anytype, b: @TypeOf(a)) bool {
                return a == b;
            }
        }.equalsEval,
        .Xor => struct {
            inline fn xorEval(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
                return a ^ b;
            }
        }.xorEval,
        else => @compileError("Not implemented"),
    };
}

fn CastFnType(comptime old_dtype: type, comptime new_dtype: type) type {
    return @TypeOf(struct {
        inline fn cast(_: old_dtype) new_dtype {
            unreachable;
        }
    }.cast);
}

fn CastFn(comptime old_dtype: type, comptime new_dtype: type) CastFnType(old_dtype, new_dtype) {
    const old_info = @typeInfo(old_dtype);
    const new_info = @typeInfo(new_dtype);
    const err_msg = std.fmt.comptimePrint("Cannot cast dtype {} to {}", .{ old_dtype, new_dtype });
    return comptime switch (new_info) {
        .Float => switch (old_info) {
            .Int => struct {
                inline fn cast(x: old_dtype) new_dtype {
                    return @floatFromInt(x);
                }
            },
            .Float => struct {
                inline fn cast(x: old_dtype) new_dtype {
                    return @floatCast(x);
                }
            },
            else => @compileError(err_msg),
        },
        .Int => switch (old_info) {
            .Float => struct {
                inline fn cast(x: old_dtype) new_dtype {
                    return @intFromFloat(x);
                }
            },
            .Bool => struct {
                inline fn cast(x: old_dtype) new_dtype {
                    return @intFromBool(x);
                }
            },
            .Int => struct {
                inline fn cast(x: old_dtype) new_dtype {
                    return @intCast(x);
                }
            },
            else => @compileError(err_msg),
        },
        else => @compileError(err_msg),
    }.cast;
}

pub inline fn asType(_: *const ZigBackend, comptime new_dtype: type, x: anytype, out: *@TypeOf(x).AsType(new_dtype)) void {
    const castFn = CastFn(@TypeOf(x).dtype, new_dtype);
    for (0..@TypeOf(out.*).size) |i| {
        out.storage.?.Zig.data[i] = @call(.always_inline, castFn, .{x.storage.?.Zig.data[i]});
    }
}

pub inline fn map(_: *const ZigBackend, comptime op: ops.MapOp, x: anytype, out: *@TypeOf(x)) void {
    const dtype: type = @TypeOf(x).dtype;
    const mapFn = MapFn(op, dtype);
    const size = @TypeOf(out.*).size;
    for (comptime 0..size) |i| {
        const elem = x.storage.?.Zig.data[i];
        out.storage.?.Zig.data[i] = @call(.always_inline, mapFn, .{elem});
    }
    // const vec: @Vector(size, dtype) = x.storage.?.Zig.data[0..size].*;
    // out.storage.?.Zig.data[0..size].* = mapFn(vec);
}

pub inline fn zip(_: *const ZigBackend, comptime op: ops.ZipOp, a: anytype, b: anytype, out: *@TypeOf(a).Broadcast(@TypeOf(b))) void {
    const zipFn = ZipFn(op, @TypeOf(a).dtype);
    const size = @TypeOf(out.*).size;
    for (0..size) |i| {
        const out_index = out.unflattenIndex(i);
        out.storage.?.Zig.data[i] = @call(
            .always_inline,
            zipFn,
            .{
                a.storage.?.Zig.data[a.flattenIndex(a.broadcastIndex(out_index))],
                b.storage.?.Zig.data[b.flattenIndex(b.broadcastIndex(out_index))],
            },
        );
    }
}

pub inline fn reduce(
    _: *const ZigBackend,
    comptime op: ops.ReduceOp,
    x: anytype,
    comptime dim: ?u8,
    out: *@TypeOf(x).Reduce(dim),
) void {
    const dtype: type = @TypeOf(x).dtype;
    const ndims: u8 = @TypeOf(x).ndims;
    const shape: [ndims]usize = @TypeOf(x).shape;
    const strides: [ndims + 1]usize = @TypeOf(x).strides;
    const size: usize = @TypeOf(x).size;
    const zipFn = comptime ZipFn(switch (op) {
        .Sum => .Add,
        .Max => .Maximum,
    }, dtype);

    if (ndims == 0 or size == 0 or (dim != null and shape[dim.?] == 0)) {
        @compileError("Cannot reduce over 0 elements");
    }
    if (comptime dim == null) {
        var acc = x.storage.?.Zig.data[0];
        for (1..size) |i| {
            acc = @call(.always_inline, zipFn, .{ acc, x.storage.?.Zig.data[i] });
        }
        out.storage.?.Zig.data[0] = acc;
        // const vec: @Vector(size, dtype) = x.storage.?.Zig.data[0..size].*;
        // out.storage.?.Zig.data[0] = @reduce(comptime switch (op) {
        //     .Sum => .Add,
        //     .Max => .Max,
        // }, vec);
    } else {
        for (0..@TypeOf(out.*).size) |out_i| {
            const x_start_i = x.flattenIndex(out.unflattenIndex(out_i));
            var acc = x.storage.?.Zig.data[x_start_i];
            for (1..shape[dim.?]) |i| {
                acc = @call(.always_inline, zipFn, .{ acc, x.storage.?.Zig.data[x_start_i + i * strides[dim.?]] });
            }
            out.storage.?.Zig.data[out_i] = acc;
        }
    }
}
