const Allocator = @import("std").mem.Allocator;
const tensor = @import("../tensor.zig");
const ops = @import("../ops.zig");
const Backend = @import("../backend.zig").Backend;
const ZigBackend = @This();
const std = @import("std");

const default_vec_len = std.simd.suggestVectorLength(usize) orelse 4;

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
        pub const simd_align = @alignOf(@Vector(default_vec_len, dtype));
        data: []align(simd_align) dtype,
        size: usize,
        pub fn fill(self: *Self, value: dtype) void {
            @memset(self.data, value);
        }
    };
}

// TODO: Replace page with a fixed buffer allocator
// Buffer size should be computed at compile time
pub fn init(_: *const ZigBackend, _: anytype) void {
    GlobalArena.init(std.heap.ArenaAllocator.init(std.heap.page_allocator));
}

pub fn storage(_: *const ZigBackend, comptime dtype: type, comptime size: usize, comptime data: ?[size]dtype) *Backend.Storage(dtype) {
    const store = GlobalArena.allocator().create(Backend.Storage(dtype)) catch @panic("Out of memory");
    store.* = .{
        .Zig = .{
            .data = GlobalArena.allocator().alignedAlloc(dtype, Storage(dtype).simd_align, size) catch @panic("Unable to allocate tensor storage"),
            .size = size,
        },
    };
    if (data != null) {
        @memcpy(store.Zig.data, data[0..]);
    }
    return store;
}

pub fn deinit(_: *const ZigBackend) void {
    GlobalArena.deinit();
}

const MapFnType = @TypeOf(struct {
    fn f(x: anytype) @TypeOf(x) {
        unreachable;
    }
}.f);

fn MapFn(comptime map_op: ops.MapOp, comptime dtype: type) MapFnType {
    return comptime switch (map_op) {
        .Neg => switch (@typeInfo(dtype)) {
            .Bool => struct {
                fn negateBoolEval(x: anytype) @TypeOf(x) {
                    return !x;
                }
            }.negateBoolEval,
            else => struct {
                fn negateEval(x: anytype) @TypeOf(x) {
                    return -x;
                }
            }.negateEval,
        },
        .Log2 => struct {
            fn log2Eval(x: anytype) @TypeOf(x) {
                return @log2(x);
            }
        }.log2Eval,
        .Exp2 => struct {
            fn exp2Eval(x: anytype) @TypeOf(x) {
                return @exp2(x);
            }
        }.exp2Eval,
        .Sqrt => struct {
            fn sqrtEval(x: anytype) @TypeOf(x) {
                return @sqrt(x);
            }
        }.sqrtEval,
        .Recip => struct {
            fn recipEval(x: anytype) @TypeOf(x) {
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
        fn f(a: anytype, _: @TypeOf(a)) switch (zip_op) {
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
            fn addEval(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
                return a + b;
            }
        }.addEval,
        .Mul => struct {
            fn mulEval(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
                return a * b;
            }
        }.mulEval,
        .Maximum => struct {
            // Cast to a shared type
            fn maximumEval(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
                const cast_type = @TypeOf(@as(dtype, 0) * @as(dtype, 0));
                return @max(@as(cast_type, a), @as(cast_type, b));
            }
        }.maximumEval,
        .Lt => struct {
            fn lessThanEval(a: anytype, b: @TypeOf(a)) bool {
                return a < b;
            }
        }.lessThanEval,
        .Eq => struct {
            fn equalsEval(a: anytype, b: @TypeOf(a)) bool {
                return a == b;
            }
        }.equalsEval,
        .Xor => struct {
            fn xorEval(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
                return a ^ b;
            }
        }.xorEval,
        else => @compileError("Not implemented"),
    };
}

fn CastFnType(comptime old_dtype: type, comptime new_dtype: type) type {
    return @TypeOf(struct {
        fn cast(_: old_dtype) new_dtype {
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
                fn cast(x: old_dtype) new_dtype {
                    return @floatFromInt(x);
                }
            },
            .Float => struct {
                fn cast(x: old_dtype) new_dtype {
                    return @floatCast(x);
                }
            },
            else => @compileError(err_msg),
        },
        .Int => switch (old_info) {
            .Float => struct {
                fn cast(x: old_dtype) new_dtype {
                    return @intFromFloat(x);
                }
            },
            .Bool => struct {
                fn cast(x: old_dtype) new_dtype {
                    return @intFromBool(x);
                }
            },
            .Int => struct {
                fn cast(x: old_dtype) new_dtype {
                    return @intCast(x);
                }
            },
            else => @compileError(err_msg),
        },
        else => @compileError(err_msg),
    }.cast;
}

pub fn asType(_: *const ZigBackend, comptime new_dtype: type, x: anytype, out: *tensor.CastedTensor(@TypeOf(x), new_dtype)) void {
    const castFn = CastFn(@TypeOf(x).dtype, new_dtype);
    for (0..@TypeOf(out.*).size) |flat_index| {
        out.storage.?.Zig.data[flat_index] = @call(.always_inline, castFn, .{x.storage.?.Zig.data[flat_index]});
    }
}

pub fn map(_: *const ZigBackend, comptime op: ops.MapOp, x: anytype, out: *@TypeOf(x)) void {
    const mapFn = MapFn(op, @TypeOf(x).dtype);
    const size = @TypeOf(out.*).size;
    const num_vecs = @divFloor(size, default_vec_len);
    inline for (0..num_vecs) |vec_i| {
        const vec_start_i = vec_i * default_vec_len;
        const stop_i = vec_start_i + default_vec_len;
        const x_vec: @Vector(default_vec_len, @TypeOf(x).dtype) = x.storage.?.Zig.data[vec_start_i..stop_i].*;
        out.storage.?.Zig.data[vec_start_i..stop_i].* = @call(.always_inline, mapFn, .{x_vec});
    }
    inline for (num_vecs * default_vec_len..size) |i| {
        out.storage.?.Zig.data[i] = @call(.always_inline, mapFn, .{x.storage.?.Zig.data[i]});
    }
}

pub fn zip(
    _: *const ZigBackend,
    comptime op: ops.ZipOp,
    a: anytype,
    b: anytype,
    out: *tensor.BroadcastedTensor(@TypeOf(a), @TypeOf(b)),
) void {
    const zipFn = ZipFn(op, @TypeOf(a).dtype);
    inline for (0..@TypeOf(out.*).size) |dst| {
        const out_index = out.unflattenIndex(dst);
        out.storage.?.Zig.data[dst] = @call(
            .always_inline,
            zipFn,
            .{
                a.storage.?.Zig.data[a.flattenIndex(a.broadcastIndex(out_index))],
                b.storage.?.Zig.data[b.flattenIndex(b.broadcastIndex(out_index))],
            },
        );
    }
}

inline fn reduce_helper(comptime op: ops.ReduceOp, comptime dtype: type, array: anytype, comptime len: usize) dtype {
    const zipFn = ZipFn(comptime switch (op) {
        .Sum => .Add,
        .Max => .Maximum,
    }, dtype);
    const vec_reduce: std.builtin.ReduceOp = comptime switch (op) {
        .Sum => .Add,
        .Max => .Max,
    };
    var acc: dtype = undefined;
    if (comptime len >= default_vec_len) {
        const init_vec: @Vector(default_vec_len, dtype) = array[0..default_vec_len].*;
        const num_vecs = @divFloor(len, default_vec_len);
        acc = @reduce(vec_reduce, init_vec);
        inline for (1..num_vecs) |vec_i| {
            const vec_start_i = vec_i * default_vec_len;
            const stop_i = vec_start_i + default_vec_len;
            const x_vec: @Vector(default_vec_len, dtype) = array[vec_start_i..stop_i].*;
            acc = @call(.always_inline, zipFn, .{ acc, @reduce(vec_reduce, x_vec) });
        }
        inline for (num_vecs * default_vec_len..len) |i| {
            acc = @call(.always_inline, zipFn, .{ acc, array[i] });
        }
    } else {
        acc = array[0];
        inline for (1..len) |i| {
            acc = @call(.always_inline, zipFn, .{ acc, array[i] });
        }
    }
    return acc;
}

pub fn reduce(
    _: *const ZigBackend,
    comptime op: ops.ReduceOp,
    x: anytype,
    comptime dim: ?u8,
    out: *tensor.ReducedTensor(@TypeOf(x), dim),
) void {
    const dtype: type = @TypeOf(x).dtype;
    const ndims: u8 = @TypeOf(x).ndims;
    const shape: [ndims]usize = @TypeOf(x).shape;
    const x_size: usize = @TypeOf(x).size;
    const out_size: usize = @TypeOf(out.*).size;
    if (ndims == 0 or x_size == 0 or (dim != null and shape[dim.?] == 0)) {
        @compileError("Cannot reduce over 0 elements");
    }
    if (comptime dim == null) {
        out.storage.?.Zig.data[0] = reduce_helper(op, dtype, x.storage.?.Zig.data);
    } else {
        inline for (0..out_size) |out_i| {
            const x_start_i = x.flattenIndex(out.unflattenIndex(out_i));
            const x_stop_i = x_start_i + shape[dim.?];
            out.storage.?.Zig.data[out_i] = reduce_helper(op, dtype, x.storage.?.Zig.data[x_start_i..x_stop_i], shape[dim.?]);
        }
    }
}
