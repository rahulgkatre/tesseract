const tensor = @import("../tensor.zig");
const ops = @import("../ops.zig");
const Backend = @import("../backend.zig").Backend;
const ZigBackend = @This();
const std = @import("std");
const Allocator = std.mem.Allocator;

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
        pub const vec_len = std.simd.suggestVectorLength(dtype) orelse @sizeOf(dtype);
        pub const vec_alignment = @alignOf(@Vector(vec_len, dtype));
        data: []align(vec_alignment) dtype,
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
    const store = GlobalArena.allocator().create(Backend.Storage(dtype)) catch unreachable;
    const store_type = Storage(dtype);
    store.* = .{
        .Zig = .{
            .data = GlobalArena.allocator().alignedAlloc(dtype, store_type.vec_alignment, size) catch unreachable,
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
                    @setFloatMode(.Optimized);
                    return !x;
                }
            }.negateBoolEval,
            else => struct {
                inline fn negateEval(x: anytype) @TypeOf(x) {
                    @setFloatMode(.Optimized);
                    return -x;
                }
            }.negateEval,
        },
        .Log2 => struct {
            inline fn log2Eval(x: anytype) @TypeOf(x) {
                @setFloatMode(.Optimized);
                return @log2(x);
            }
        }.log2Eval,
        .Exp2 => struct {
            inline fn exp2Eval(x: anytype) @TypeOf(x) {
                @setFloatMode(.Optimized);
                return @exp2(x);
            }
        }.exp2Eval,
        .Sqrt => struct {
            inline fn sqrtEval(x: anytype) @TypeOf(x) {
                @setFloatMode(.Optimized);
                return @sqrt(x);
            }
        }.sqrtEval,
        .Recip => struct {
            inline fn recipEval(x: anytype) @TypeOf(x) {
                @setFloatMode(.Optimized);
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
                @setFloatMode(.Optimized);
                return a + b;
            }
        }.addEval,
        .Mul => struct {
            inline fn mulEval(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
                @setFloatMode(.Optimized);
                return a * b;
            }
        }.mulEval,
        .Maximum => struct {
            inline fn maximumEval(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
                @setFloatMode(.Optimized);
                return @max(a, b);
            }
        }.maximumEval,
        .Lt => struct {
            inline fn lessThanEval(a: anytype, b: @TypeOf(a)) bool {
                @setFloatMode(.Optimized);
                return a < b;
            }
        }.lessThanEval,
        .Eq => struct {
            inline fn equalsEval(a: anytype, b: @TypeOf(a)) bool {
                @setFloatMode(.Optimized);
                // TODO: float a == float b should use isClose
                return a == b;
            }
        }.equalsEval,
        .Xor => struct {
            inline fn xorEval(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
                @setFloatMode(.Optimized);
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
    const _x = x.storage.?.Zig.data;
    var _out = out.storage.?.Zig.data;
    for (0..@TypeOf(out.*).size) |i| {
        _out[i] = @call(.always_inline, castFn, .{_x[i]});
    }
}

pub inline fn map(_: *const ZigBackend, comptime op: ops.MapOp, x: anytype, out: *@TypeOf(x)) void {
    const mapFn = MapFn(op, @TypeOf(x).dtype);
    const dtype: type = @TypeOf(x).dtype;
    const size: usize = @TypeOf(x).size;
    const storage_type = Storage(dtype);
    const _x: []align(storage_type.vec_alignment) dtype = x.storage.?.Zig.data;
    var _out: []align(storage_type.vec_alignment) dtype = out.storage.?.Zig.data;
    const vec_len: usize = storage_type.vec_len;
    const num_vecs: usize = comptime @divTrunc(size, vec_len);
    for (0..num_vecs) |vec_i| {
        const offset = vec_i * vec_len;
        @prefetch(_x[offset + vec_len ..], .{ .locality = 0 });
        const aligned_out: *@Vector(vec_len, dtype) = @alignCast(@ptrCast(_out[offset..][0..vec_len]));
        aligned_out.* = @call(.always_inline, mapFn, .{@as(*const @Vector(vec_len, dtype), @alignCast(@ptrCast(_x[offset..][0..vec_len]))).*});
    }
    inline for (comptime vec_len * num_vecs..size) |i| {
        _out[i] = @call(.always_inline, mapFn, .{_x[i]});
    }
}

pub inline fn zip(_: *const ZigBackend, comptime op: ops.ZipOp, a: anytype, b: anytype, out: *@TypeOf(a).Broadcast(@TypeOf(b))) void {
    const zipFn = ZipFn(op, @TypeOf(a).dtype);
    const dtype: type = @TypeOf(a).dtype;
    const storage_type = Storage(dtype);
    const _a: []align(storage_type.vec_alignment) dtype = a.storage.?.Zig.data;
    const _b: []align(storage_type.vec_alignment) dtype = b.storage.?.Zig.data;
    var _out: []align(storage_type.vec_alignment) dtype = out.storage.?.Zig.data;
    if (@TypeOf(a) == @TypeOf(b)) {
        // Vectorize perfectly when a and b have same type (shape, strides, etc.)
        const size: usize = @TypeOf(a).size;
        const vec_len: usize = storage_type.vec_len;
        const num_vecs: usize = comptime @divTrunc(size, vec_len);
        for (0..num_vecs) |vec_i| {
            const offset = vec_i * vec_len;
            @prefetch(_a[offset + vec_len ..], .{ .locality = 0 });
            @prefetch(_b[offset + vec_len ..], .{ .locality = 0 });
            const aligned_out: *@Vector(vec_len, dtype) = @alignCast(@ptrCast(_out[offset..][0..vec_len]));
            aligned_out.* = @call(.always_inline, zipFn, .{
                @as(*const @Vector(vec_len, dtype), @alignCast(@ptrCast(_a[offset..][0..vec_len]))).*,
                @as(*const @Vector(vec_len, dtype), @alignCast(@ptrCast(_b[offset..][0..vec_len]))).*,
            });
        }
        inline for (comptime vec_len * num_vecs..size) |i| {
            _out[i] = @call(.always_inline, zipFn, .{ _a[i], _b[i] });
        }
    } else {
        // TODO: Make this vectorize better
        for (comptime 0..@TypeOf(out.*).size) |out_i| {
            const out_index = out.unflattenIndex(out_i);
            _out[out_i] = @call(.always_inline, zipFn, .{
                _a[a.flattenIndex(a.broadcastIndex(out_index))],
                _b[b.flattenIndex(b.broadcastIndex(out_index))],
            });
        }
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

    const reduceOp: std.builtin.ReduceOp = comptime switch (op) {
        .Sum => .Add,
        .Max => .Max,
    };

    const storage_type = Storage(dtype);
    const _x: []align(storage_type.vec_alignment) dtype = x.storage.?.Zig.data;
    var _out: []align(storage_type.vec_alignment) dtype = out.storage.?.Zig.data;
    const vec_len: usize = storage_type.vec_len;
    const num_vecs: usize = comptime @divTrunc(size, vec_len);
    if (comptime dim == null) {
        var acc: dtype = @reduce(reduceOp, @as(*const @Vector(vec_len, dtype), @alignCast(@ptrCast(_x[0..vec_len]))).*);
        for (1..num_vecs) |vec_i| {
            const offset = vec_i * vec_len;
            @prefetch(_x[offset + vec_len ..], .{ .locality = 0 });
            acc = @call(.always_inline, zipFn, .{
                acc,
                @reduce(reduceOp, @as(*const @Vector(vec_len, dtype), @alignCast(@ptrCast(_x[offset..][0..vec_len]))).*),
            });
        }
        inline for (comptime vec_len * num_vecs..size) |i| {
            acc = @call(.always_inline, zipFn, .{ acc, _x[i] });
        }
        out.storage.?.Zig.data[0] = acc;
    } else {
        // TODO: Make this vectorize better
        const stride = strides[dim.?];
        const dimsize = shape[dim.?];
        for (comptime 0..@TypeOf(out.*).size) |out_i| {
            const offset = x.flattenIndex(out.unflattenIndex(out_i));
            var acc = _x[offset];
            for (comptime 1..dimsize) |i| {
                const x_i = offset + i * stride;
                acc = @call(.always_inline, zipFn, .{ acc, _x[x_i] });
            }
            _out[out_i] = acc;
        }
    }
}
