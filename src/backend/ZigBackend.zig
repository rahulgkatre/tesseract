const tensor = @import("../tensor.zig");
const ops = @import("../ops.zig");
const Backend = @import("../backend.zig").Backend;
const ZigBackend = @This();
const std = @import("std");
const Allocator = std.mem.Allocator;

const VectorizationMode = enum { Scalar, Chunked, Automatic };
const global_vec_mode: VectorizationMode = .Scalar;

const StorageArena = struct {
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

pub fn storage(_: *const ZigBackend, comptime dtype: type, comptime size: usize) *Backend.Storage(dtype) {
    const store = StorageArena.allocator().create(Backend.Storage(dtype)) catch unreachable;
    const store_type = Storage(dtype);
    store.* = .{
        .Zig = .{
            .data = StorageArena.allocator().alignedAlloc(dtype, store_type.vec_alignment, size) catch unreachable,
            .size = size,
        },
    };
    return store;
}

// TODO: Replace page allocator with a fixed buffer allocator
// Buffer size should be computed at compile time
pub fn runtime(_: *const ZigBackend, _: anytype) void {
    StorageArena.init(std.heap.ArenaAllocator.init(std.heap.page_allocator));
}

pub fn finished(_: *const ZigBackend) void {
    StorageArena.deinit();
}

const MapFn = @TypeOf(struct {
    inline fn f(x: anytype) @TypeOf(x) {
        unreachable;
    }
}.f);

fn getMapFn(comptime map_op: ops.MapOp, comptime dtype: type) MapFn {
    return comptime switch (map_op) {
        .Neg => switch (@typeInfo(dtype)) {
            .Bool => struct {
                inline fn not(x: anytype) @TypeOf(x) {
                    return !x;
                }
            }.not,
            else => struct {
                inline fn neg(x: anytype) @TypeOf(x) {
                    @setFloatMode(.Optimized);
                    return -x;
                }
            }.neg,
        },
        .Log2 => struct {
            inline fn log2(x: anytype) @TypeOf(x) {
                @setFloatMode(.Optimized);
                return @log2(x);
            }
        }.log2,
        .Exp2 => struct {
            inline fn exp2(x: anytype) @TypeOf(x) {
                @setFloatMode(.Optimized);
                return @exp2(x);
            }
        }.exp2,
        .Sqrt => struct {
            inline fn sqrt(x: anytype) @TypeOf(x) {
                @setFloatMode(.Optimized);
                return @sqrt(x);
            }
        }.sqrt,
        .Recip => struct {
            inline fn recip(x: anytype) @TypeOf(x) {
                @setFloatMode(.Optimized);
                return switch (@typeInfo(@TypeOf(x))) {
                    .Vector => |v| @as(@Vector(v.len, v.child), @splat(1.0)) / x,
                    else => 1.0 / x,
                };
            }
        }.recip,
        else => @compileError("Not implemented"),
    };
}

pub inline fn map(_: *const ZigBackend, comptime op: ops.MapOp, x_ptr: anytype, out: *@TypeOf(x_ptr.*)) void {
    const map_fn = getMapFn(op, @TypeOf(x_ptr.*).dtype);
    const dtype: type = @TypeOf(x_ptr.*).dtype;
    const size: usize = @TypeOf(x_ptr.*).size;
    const storage_type = Storage(dtype);
    const vec_len: usize = storage_type.vec_len;
    const num_vecs: usize = comptime @divTrunc(size, vec_len);

    const _x: []align(storage_type.vec_alignment) dtype = x_ptr.*.storage.?.Zig.data;
    var _out: []align(storage_type.vec_alignment) dtype = out.storage.?.Zig.data;
    switch (global_vec_mode) {
        .Scalar => {
            for (0..size) |i| {
                _out[i] = @call(.always_inline, map_fn, .{_x[i]});
            }
        },
        .Chunked => {
            for (0..num_vecs) |vec_i| {
                const offset = vec_i * vec_len;
                @prefetch(_x[offset + vec_len ..], .{ .locality = 0 });
                const aligned_out: *@Vector(vec_len, dtype) = @alignCast(@ptrCast(_out[offset..][0..vec_len]));
                aligned_out.* = @call(.always_inline, map_fn, .{@as(*const @Vector(vec_len, dtype), @alignCast(@ptrCast(_x[offset..][0..vec_len]))).*});
            }
            inline for (comptime vec_len * num_vecs..size) |i| {
                _out[i] = @call(.always_inline, map_fn, .{_x[i]});
            }
        },
        .Automatic => {
            const vec_x = @as(@Vector(size, dtype), _x[0..size].*);
            const result = @as([]dtype, @constCast(@ptrCast(@as([size]dtype, @call(.always_inline, map_fn, .{vec_x}))[0..])));
            @memcpy(_out, result);
            // This emits unaligned (vmovups)
            // _out[0..size].* = result[0..size].*;
        },
    }
}

fn getCastFn(comptime old_dtype: type, comptime new_dtype: type) (fn (old_dtype) callconv(.Inline) new_dtype) {
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

pub inline fn cast(
    _: *const ZigBackend,
    comptime new_dtype: type,
    x_ptr: anytype,
    out: *@TypeOf(x_ptr.*).Cast(new_dtype),
) void {
    const castFn = getCastFn(@TypeOf(x_ptr.*).dtype, new_dtype);
    const _x = x_ptr.storage.?.Zig.data;
    var _out = out.storage.?.Zig.data;
    for (0..@TypeOf(out.*).size) |i| {
        _out[i] = @call(.always_inline, castFn, .{_x[i]});
    }
}

fn ZipFn(comptime zip_op: ops.ZipOp, comptime dtype: type) type {
    return @TypeOf(switch (zip_op) {
        .Lt, .Eq => struct {
            inline fn func(a: anytype, _: @TypeOf(a)) bool {
                unreachable;
            }
        },
        else => struct {
            inline fn func(a: anytype, _: @TypeOf(a)) dtype {
                unreachable;
            }
        },
    }.func);
}
fn getZipFn(comptime zip_op: ops.ZipOp, comptime dtype: type) ZipFn(zip_op, dtype) {
    return comptime switch (zip_op) {
        .Add => struct {
            inline fn add(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
                @setFloatMode(.Optimized);
                return a + b;
            }
        }.add,
        .Mul => struct {
            inline fn mul(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
                @setFloatMode(.Optimized);
                return a * b;
            }
        }.mul,
        .Maximum => struct {
            inline fn maximum(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
                @setFloatMode(.Optimized);
                return @max(a, b);
            }
        }.maximum,
        .Lt => struct {
            inline fn lessThan(a: anytype, b: @TypeOf(a)) bool {
                @setFloatMode(.Optimized);
                return a < b;
            }
        }.lessThan,
        .Eq => struct {
            inline fn equals(a: anytype, b: @TypeOf(a)) bool {
                @setFloatMode(.Optimized);
                // TODO: float a == float b should use isClose
                return a == b;
            }
        }.equals,
        .Xor => struct {
            inline fn xor(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
                @setFloatMode(.Optimized);
                return a ^ b;
            }
        }.xor,
        else => @compileError("Not implemented"),
    };
}
pub inline fn zip(
    _: *const ZigBackend,
    comptime op: ops.ZipOp,
    a_ptr: anytype,
    b_ptr: anytype,
    out: *@TypeOf(a_ptr.*).Broadcast(@TypeOf(b_ptr.*)),
) void {
    const zip_fn = getZipFn(op, @TypeOf(a_ptr.*).dtype);
    const dtype: type = @TypeOf(a_ptr.*).dtype;
    const storage_type = Storage(dtype);
    const _a: []align(storage_type.vec_alignment) dtype = a_ptr.storage.?.Zig.data;
    const _b: []align(storage_type.vec_alignment) dtype = b_ptr.storage.?.Zig.data;
    var _out: []align(storage_type.vec_alignment) dtype = out.storage.?.Zig.data;
    if (@TypeOf(a_ptr.*) == @TypeOf(b_ptr.*)) {
        const size: usize = @TypeOf(a_ptr.*).size;
        switch (global_vec_mode) {
            .Scalar => {
                for (0..size) |i| {
                    _out[i] = @call(.always_inline, zip_fn, .{ _a[i], _b[i] });
                }
            },
            .Chunked => {
                const vec_len: usize = storage_type.vec_len;
                const num_vecs: usize = comptime @divTrunc(size, vec_len);
                for (0..num_vecs) |vec_i| {
                    const offset = vec_i * vec_len;
                    @prefetch(_a[offset + vec_len ..], .{ .locality = 0 });
                    @prefetch(_b[offset + vec_len ..], .{ .locality = 0 });
                    const aligned_out: *@Vector(vec_len, dtype) = @alignCast(@ptrCast(_out[offset..][0..vec_len]));
                    aligned_out.* = @call(.always_inline, zip_fn, .{
                        @as(*const @Vector(vec_len, dtype), @alignCast(@ptrCast(_a[offset..][0..vec_len]))).*,
                        @as(*const @Vector(vec_len, dtype), @alignCast(@ptrCast(_b[offset..][0..vec_len]))).*,
                    });
                }
                inline for (vec_len * num_vecs..size) |i| {
                    _out[i] = @call(.always_inline, zip_fn, .{ _a[i], _b[i] });
                }
            },
            .Automatic => {
                const aligned_src_a = @as(*const @Vector(size, dtype), @alignCast(@ptrCast(_b)));
                const aligned_src_b = @as(*const @Vector(size, dtype), @alignCast(@ptrCast(_b)));
                const result = @as([]align(storage_type.vec_alignment) dtype, @alignCast(@constCast(@ptrCast(@as([size]dtype, @call(.always_inline, zip_fn, .{ aligned_src_a.*, aligned_src_b.* }))[0..]))));
                @memcpy(_out, result);
            },
        }
    } else {
        for (0..@TypeOf(out.*).size) |out_i| {
            const out_index = out.posToIdx(out_i);
            _out[out_i] = @call(.always_inline, zip_fn, .{
                _a[a_ptr.idxToPos(a_ptr.broadcastIndex(out_index))],
                _b[b_ptr.idxToPos(b_ptr.broadcastIndex(out_index))],
            });
        }
    }
}

pub inline fn reduce(
    _: *const ZigBackend,
    comptime op: ops.ReduceOp,
    x_ptr: anytype,
    comptime dim: ?u8,
    out: *@TypeOf(x_ptr.*).Reduce(dim),
) void {
    const dtype: type = @TypeOf(x_ptr.*).dtype;
    const ndims: u8 = @TypeOf(x_ptr.*).ndims;
    const shape: [ndims]usize = @TypeOf(x_ptr.*).shape;
    const strides: [ndims + 1]usize = @TypeOf(x_ptr.*).strides;
    const size: usize = @TypeOf(x_ptr.*).size;

    if (ndims == 0 or size == 0 or (dim != null and shape[dim.?] == 0)) {
        @compileError("Cannot reduce over 0 elements");
    }

    const zip_fn = comptime getZipFn(switch (op) {
        .Sum => .Add,
        .Max => .Maximum,
    }, dtype);
    const vec_reduce_op: std.builtin.ReduceOp = comptime switch (op) {
        .Sum => .Add,
        .Max => .Max,
    };

    const storage_type = Storage(dtype);
    const _x: []align(storage_type.vec_alignment) dtype = x_ptr.storage.?.Zig.data;
    var _out: []align(storage_type.vec_alignment) dtype = out.storage.?.Zig.data;
    const vec_len: usize = storage_type.vec_len;
    const num_vecs: usize = comptime @divTrunc(size, vec_len);
    if (comptime dim == null) {
        var acc: dtype = @reduce(vec_reduce_op, @as(*const @Vector(vec_len, dtype), @alignCast(@ptrCast(_x[0..vec_len]))).*);
        for (1..num_vecs) |vec_i| {
            const offset = vec_i * vec_len;
            @prefetch(_x[offset + vec_len ..], .{ .locality = 0 });
            acc = @call(.always_inline, zip_fn, .{
                acc,
                @reduce(vec_reduce_op, @as(*const @Vector(vec_len, dtype), @alignCast(@ptrCast(_x[offset..][0..vec_len]))).*),
            });
        }
        inline for (comptime vec_len * num_vecs..size) |i| {
            acc = @call(.always_inline, zip_fn, .{ acc, _x[i] });
        }
        out.storage.?.Zig.data[0] = acc;
    } else {
        const stride = strides[dim.?];
        const dimsize = shape[dim.?];
        for (comptime 0..@TypeOf(out.*).size) |i| {
            const base = x_ptr.idxToPos(out.posToIdx(i));
            var acc = _x[base];
            for (comptime 1..dimsize) |j| {
                const k = base + j * stride;
                acc = @call(.always_inline, zip_fn, .{ acc, _x[k] });
            }
            _out[i] = acc;
        }
    }
}
