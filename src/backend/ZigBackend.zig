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
        // TODO: Why does this cause an error?
        // global_arena = undefined;
    }
    fn allocator() std.mem.Allocator {
        return global_arena.allocator();
    }
};

pub fn Storage(comptime dtype: type) type {
    return struct {
        const Self = @This();
        data: ?[]dtype,
        size: usize,
        pub fn init(self: *Self) void {
            if (self.data == null) {
                self.data = GlobalArena.allocator().alloc(dtype, self.size) catch @panic("Unable to allocate tensor storage");
            }
        }
        pub fn fill(self: *Self, value: dtype) void {
            @memset(self.data.?, value);
        }
    };
}

// TODO: Replace page with a fixed buffer allocator
// Buffer size should be computed at compile time
pub fn init(_: *const ZigBackend, _: anytype) void {
    GlobalArena.init(std.heap.ArenaAllocator.init(std.heap.page_allocator));
}

pub fn storage(_: *const ZigBackend, comptime dtype: type, size: usize) Backend.Storage(dtype) {
    return .{
        .Zig = .{
            .data = null,
            .size = size,
        },
    };
}

pub fn deinit(_: *const ZigBackend) void {
    GlobalArena.deinit();
}

// TODO: What is the standardized way for determining the float type to cast an int type to
fn IntToFloatBySize(comptime int_type: type) type {
    const size = @sizeOf(int_type);
    if (size <= 2) {
        return f16;
    } else if (size <= 4) {
        return f32;
    } else if (size <= 8) {
        return f64;
    } else {
        return f128;
    }
}

fn ScalarMapFnReturnType(comptime map_op: ops.MapOp, comptime dtype: type) type {
    return @TypeOf(switch (map_op) {
        .Neg => struct {
            inline fn f(_: dtype) dtype {
                @compileError("Not a real function");
            }
        },
        else => struct {
            inline fn f(_: dtype) IntToFloatBySize(dtype) {
                @compileError("Not a real function");
            }
        },
    }.f);
}

fn ScalarMapFn(comptime map_op: ops.MapOp, comptime dtype: type) ScalarMapFnReturnType(map_op, dtype) {
    return comptime switch (map_op) {
        .Neg => switch (@typeInfo(dtype)) {
            .Bool => struct {
                inline fn negateBoolEval(x: dtype) dtype {
                    return !x;
                }
            }.negateBoolEval,
            else => struct {
                inline fn negateEval(x: dtype) dtype {
                    return -x;
                }
            }.negateEval,
        },
        .Log2 => struct {
            inline fn log2Eval(x: dtype) IntToFloatBySize(dtype) {
                return @log2(x);
            }
        }.log2Eval,
        .Exp2 => struct {
            inline fn exp2Eval(x: dtype) IntToFloatBySize(dtype) {
                return @exp2(x);
            }
        }.exp2Eval,
        .Sqrt => struct {
            inline fn sqrtEval(x: dtype) dtype {
                return @sqrt(x);
            }
        }.sqrtEval,
        .Recip => struct {
            inline fn recipEval(x: dtype) dtype {
                return 1.0 / x;
            }
        }.recipEval,
        else => @compileError("Not implemented"),
    };
}

fn ScalarZipFnReturnType(comptime zip_op: ops.ZipOp, comptime a_dtype: type, comptime b_dtype: type) type {
    return @TypeOf(switch (zip_op) {
        .Lt, .Eq => struct {
            inline fn f(_: a_dtype, _: a_dtype) bool {
                @compileError("Not a real function");
            }
        },
        .Xor => struct {
            inline fn f(_: a_dtype, _: a_dtype) @TypeOf(@as(a_dtype, 0) ^ @as(a_dtype, 0)) {
                @compileError("Not a real function");
            }
        },
        else => struct {
            inline fn f(_: a_dtype, _: b_dtype) @TypeOf(@as(a_dtype, 0) + @as(b_dtype, 0)) {
                @compileError("Not a real function");
            }
        },
    }.f);
}

fn ScalarZipFn(comptime zip_op: ops.ZipOp, comptime a_dtype: type, comptime b_dtype: type) ScalarZipFnReturnType(zip_op, a_dtype, b_dtype) {
    return comptime switch (zip_op) {
        .Add => struct {
            inline fn addEval(a: a_dtype, b: b_dtype) @TypeOf(@as(a_dtype, 0) + @as(b_dtype, 0)) {
                return a + b;
            }
        }.addEval,
        .Mul => struct {
            inline fn mulEval(a: a_dtype, b: b_dtype) @TypeOf(@as(a_dtype, 0) * @as(b_dtype, 0)) {
                return a * b;
            }
        }.mulEval,
        .Maximum => struct {
            // Cast to a shared type
            inline fn maximumEval(a: a_dtype, b: b_dtype) @TypeOf(@as(a_dtype, 0) * @as(b_dtype, 0)) {
                const cast_type = @TypeOf(@as(a_dtype, 0) * @as(a_dtype, 0));
                return @max(@as(cast_type, a), @as(cast_type, b));
            }
        }.maximumEval,
        .Lt => struct {
            comptime {
                if (a_dtype != b_dtype) {
                    @compileError("dtypes must match");
                }
            }
            const dtype = a_dtype;
            inline fn lessThanEval(a: dtype, b: dtype) bool {
                return a < b;
            }
        }.lessThanEval,
        .Eq => struct {
            comptime {
                if (a_dtype != b_dtype) {
                    @compileError("dtypes must match");
                }
            }
            const dtype = a_dtype;
            inline fn equalsEval(a: dtype, b: dtype) bool {
                return a == b;
            }
        }.equalsEval,
        .Xor => struct {
            comptime {
                if (a_dtype != b_dtype) {
                    @compileError("dtypes must match");
                }
            }
            const dtype = a_dtype;
            inline fn xorEval(a: dtype, b: dtype) @TypeOf(@as(dtype, 0) ^ @as(dtype, 0)) {
                return a ^ b;
            }
        }.xorEval,
        else => @compileError("Not implemented"),
    };
}

fn ScalarCastFnReturnType(comptime old_dtype: type, comptime new_dtype: type) type {
    return @TypeOf(struct {
        inline fn cast(_: old_dtype) new_dtype {
            unreachable;
        }
    }.cast);
}

fn ScalarCastFn(comptime old_dtype: type, comptime new_dtype: type) ScalarCastFnReturnType(old_dtype, new_dtype) {
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

pub fn asType(_: *const ZigBackend, comptime new_dtype: type, x: anytype, out: *tensor.CastedTensor(@TypeOf(x), new_dtype)) void {
    const old_dtype: type = @field(@TypeOf(x), "dtype");
    const castFn = ScalarCastFn(old_dtype, new_dtype);
    inline for (0..@field(@TypeOf(out.*), "size")) |flat_index| {
        out.storage.Zig.data.?[flat_index] = castFn(x.storage.Zig.data.?[flat_index]);
    }
}

pub fn map(_: *const ZigBackend, comptime op: ops.MapOp, x: anytype, out: *@TypeOf(x)) void {
    const mapFn = ScalarMapFn(op, @field(@TypeOf(x), "dtype"));
    inline for (0..@field(@TypeOf(out.*), "size")) |flat_index| {
        out.storage.Zig.data.?[flat_index] = mapFn(x.storage.Zig.data.?[flat_index]);
    }
}

pub fn zip(
    _: *const ZigBackend,
    comptime op: ops.ZipOp,
    a: anytype,
    b: anytype,
    out: *tensor.BroadcastedTensor(@TypeOf(a), @TypeOf(b)),
) void {
    const zipFn = ScalarZipFn(
        op,
        @field(@TypeOf(a), "dtype"),
        @field(@TypeOf(b), "dtype"),
    );
    inline for (0..@field(@TypeOf(out.*), "size")) |out_flat_index| {
        const out_index = out.unflattenIndex(out_flat_index);
        const a_index = a.broadcastIndex(out_index);
        const b_index = b.broadcastIndex(out_index);

        out.storage.Zig.data.?[out_flat_index] = zipFn(
            a.storage.Zig.data.?[a.flattenIndex(a_index)],
            b.storage.Zig.data.?[b.flattenIndex(b_index)],
        );
    }
}

pub fn reduce(
    _: *const ZigBackend,
    comptime op: ops.ReduceOp,
    x: anytype,
    comptime dim: ?u8,
    out: *tensor.ReducedTensor(@TypeOf(x), dim),
) void {
    const dtype: type = @field(@TypeOf(x), "dtype");
    const ndims: u8 = @field(@TypeOf(x), "ndims");
    const shape: [ndims]usize = @field(@TypeOf(x), "shape");
    const x_size: usize = @field(@TypeOf(x), "size");
    const out_size: usize = @field(@TypeOf(out.*), "size");

    if (ndims == 0 or (dim != null and shape[dim.?] == 0)) {
        @compileError("Cannot reduce over 0 elements");
    }
    if (dim == null) {
        const builtin_reduceop: std.builtin.ReduceOp = comptime switch (op) {
            .Sum => .Add,
            .Max => .Max,
        };
        // Use SIMD to reduce the entire data to a single value
        const data_vec: @Vector(x_size, dtype) = x.storage.Zig.data.?[0..x_size].*;
        out.storage.Zig.data.?[0] = @reduce(builtin_reduceop, data_vec);
    } else {
        const zipFn = ScalarZipFn(
            switch (op) {
                .Sum => .Add,
                .Max => .Maximum,
            },
            dtype,
            dtype,
        );

        var out_index: [ndims]usize = undefined;
        var x_start_flat_index: usize = undefined;
        var x_flat_index: usize = undefined;
        var acc: dtype = undefined;
        inline for (0..out_size) |out_flat_index| {
            out_index = out.unflattenIndex(out_flat_index);
            x_start_flat_index = x.flattenIndex(out_index);
            acc = x.storage.Zig.data.?[x_start_flat_index];
            inline for (1..shape[dim.?]) |i| {
                x_flat_index = x_start_flat_index + i * x.strides[dim.?];
                acc = zipFn(acc, x.storage.Zig.data.?[x_flat_index]);
            }
            out.storage.Zig.data.?[out_flat_index] = acc;
        }
    }
}
