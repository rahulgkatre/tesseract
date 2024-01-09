const Allocator = @import("std").mem.Allocator;
const tensor = @import("../tensor.zig");
const ops = @import("../ops.zig");
const Backend = @import("backend.zig").Backend;
const ZigBackend = @This();

pub fn ZigStorage(comptime dtype: type) type {
    return struct {
        const Self = @This();
        data: []dtype,
        allocator: *const Allocator,
        pub fn init(size: usize, allocator: *const Allocator) !*Backend.Storage(dtype) {
            const data = try allocator.alloc(dtype, size);
            const storage = try allocator.create(Backend.Storage(dtype));
            storage.* = .{
                .Zig = .{
                    .data = data,
                    .allocator = allocator,
                },
            };
            return storage;
        }
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
        }
        pub fn fill(self: *Self, value: dtype) void {
            @memset(self.data, value);
        }
    };
}

allocator: ?*const Allocator = null,

pub fn init(self: *ZigBackend, args: anytype) void {
    self.allocator = args.allocator;
}

pub fn alloc(self: *const ZigBackend, comptime dtype: type, size: usize) !*Backend.Storage(dtype) {
    if (self.allocator != null) {
        return try ZigStorage(dtype).init(size, self.allocator.?);
    }
    @panic("No allocator provided");
}

pub fn deinitStorage(_: *const ZigBackend, storage: anytype) void {
    storage.deinit();
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

fn ScalarMapOpEvalFnReturnType(comptime map_op: ops.MapOp, comptime dtype: type) type {
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

fn ScalarMapOpEvalFn(comptime map_op: ops.MapOp, comptime dtype: type) ScalarMapOpEvalFnReturnType(map_op, dtype) {
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
                return @log2(@as(IntToFloatBySize(dtype), x));
            }
        }.log2Eval,
        .Exp2 => struct {
            inline fn exp2Eval(x: dtype) IntToFloatBySize(dtype) {
                return @exp2(@as(IntToFloatBySize(dtype), x));
            }
        }.exp2Eval,
        .Sqrt => struct {
            inline fn sqrtEval(x: dtype) IntToFloatBySize(dtype) {
                return @sqrt(@as(IntToFloatBySize(dtype), x));
            }
        }.sqrtEval,
        .Recip => struct {
            inline fn recipEval(x: dtype) IntToFloatBySize(dtype) {
                return 1.0 / @as(IntToFloatBySize(dtype), x);
            }
        }.recipEval,
        else => @compileError("Not implemented"),
    };
}

fn ScalarZipOpEvalFnReturnType(comptime zip_op: ops.ZipOp, comptime a_dtype: type, comptime b_dtype: type) type {
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

fn ScalarZipOpEvalFn(
    comptime zip_op: ops.ZipOp,
    comptime a_dtype: type,
    comptime b_dtype: type,
) ScalarZipOpEvalFnReturnType(
    zip_op,
    a_dtype,
    b_dtype,
) {
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

pub fn mapEval(_: *const ZigBackend, comptime op: ops.MapOp, x: anytype, out: *const @TypeOf(x)) void {
    const mapFn = ScalarMapOpEvalFn(
        op,
        @field(@TypeOf(x), "dtype"),
    );
    inline for (0..@field(@TypeOf(out.*), "size")) |flat_index| {
        out.storage.?.Zig.data[flat_index] = mapFn(x.storage.?.Zig.data[flat_index]);
    }
}

pub fn zipEval(
    _: *const ZigBackend,
    comptime op: ops.ZipOp,
    a: anytype,
    b: anytype,
    out: *tensor.BroadcastedTensor(@TypeOf(a), @TypeOf(b)),
) void {
    const zipFn = ScalarZipOpEvalFn(
        op,
        @field(@TypeOf(a), "dtype"),
        @field(@TypeOf(b), "dtype"),
    );
    inline for (0..@field(@TypeOf(out.*), "size")) |out_flat_index| {
        const out_index = out.unflattenIndex(out_flat_index);
        const a_index = a.broadcastIndex(out_index);
        const b_index = b.broadcastIndex(out_index);

        out.storage.?.Zig.data[out_flat_index] = zipFn(
            a.storage.?.Zig.data[a.flattenIndex(a_index)],
            b.storage.?.Zig.data[b.flattenIndex(b_index)],
        );
    }
}

const BuiltInReduceOp = @import("std").builtin.ReduceOp;

pub fn reduceEval(
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

    // TODO: Add checks for 0 dimensionality
    // Should @compileError when this is the case
    if (dim == null) {
        const builtin_reduceop: BuiltInReduceOp = comptime switch (op) {
            .Sum => .Add,
            .Max => .Max,
        };
        // Use SIMD to reduce the entire data to a single value
        const data_vec: @Vector(x_size, dtype) = x.storage.?.Zig.data[0..x_size].*;
        out.storage.?.Zig.data[0] = @reduce(builtin_reduceop, data_vec);
    } else {
        const zipFn = ScalarZipOpEvalFn(
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
            acc = x.storage.?.Zig.data[x_start_flat_index];
            inline for (1..shape[dim.?]) |i| {
                x_flat_index = x_start_flat_index + i * x.strides[dim.?];
                acc = zipFn(acc, x.storage.?.Zig.data[x_flat_index]);
            }
            out.storage.?.Zig.data[out_flat_index] = acc;
        }
    }
}
