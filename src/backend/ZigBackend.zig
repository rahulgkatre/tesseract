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
    };
}

allocator: ?*const Allocator = null,

pub fn init(self: *ZigBackend, args: struct { allocator: *const Allocator }) void {
    self.allocator = args.allocator;
}

fn ScalarMapOpReturnType(comptime map_op: ops.MapOp, comptime dtype: type) type {
    return switch (map_op) {
        .Neg => dtype, // Neg can apply to any numeric type (or boolean)
        else => t: {
            const size = @sizeOf(dtype);
            if (size <= 2) {
                break :t f16;
            } else if (size <= 4) {
                break :t f32;
            } else if (size <= 8) {
                break :t f64;
            } else {
                break :t f128;
            }
        },
    };
}

fn scalarMapOpEvalFnReturnType(comptime map_op: ops.MapOp, comptime dtype: type) type {
    return @TypeOf(struct {
        inline fn f(x: dtype) ScalarMapOpReturnType(map_op, dtype) {
            _ = x;

            @panic("Not a real function");
        }
    }.f);
}

fn scalarMapOpEvalFn(comptime map_op: ops.MapOp, comptime dtype: type) scalarMapOpEvalFnReturnType(map_op, dtype) {
    return struct {
        inline fn f(x: dtype) ScalarMapOpReturnType(map_op, dtype) {
            return switch (map_op) {
                .Neg => if (@typeInfo(@TypeOf(x)) == .Bool) !x else -x,
                .Log2 => @log2(x),
                .Exp2 => @exp2(x),
                .Sqrt => @sqrt(x),
                .Recip => if (@typeInfo(@TypeOf(x)) == .Float) 1.0 / x else @divTrunc(1, x),
                else => @panic("Not implemented"),
            };
        }
    }.f;
}

fn ScalarZipOpReturnType(comptime zip_op: ops.ZipOp, comptime a_dtype: type, comptime b_dtype: type) type {
    return switch (zip_op) {
        .Lt, .Eq => bool,
        .Xor => @TypeOf(@as(a_dtype, 0) ^ @as(b_dtype, 0)),
        else => @TypeOf(@as(a_dtype, 0) * @as(a_dtype, 0)),
    };
}

fn ScalarZipOpFnReturnType(comptime zip_op: ops.ZipOp, comptime a_dtype: type, comptime b_dtype: type) type {
    return @TypeOf(struct {
        inline fn f(_: a_dtype, _: b_dtype) ScalarZipOpReturnType(zip_op, a_dtype, b_dtype) {
            @panic("Not a real function");
        }
    }.f);
}

fn scalarZipOpEvalFn(comptime zip_op: ops.ZipOp, comptime a_dtype: type, comptime b_dtype: type) ScalarZipOpFnReturnType(zip_op, a_dtype, b_dtype) {
    return struct {
        inline fn f(a: a_dtype, b: b_dtype) ScalarZipOpReturnType(zip_op, a_dtype, b_dtype) {
            return switch (zip_op) {
                .Add => a + b,
                .Mul => a * b,
                .Maximum => @max(a, b),
                .Lt => a < b,
                .Eq => a == b,
                .Xor => a ^ b,
                else => @panic("Not implemented"),
            };
        }
    }.f;
}

pub fn mapEval(self: *const ZigBackend, comptime op: ops.MapOp, x: anytype, out: *const @TypeOf(x)) void {
    _ = self;
    const map_op_eval_fn = scalarMapOpEvalFn(op, @field(@TypeOf(x), "dtype"));
    inline for (0..@field(@TypeOf(out.*), "size")) |flat_index| {
        out.storage.?.Zig.data[flat_index] = map_op_eval_fn(x.storage.?.Zig.data[flat_index]);
    }
}

pub fn zipEval(self: *const ZigBackend, comptime op: ops.ZipOp, a: anytype, b: anytype, out: *tensor.BroadcastedTensor(@TypeOf(a), @TypeOf(b))) void {
    _ = self;
    const zip_op_eval_fn = scalarZipOpEvalFn(op, @field(@TypeOf(a), "dtype"), @field(@TypeOf(b), "dtype"));
    inline for (0..@field(@TypeOf(out.*), "size")) |out_flat_index| {
        const out_index = out.unflattenIndex(out_flat_index);
        const a_index = a.broadcastIndex(out_index);
        const b_index = b.broadcastIndex(out_index);

        out.storage.?.Zig.data[out_flat_index] = zip_op_eval_fn(
            a.storage.?.Zig.data[a.flattenIndex(a_index)],
            b.storage.?.Zig.data[b.flattenIndex(b_index)],
        );
    }
}

const BuiltInReduceOp = @import("std").builtin.ReduceOp;

pub fn reduceEval(self: *const ZigBackend, comptime op: ops.ReduceOp, x: anytype, comptime dim: ?u8, out: *tensor.ReducedTensor(@TypeOf(x), dim)) void {
    _ = self;
    if (dim == null) {
        const builtin_reduceop: BuiltInReduceOp = comptime switch (op) {
            .Sum => .Add,
            .Max => .Max,
        };
        const data_vec: @Vector(@field(@TypeOf(x), "size"), @field(@TypeOf(x), "dtype")) = x.storage.?.Zig.data[0..@field(@TypeOf(x), "size")].*;
        out.storage.?.Zig.data[0] = @reduce(builtin_reduceop, data_vec);
    } else {
        const zip_op_eval_fn = scalarZipOpEvalFn(
            switch (op) {
                .Sum => .Add,
                .Max => .Maximum,
            },
            @field(@TypeOf(x), "dtype"),
            @field(@TypeOf(x), "dtype"),
        );
        inline for (0..@field(@TypeOf(out.*), "size")) |out_flat_index| {
            const out_index = out.unflattenIndex(out_flat_index);
            const x_start = x.flattenIndex(out_index);
            var acc = x.storage.?.Zig.data[x_start];
            var x_flat_index: usize = undefined;
            for (1..x.shape[dim.?]) |i| {
                x_flat_index = x_start + i * x.strides[dim.?];
                acc = zip_op_eval_fn(acc, x.storage.?.Zig.data[x_flat_index]);
            }
            out.storage.?.Zig.data[out_flat_index] = acc;
        }
    }
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
