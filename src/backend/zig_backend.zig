const Allocator = @import("std").mem.Allocator;
const tensor = @import("../tensor.zig");
const ops = @import("../ops.zig");
const Backend = @import("backend.zig").Backend;

pub const ZigBackend = struct {
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

    fn ScalarMapOpReturnType(comptime map_op: ops.MapOp, comptime x: anytype) type {
        return switch (map_op) {
            .Neg => @TypeOf(x), // Neg can apply to any numeric type (or boolean)
            else => @TypeOf(x + 0.0), // Other
        };
    }

    fn scalarMapOpEval(comptime map_op: ops.MapOp, x: anytype) ScalarMapOpReturnType(map_op, x) {
        return comptime switch (map_op) {
            .Neg => if (@typeInfo(@TypeOf(x)) == .Bool) !x else -x,
            .Log2 => @log2(x + 0.0),
            .Exp2 => @exp2(x + 0.0),
            .Sqrt => @sqrt(x + 0.0),
            .Recip => @divExact(1.0, x + 0.0),
        };
    }

    fn ScalarZipOpReturnType(comptime zip_op: ops.ZipOp, comptime a: anytype, comptime b: anytype) type {
        return switch (zip_op) {
            .Lt, .Eq => bool,
            .Xor => @TypeOf(a ^ b),
            else => @TypeOf(a + b),
        };
    }

    fn scalarZipOpEval(comptime zip_op: ops.ZipOp, a: anytype, b: anytype) ScalarZipOpReturnType(zip_op, a, b) {
        return comptime switch (zip_op) {
            .Add => a + b,
            .Mul => a * b,
            .Maximum => @max(a, b),
            .Lt => a < b,
            .Eq => a == b,
            .Xor => a ^ b,
            else => @panic("Not implemented"),
        };
    }

    fn ScalarOpReturnType(comptime op: ops.Op) type {
        return @TypeOf(switch (op) {
            .MapOp => |map_op| struct {
                inline fn f(x: anytype) ScalarMapOpReturnType(map_op, x) {
                    return comptime scalarMapOpEval(map_op, x);
                }
            },
            .ZipOp => |zip_op| struct {
                inline fn f(a: anytype, b: anytype) ScalarZipOpReturnType(zip_op, a, b) {
                    return comptime scalarZipOpEval(zip_op, a, b);
                }
            },
            else => @panic("Not implemented"),
        }.f);
    }

    pub fn scalarOpEval(comptime op: ops.Op) ScalarOpReturnType(op) {
        return comptime switch (op) {
            .MapOp => |map_op| struct {
                inline fn f(x: anytype) ScalarMapOpReturnType(map_op, x) {
                    return comptime scalarMapOpEval(map_op, x);
                }
            },
            .ZipOp => |zip_op| struct {
                inline fn f(a: anytype, b: anytype) ScalarZipOpReturnType(zip_op, a, b) {
                    return comptime scalarZipOpEval(zip_op, a, b);
                }
            },
            else => @panic("Not implemented"),
        }.f;
    }

    pub fn init(self: *ZigBackend, args: struct { allocator: *const Allocator }) void {
        self.allocator = args.allocator;
    }

    pub fn mapEval(self: *const ZigBackend, comptime op: ops.MapOp, x: anytype, out: *const @TypeOf(x)) void {
        _ = op;
        _ = out;

        _ = self;
        // inline for (0..@field(@TypeOf(out.*), "size")) |flat_index| {
        //     out.storage.?.Zig.data[flat_index] = @call(.always_inline, scalarOpEval(.{ .MapOp = op }), .{x.storage.?.Zig.data[flat_index]});
        // }
    }

    pub fn zipEval(self: *const ZigBackend, comptime op: ops.ZipOp, a: anytype, b: anytype, out: *const tensor.BroadcastedTensor(@TypeOf(a), @TypeOf(b))) void {
        _ = op;
        _ = out;
        _ = self;
        // inline for (0..@field(@TypeOf(out.*), "size")) |out_flat_index| {
        //     const out_index = @TypeOf(out.*).unflattenIndex(out_flat_index);
        //     const a_index = @TypeOf(a).broadcastIndex(out_index);
        //     const b_index = @TypeOf(b).broadcastIndex(out_index);
        //     out.storage.?.Zig.data[out_flat_index] = @call(.always_inline, scalarOpEval(.{ .ZipOp = op }), .{
        //         a.storage.?.Zig.data[@TypeOf(a).flattenIndex(a_index)],
        //         b.storage.?.Zig.data[@TypeOf(b).flattenIndex(b_index)],
        //     });
        // }
    }

    pub fn reduceEval(self: *const Backend, op: ops.ReduceOp, zip_op: ops.ZipOp, x: anytype, dim: ?u8, acc_start: anytype, out: *const tensor.ReducedTensor(@TypeOf(x), dim)) void {
        _ = zip_op;
        _ = acc_start;
        _ = out;
        _ = self;
        _ = op;
        // if (dim == null) {
        //     reduce across the entire input
        // }
        // inline for (0..out.size) |out_flat_index| {
        //     const out_index = out.unflattenIndex(out_flat_index);
        //     const x_start = x.flattenIndex(out_index);
        //     var acc = acc_start;
        //     for (0..x.shape[dim]) |i| {
        //         x_flat_index = x_start + i * x.strides[dim];
        //         acc = @call(.always_inline, zip_op, .{ acc, x.storage.data[x_flat_index] });
        //     }
        //     out.storage.data[out_flat_index] = acc;
        // }
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
};
