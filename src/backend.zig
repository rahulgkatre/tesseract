const std = @import("std");
const Allocator = std.mem.Allocator;
const LazyBuffer = @import("buffer.zig").Buffer;
const ops = @import("ops.zig");
const comptimePrint = std.fmt.comptimePrint;
const tensor = @import("tensor.zig");

pub const BackendTypes = enum {
    Zig,
};

pub const Backend = union(BackendTypes) {
    Zig: ZigBackend,
    // TODO: Other backends
    // ArrayFire: ArrayFireBackend
    // CUDA: CudaBackend
    // ...
    pub fn init(self: *Backend, args: anytype) void {
        return switch (self.*) {
            inline else => |*b| b.init(args),
        };
    }
    pub fn allocBuffer(self: *const Backend, comptime dtype: type, size: usize) !*LazyBuffer(dtype) {
        return switch (self.*) {
            inline else => |*b| try b.allocBuffer(dtype, size),
        };
    }
    pub fn mapLazy(self: *const Backend, op: ops.MapOp, x: anytype) @TypeOf(x) {
        var out = @TypeOf(x).init(self);
        out.eval_fn = struct {
            fn eval(ptr: *const @TypeOf(out)) void {
                x.eval();
                if (!@inComptime()) {
                    std.debug.print("\n{s}@{d} = {any} {s}@{d}", .{ ptr.str, @intFromPtr(ptr), op, x.str, @intFromPtr(x) });
                } else {
                    @compileLog(comptimePrint("{s} = {any} {s}", .{ ptr.str, op, x.str }));
                }
                switch (self.*) {
                    inline else => |*eval_backend| eval_backend.mapEval(op, x, ptr),
                }
            }
        }.eval;
        return out;
    }
    pub fn zipLazy(self: *const Backend, op: ops.ZipOp, a: anytype, b: anytype) tensor.BroadcastedTensor(@TypeOf(a), @TypeOf(b)) {
        var out = tensor.BroadcastedTensor(@TypeOf(a), @TypeOf(b)).init(self);
        out.eval_fn = struct {
            fn eval(ptr: *const @TypeOf(out)) void {
                a.eval();
                b.eval();
                if (!@inComptime()) {
                    std.debug.print("\n{s}@{d} = {any} {s}@{d} {s}@{d}", .{ ptr.str, @intFromPtr(ptr), op, a.str, @intFromPtr(&a), b.str, @intFromPtr(&b) });
                } else {
                    @compileLog(comptimePrint("{s} = {any} {s} {s}", .{ ptr.str, op, a.str, b.str }));
                }
                switch (self.*) {
                    inline else => |*eval_backend| eval_backend.zipEval(op, a, b, ptr),
                }
            }
        }.eval;
        return out;
    }
    pub fn reduceLazy(self: *const Backend, op: ops.ReduceOp, x: anytype, reduce_dim: u8) tensor.ReducedTensor(@TypeOf(x), reduce_dim) {
        var out = tensor.ReducedTensor(@TypeOf(x), reduce_dim).init(self);
        out.eval_fn = struct {
            fn eval(ptr: *const @TypeOf(out)) void {
                x.eval();
                if (!@inComptime()) {
                    std.debug.print("\n{s}@{d} = {any} {s}@{d} {d}", .{ ptr.str, @intFromPtr(ptr), op, x.str, @intFromPtr(&x), reduce_dim });
                } else {
                    @compileLog(comptimePrint("{s} = {any} {s} {d}", .{ ptr.str, op, x.str, reduce_dim }));
                }
                // TODO: Compute the start value for the accumulator based on the op, and the zip op used to accumulate
                // by switching on the reduce op
                // switch (self.*) {
                //     inline else => |*eval_backend| eval_backend.reduceEval(op, zip_op, x, reduce_dim, acc_start, ptr),
                // }
            }
        }.eval;
        return out;
    }
};

// TODO: Move this to its own file in a directory called backends
pub const ZigBackend = struct {
    const ZigLazyBuffer = @import("buffer.zig").ZigBuffer;
    allocator: ?*const Allocator = null,

    pub fn init(self: *ZigBackend, args: anytype) void {
        self.allocator = args.allocator;
    }

    pub fn mapEval(self: *const ZigBackend, op: ops.MapOp, x: anytype, out: *const @TypeOf(x)) void {
        _ = op;
        _ = out;
        _ = self;
        // TODO: Iterate over each of the output elements and compute op(x)
        // Something like this:
        // Also in minitorch, map can broadcast, which I'm not exactly sure why because its 1-1 anyways
        // inline for (0..out.size) |flat_index| {
        //     out.buffer.data[flat_index] = @call(.always_inline, op, .{x[flat_index]});
        // }
    }

    pub fn zipEval(self: *const ZigBackend, op: ops.ZipOp, a: anytype, b: anytype, out: *const tensor.BroadcastedTensor(@TypeOf(a), @TypeOf(b))) void {
        _ = op;
        _ = out;
        _ = self;
        // inline for (0..out.size) |out_flat_index| {
        //     const out_index = out.unflattenIndex(out_flat_index);
        //     const a_index = a.broadcastIndex(out.ndims, out_index);
        //     const b_index = b.broadcastIndex(out.ndims, out.index);
        //     out.buffer.data[out_flat_index] = @call(.always_inline, op, .{ a.buffer.data[a.flattenIndex(a_index)], b.buffer.data[b.flattenIndex(b_index)] });
        // }
    }

    pub fn reduceEval(self: *const Backend, op: ops.ReduceOp, zip_op: ops.ZipOp, x: anytype, reduce_dim: u8, acc_start: anytype, out: *const tensor.ReducedTensor(@TypeOf(x), reduce_dim)) void {
        _ = zip_op;
        _ = acc_start;
        _ = out;
        _ = self;
        _ = op;
        // inline for (0..out.size) |out_flat_index| {
        //     const out_index = out.unflattenIndex(out_flat_index);
        //     const x_start = x.flattenIndex(out_index);
        //     var acc = acc_start;
        //     for (0..x.shape[reduce_dim]) |i| {
        //         x_flat_index = x_start + i * x.strides[reduce_dim];
        //         acc = @call(.always_inline, zip_op, .{ acc, x.buffer.data[x_flat_index] });
        //     }
        //     out.buffer.data[out_flat_index] = acc;
        // }
    }

    pub fn allocBuffer(self: *const ZigBackend, comptime dtype: type, size: usize) !*LazyBuffer(dtype) {
        if (self.allocator != null) {
            return try ZigLazyBuffer(dtype).init(size, self.allocator.?);
        }
        @panic("No allocator provided");
    }

    // pub fn freeBuffer(_: *const ZigBackend, buffer: *LazyBuffer(comptime dtype: type)) void {
    //     buffer.deinit();
    // }
};
