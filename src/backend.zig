const std = @import("std");
const Allocator = std.mem.Allocator;
const LazyBuffer = @import("buffer.zig").LazyBuffer;
const ops = @import("ops.zig");
const comptimePrint = std.fmt.comptimePrint;
const tensor = @import("tensor.zig");

// TODO: Make a backend struct
// Backends provide APIS for performing various computations and management of buffers in memory
// Backends are not limited to CPU backend, they can run on other devices (GPU, TPU, etc.)
// In the case of the Zig CPU backend the buffer will be an anyopaque slice
pub const Backend = union(enum) {
    Zig: ZigBackend,
    // TODO: Other backends
    // ArrayFire: ArrayFireBackend
    // CUDA: CudaBackend
    // ...
    pub fn allocBuffer(self: *const Backend, comptime dtype: type, size: usize) !*LazyBuffer {
        return switch (self.*) {
            inline else => |*b| try b.allocBuffer(dtype, size),
        };
    }
    pub fn lazy_map(self: *const Backend, op: ops.MapOp, x: anytype) @TypeOf(x) {
        var out = @TypeOf(x).init(self);
        out.eval_fn = struct {
            fn eval(ptr: *const @TypeOf(out)) void {
                x.eval();
                if (!@inComptime()) {
                    std.debug.print("\n{s}@{d} = {any} {s}@{d}", .{ ptr.info(), @intFromPtr(ptr), op, x.info(), @intFromPtr(x) });
                } else {
                    @compileLog(comptimePrint("{s} = {any} {s}", .{ ptr.info(), op, x.info() }));
                }
                switch (self.*) {
                    inline else => |*eval_backend| eval_backend.eval_map(op, x, ptr),
                }
            }
        }.eval;
        return out;
    }
    pub fn lazy_zip(self: *const Backend, op: ops.ZipOp, a: anytype, b: anytype) tensor.BroadcastedTensor(@TypeOf(a), @TypeOf(b)) {
        var out = tensor.BroadcastedTensor(@TypeOf(a), @TypeOf(b)).init(self);
        out.eval_fn = struct {
            fn eval(ptr: *const @TypeOf(out)) void {
                a.eval();
                b.eval();
                if (!@inComptime()) {
                    std.debug.print("\n{s}@{d} = {any} {s}@{d} {s}@{d}", .{ ptr.info(), @intFromPtr(ptr), op, a.info(), @intFromPtr(&a), b.info(), @intFromPtr(&b) });
                } else {
                    @compileLog(comptimePrint("{s} = {any} {s} {s}", .{ ptr.info(), op, a.info(), b.info() }));
                }
                switch (self.*) {
                    inline else => |*eval_backend| eval_backend.eval_zip(op, a, b, ptr),
                }
            }
        }.eval;
        return out;
    }
    pub fn lazy_reduce(self: *const Backend, op: ops.ReduceOp, x: anytype, reduce_dim: u8) tensor.ReducedTensor(@TypeOf(x), reduce_dim) {
        var out = tensor.ReducedTensor(@TypeOf(x), reduce_dim).init(self);
        out.eval_fn = struct {
            fn eval(ptr: *const @TypeOf(out)) void {
                x.eval();
                if (!@inComptime()) {
                    std.debug.print("\n{s}@{d} = {any} {s}@{d} {d}", .{ ptr.info(), @intFromPtr(ptr), op, x.info(), @intFromPtr(&x), reduce_dim });
                } else {
                    @compileLog(comptimePrint("{s} = {any} {s} {d}", .{ ptr.info(), op, x.info(), reduce_dim }));
                }
                // TODO: Compute the start value for the accumulator based on the op, and the zip op used to accumulate
                // by switching on the reduce op
                // switch (self.*) {
                //     inline else => |*eval_backend| eval_backend.eval_reduce(op, zip_op x, reduce_dim, acc_start, ptr),
                // }
            }
        }.eval;
        return out;
    }
};

// TODO: Move this to its own file in a directory called backends
pub const ZigBackend = struct {
    const ZigLazyBuffer = @import("buffer.zig").ZigLazyBuffer;
    allocator: ?Allocator = null,

    pub fn eval_map(self: *const ZigBackend, op: ops.MapOp, x: anytype, out: *const @TypeOf(x)) void {
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

    pub fn eval_zip(self: *const ZigBackend, op: ops.ZipOp, a: anytype, b: anytype, out: *const tensor.BroadcastedTensor(@TypeOf(a), @TypeOf(b))) void {
        _ = op;
        _ = out;
        _ = self;
        // var out_index: [out.ndims]usize = undefined;
        // var a_index: [a.ndims]usize = undefined;
        // var b_index: [b.ndims]usize = undefined;
        // inline for (0..out.size) |out_flat_index| {
        //     out_index = out.unflattenIndex(out_flat_index);
        //     a_index = a.broadcastIndex(out.ndims, out_index);
        //     b_index = b.broadcastIndex(out.ndims, out.index);
        //     out.buffer.data[out_flat_index] = @call(.always_inline, op, .{ a.buffer.data[a.flattenIndex(a_index)], b.buffer.data[b.flattenIndex(b_index)] });
        // }
    }

    pub fn eval_reduce(self: *const Backend, op: ops.ReduceOp, zip_op: ops.ZipOp, x: anytype, reduce_dim: u8, acc_start: anytype, out: *const tensor.ReducedTensor(@TypeOf(x), reduce_dim)) void {
        _ = zip_op;
        _ = acc_start;
        _ = out;
        _ = self;
        _ = op;
        // var out_index: [out.ndims]usize = undefined;
        // var x_flat_index: usize = undefined;
        // var acc = undefined;
        // var x_start = undefined;
        // inline for (0..out.size) |out_flat_index| {
        //     out_index = out.unflattenIndex(out_flat_index);
        //     x_start = x.flattenIndex(out_index);
        //     acc = acc_start;
        //     for (0..x.shape[reduce_dim]) |i| {
        //         x_flat_index = x_start + i * x.strides[reduce_dim];
        //         acc = @call(.always_inline, zip_op, .{ acc, x.buffer.data[x_flat_index] });
        //     }
        //     out.buffer.data[out_flat_index] = acc;
        // }
    }

    pub fn allocBuffer(self: *const ZigBackend, comptime dtype: type, size: usize) !*LazyBuffer {
        if (self.allocator != null) {
            const zig_buffer = try ZigLazyBuffer(dtype).init(size, self.allocator.?);
            return &zig_buffer.lazy_buffer;
        }
        @panic("No allocator provided");
    }

    // pub fn freeBuffer(_: *const ZigBackend, buffer: *LazyBuffer(comptime dtype: type)) void {
    //     buffer.deinit();
    // }
};
