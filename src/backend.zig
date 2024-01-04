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
    // pub fn allocBuffer(self: *const Backend, comptime dtype: type, size: usize) !*LazyBuffer {
    //     return switch (self.*) {
    //         inline else => |*b| try b.allocBuffer(dtype, size),
    //     };
    // }
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
                    std.debug.print("\n{s}@{d} = {any} {s}@{d} {d}", .{
                        ptr.info(),
                        @intFromPtr(ptr),
                        op,
                        x.info(),
                        @intFromPtr(&x),
                        reduce_dim,
                    });
                } else {
                    @compileLog(comptimePrint("{s} = {any} {s} {d}", .{
                        ptr.info(),
                        op,
                        x.info(),
                        reduce_dim,
                    }));
                }
            }
        }.eval;
        return out;
    }
};

pub const ZigBackend = struct {
    const ZigLazyBuffer = @import("buffer.zig").ZigLazyBuffer;
    allocator: ?Allocator = null,

    // pub fn allocBuffer(self: *const ZigBackend, comptime dtype: type, size: usize) !*LazyBuffer {
    //     if (self.allocator != null) {
    //         const zigBuffer = try ZigLazyBuffer(dtype).init(size, self.allocator.?);
    //         return &zigBuffer.graph_buffer;
    //     }
    //     @panic("No allocator provided");
    // }

    // pub fn freeBuffer(_: *const ZigBackend, buffer: *LazyBuffer(comptime dtype: type)) void {
    //     buffer.deinit();
    // }
};

// eval_map_fn: *const fn (ptr: *const Self, op_call: ops.OpCall) void,
// eval_zip_fn: *const fn (ptr: *const Self, op_call: ops.OpCall) void,
// eval_reduce_fn: *const fn (ptr: *const Self, op_call: ops.OpCall) void,
// pub fn eval_map(self: *const Self, op_call: ops.OpCall) void {
//     return self.eval_map_fn(self, op_call);
// }
// pub fn eval_zip(self: *const Self, op_call: ops.OpCall) void {
//     return self.eval_zip_fn(self, op_call);
// }
// pub fn eval_reduce(self: *const Self, op_call: ops.OpCall) void {
//     return self.eval_reduce_fn(self, op_call);
// }

// TODO: Backends should also implement functions for running the compute graph
// fn eval_map(self: *const Self, op_call: ops.OpCall) void {
//     switch (op_call) {
//         .MapOp => |map_op_call| {
//             // TODO: If the self Tensor is realized, execute the operation on its data using the args provided by the op_call
//             // Add any extra args as necessary (e.g. output location)
//             _ = map_op_call;
//         },
//         else => @panic("Invalid map op call"),
//     }
//     _ = self;
// }
// fn eval_zip(self: *const Self, op_call: ops.OpCall) void {
//     switch (op_call) {
//         .ZipOp => |zip_op_call| {
//             // TODO: If the self Tensor is realized, execute the operation on its data using the args provided by the op_call
//             // Add any extra args as necessary (e.g. output location)
//             _ = zip_op_call;
//         },
//         else => @panic("Invalid zip op call"),
//     }
//     _ = self;
// }
// fn eval_reduce(self: *const Self, op_call: ops.OpCall) void {
//     _ = self;
//     switch (op_call) {
//         .ReduceOp => |reduce_op_call| {
//             // TODO: If the self Tensor is realized, execute the operation on its data using the args provided by the op_call
//             // Add any extra args as necessary (e.g. output location)
//             _ = reduce_op_call;
//         },
//         else => @panic("Invalid reduce op call"),
//     }
// }
