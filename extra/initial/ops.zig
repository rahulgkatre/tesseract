const Buffer = @import("buffer.zig").Buffer;
const utils = @import("./tensor_utils.zig");

const std = @import("std");

// const Ops = struct {
//     pub inline fn sigmoid(x: anytype) @TypeOf(x) {
//         return if (x >= 0.0) @divExact(1.0, 1.0 + @exp2(-x)) else @divExact(@exp2(x), (1.0 + @exp2(x)));
//     }

//     pub inline fn sigmoid_back(x: anytype, grad: anytype) @TypeOf(grad) {
//         return sigmoid(x) * (1 - sigmoid(x)) * grad;
//     }

//     pub inline fn relu(x: anytype) @TypeOf(x) {
//         return if (x >= 0.0) x else 0.0;
//     }

//     pub inline fn relu_back(x: anytype, grad: anytype) @TypeOf(grad) {
//         return if (x >= 0.0) grad else 0.0;
//     }

//     const EPSILON = 1e-6;

//     pub inline fn log(x: anytype) @TypeOf(x) {
//         return @log2(x + EPSILON);
//     }

//     pub inline fn log_back(x: anytype, grad: anytype) @TypeOf(grad) {
//         return @divExact(grad, if (x == 0.0) x + EPSILON else x);
//     }

//     pub inline fn inv(x: anytype) @TypeOf(x) {
//         return @divExact(1.0, if (x == 0.0) x + EPSILON else x);
//     }

//     pub inline fn inv_back(x: anytype, grad: anytype) @TypeOf(grad) {
//         return -grad * inv(x) * inv(x);
//     }
// };

pub const MapOp = enum { Neg, Log2, Exp2, Recip, Sin };
pub const ZipOp = enum { Add, Mul, Maximum, Mod, Cmp };
pub const ReduceOp = enum { Sum, Max };
pub const TernaryOp = enum { MulAcc, Where };

pub const MapFn = struct {
    const num_inputs = 1;
    op: MapOp,
    forward: *anyopaque,
    backward: *anyopaque,
};

pub const ZipFn = struct {
    const num_inputs = 2;
    op: ZipOp,
    forward: *anyopaque,
    backward: *anyopaque,
};

pub const ReduceFn = struct {
    const num_inputs = 1;
    op: ReduceOp,
    dim: u8,
    forward: *anyopaque,
    backward: *anyopaque,
};

pub const TernaryFn = struct {
    const num_inputs = 3;
    op: TernaryOp,
    forward: *anyopaque,
    backward: *anyopaque,
};

pub const Function = enum { MapFn, ZipFn, ReduceFn, TernaryFn };

// fn map(
//     comptime T: type,
//     unary_fn: anytype,
//     in: Buffer(T),
//     out: Buffer(T),
// ) void {
//     comptime {
//         _ = try utils.broadcastShape(in.shape, out.shape);
//     }
//     var in_pos: usize = undefined;
//     var in_idx = try out.allocator.alloc(usize, in.ndims);
//     defer in.allocator.free(in_idx);
//     var out_idx = try out.allocator.alloc(usize, out.ndims);
//     defer out.allocator.free(out_idx);
//     for (out.data, 0..out.data.len) |out_elem, out_pos| {
//         try out.pos2idx(out_pos, &out_idx);
//         try utils.broadcastIndex(out.shape, out_idx, in.shape, &in_idx);
//         in_pos = try in.idx2pos(in_idx);
//         out_elem = @call(.always_inline, unary_fn, .{in.data[in_pos]});
//     }
// }

// fn zip(
//     comptime T: type,
//     binary_fn: anytype,
//     a: Buffer(T),
//     b: Buffer(T),
//     out: Buffer(T),
// ) void {
//     comptime {
//         _ = try utils.broadcastShape(a.shape, out.shape);
//         _ = try utils.broadcastShape(b.shape, out.shape);
//     }
//     var a_pos: usize = 0;
//     var a_idx = try a.allocator.alloc(usize, a.ndims);
//     defer a.allocator.free(a_idx);

//     var b_pos: usize = 0;
//     var b_idx = try b.allocator.alloc(usize, b.ndims);
//     defer b.allocator.free(b_idx);

//     var out_idx = try out.allocator.alloc(usize, out.ndims);
//     defer out.allocator.free(out_idx);

//     for (out.data, 0..out.data.len) |out_elem, out_pos| {
//         try out.pos2idx(out_pos, &out_idx);
//         try utils.broadcastIndex(out.shape, out_idx, a.shape, &a_idx);
//         try utils.broadcastIndex(out.shape, out_idx, b.shape, &b_idx);
//         a_pos = try a.idx2pos(a_idx, a.strides);
//         b_pos = try b.idx2pos(b_idx, b.strides);
//         out_elem = @call(.always_inline, binary_fn, .{ a.data[a_pos], b.data[b_pos] });
//     }
// }

// fn reduce(
//     comptime T: type,
//     reduce_fn: anytype,
//     reduce_dim: usize,
//     reduce_start: T,
//     in: Buffer(T),
//     out: Buffer(T),
// ) void {
//     comptime {
//         _ = try utils.broadcastShape(in.shape, out.shape);
//     }
//     var in_pos: usize = undefined;
//     var out_idx = try out.allocator.alloc(usize, out.ndims);
//     defer out.allocator.free(out_idx);

//     var acc = reduce_start;
//     for (out.data, 0..out.data.len) |out_elem, out_pos| {
//         try out.pos2idx(out_pos, &out_idx);
//         in_pos = in.idx2pos(out_idx);
//         acc = reduce_start;
//         for (0..in.shape[reduce_dim]) |_| {
//             acc = @call(.always_inline, reduce_fn, .{ in.data[in_pos], acc });
//             in_pos += in.strides[reduce_dim];
//         }
//         out_elem = acc;
//     }
// }
