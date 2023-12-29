const GraphTensor = @import("graph.zig").GraphTensor;

pub const MapOp = enum { Neg, Log2, Exp2, Sqrt, Recip, Cast };
pub const ZipOp = enum { Add, Mul, Maximum, Mod, Lt, Eq, Xor };
pub const ReduceOp = enum { Sum, Max };
pub const ShapeOp = enum { Reshape, Permute, Expand, Pad, Shrink, Stride, AsStrided };

pub const OpTypes = enum { MapOp, ZipOp, ReduceOp };
pub const Op = union(OpTypes) { MapOp: MapOp, ZipOp: ZipOp, ReduceOp: ReduceOp };

pub const OpArgs = union(OpTypes) {
    MapOp: struct {
        a_ptr: *const GraphTensor,
    },
    ZipOp: struct {
        a_ptr: *const GraphTensor,
        b_ptr: *const GraphTensor,
    },
    ReduceOp: struct {
        a_ptr: *const GraphTensor,
        reduce_dim: u8,
    },
};