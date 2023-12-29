const GraphTensor = @import("graph.zig").GraphTensor;

pub const MapOp = enum { Neg };
pub const ZipOp = enum { Add };
pub const ReduceOp = enum { Sum };

pub const OpTypes = enum { MapOp, ZipOp, ReduceOp };
pub const Op = union(OpTypes) { MapOp: MapOp, ZipOp: ZipOp, ReduceOp: ReduceOp };

pub const OpArgs = union(OpTypes) {
    MapOp: struct {
        self_ptr: *const GraphTensor,
    },
    ZipOp: struct {
        self_ptr: *const GraphTensor,
        other_ptr: *const GraphTensor,
    },
    ReduceOp: struct {
        self_ptr: *const GraphTensor,
        reduce_dim: u8,
    },
};
