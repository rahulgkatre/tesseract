const GraphTensor = @import("graph.zig").GraphTensor;

pub const MapOp = enum { Neg };
pub const ZipOp = enum { Add };
pub const ReduceOp = enum { Sum };
pub const Op = union(enum) { MapOp: MapOp, ZipOp: ZipOp, ReduceOp: ReduceOp };

pub const OpArgs = union(enum) {
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