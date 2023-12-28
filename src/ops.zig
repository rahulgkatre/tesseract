pub const MapOp = enum { Neg };
pub const ZipOp = enum { Add };
pub const ReduceOp = enum { Sum };
pub const Op = union(enum) { MapOp: MapOp, ZipOp: ZipOp, ReduceOp: ReduceOp };
