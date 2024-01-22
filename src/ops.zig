// Map, Zip and Reduce ops are arithmetic operations for unary functions, binary functions,
// and reducing a dimension of a tensor to a single value by applying some binary function
pub const MapOp = enum { Neg, Log2, Exp2, Sqrt, Recip, Sin };
pub const ZipOp = enum { Add, Mul, Maximum, Mod, Lt, Eq, Xor };
pub const ReduceOp = enum { Sum, Max };

pub const OpTypes = enum { MapOp, ZipOp, ReduceOp };
pub const Op = union(OpTypes) { MapOp: MapOp, ZipOp: ZipOp, ReduceOp: ReduceOp };

// TODO: Optional: add a comptime hash map to convert op enums to symbols for graph visualiztion
