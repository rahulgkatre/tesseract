// Map, Zip and Reduce ops are arithmetic operations for unary functions, binary functions,
// and reducing a dimension of a tensor to a single value by applying some binary function
pub const MapOp = enum { Neg, Log2, Exp2, Sqrt, Recip, Sin };
pub const ZipOp = enum { Add, Mul, Maximum, Mod, Lt, Eq, Xor };
pub const ReduceOp = enum { Sum, Max };
// TypeOps affect the type of the tensor because they modify dtype, shape, or stride
// pub const TypeOp = enum { Reshape, Permute, Expand, Pad, Shrink, Stride, AsStrided, AsType };
pub const TypeOp = enum { Permute, View };
pub const OpTypes = enum { MapOp, ZipOp, ReduceOp, TypeOp };
pub const Op = union(OpTypes) { MapOp: MapOp, ZipOp: ZipOp, ReduceOp: ReduceOp, TypeOp: TypeOp };
