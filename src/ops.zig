pub const MapOp = enum { Neg, Log2, Exp2, Sqrt, Recip, Cast };
pub const ZipOp = enum { Add, Mul, Maximum, Mod, Lt, Eq, Xor };
pub const ReduceOp = enum { Sum, Max };
pub const ShapeOp = enum { Reshape, Permute, Expand, Pad, Shrink, Stride, AsStrided };

pub const OpTypes = enum { MapOp, ZipOp, ReduceOp, ShapeOp };
