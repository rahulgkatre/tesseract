// Map, Zip and Reduce ops are arithmetic operations for unary functions, binary functions,
// and reducing a dimension of a tensor to a single value by applying some binary function
pub const MapOp = enum { Neg, Log2, Exp2, Sqrt, Recip, Sin };
pub const ZipOp = enum { Add, Mul, Maximum, Mod, LessThan, Equals, Xor };
pub const ReduceOp = enum { Sum, Max };
pub const TypeOp = enum { AsType, AsStrided, View, Permute };
pub const InitOp = enum { Input, From, Full, Rand };

pub const OpTypes = enum {
    MapOp,
    ZipOp,
    ReduceOp,
    TypeOp,
    InitOp,
};
pub const Op = union(enum) {
    MapOp: MapOp,
    ZipOp: ZipOp,
    ReduceOp: ReduceOp,
    TypeOp: TypeOp,
    InitOp: InitOp,
};
