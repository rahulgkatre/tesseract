pub const MapOp = enum { Neg, Log2, Exp2, Sqrt, Recip, Cast };
pub const ZipOp = enum { Add, Mul, Maximum, Mod, Lt, Eq, Xor };
pub const ReduceOp = enum { Sum, Max };
pub const ShapeOp = enum { Reshape, Permute, Expand, Pad, Shrink, Stride, AsStrided };

pub const OpKinds = enum { MapOp, ZipOp, ReduceOp };
// pub const OpCall = union(OpKinds) {
//     MapOp: struct {
//         op: MapOp,
//         a: *const GraphTensor,
//     },
//     ZipOp: struct {
//         op: ZipOp,
//         a: *const GraphTensor,
//         b: *const GraphTensor,
//     },
//     ReduceOp: struct {
//         op: ReduceOp,
//         a: *const GraphTensor,
//         reduce_dim: u8
//     },
// };
