const std = @import("std");
const tensor = @import("tensor.zig");
const dtypes = @import("dtypes.zig");

const TensorType = tensor.TensorType;
const isTensor = tensor.isTensor;
const IntTensor = tensor.IntTensor;
const BoolTensor = tensor.BoolTensor;
const FloatTensor = tensor.FloatTensor;

pub fn Functions(comptime T: type) type {
    comptime std.debug.assert(isTensor(T));
    const Tensor = tensor.Tensor(T.dtype, T.ndims, T.shape);
    return struct {
        // MapOps
        pub fn exp2(comptime a: Tensor) FloatTensor(Tensor) {
            return a.unaryFn(.Exp2);
        }
        pub fn log2(comptime a: Tensor) FloatTensor(Tensor) {
            return a.unaryFn(.Log2);
        }
        pub fn neg(comptime a: Tensor) Tensor {
            return a.unaryFn(.Neg);
        }
        pub fn recip(comptime a: Tensor) FloatTensor(Tensor) {
            return a.unaryFn(.Rcp);
        }
        pub fn sin(comptime a: Tensor) FloatTensor(Tensor) {
            return a.unaryFn(.Sin);
        }
        pub fn sqrt(comptime a: Tensor) FloatTensor(Tensor) {
            return a.unaryFn(.Sqrt);
        }
        pub fn id(comptime a: Tensor) Tensor {
            return a.unaryFn(.Id);
        }
        // ZipOps
        pub fn add(comptime a: Tensor, comptime b: anytype) Tensor.BinaryFnResult(TensorType(b), .Add) {
            return a.binaryFn(b, .Add);
        }
        pub fn mul(comptime a: Tensor, comptime b: anytype) Tensor.BinaryFnResult(TensorType(b), .Mul) {
            return a.binaryFn(b, .Mul);
        }
        pub fn maximum(comptime a: Tensor, comptime b: anytype) Tensor.BinaryFnResult(TensorType(b), .Max) {
            return a.binaryFn(b, .Max);
        }
        pub fn mod(comptime a: IntTensor(Tensor), comptime b: anytype) IntTensor(Tensor.Broadcast(TensorType(b).shape)) {
            return a.binaryFn(b, .Mod);
        }
        pub fn lessThan(comptime a: Tensor, comptime b: anytype) BoolTensor(Tensor.Broadcast(TensorType(b).shape)) {
            return a.binaryFn(b, .Lt);
        }
        pub fn equals(comptime a: BoolTensor(Tensor), comptime b: anytype) BoolTensor(Tensor.Broadcast(TensorType(b).shape)) {
            return a.binaryFn(b, .Eq);
        }
        pub fn xor(comptime a: BoolTensor(Tensor), comptime b: anytype) BoolTensor(Tensor.Broadcast(TensorType(b).shape)) {
            return a.binaryFn(b, .Xor);
        }
        // ReduceOps
        pub fn sum(comptime a: Tensor, comptime dims: anytype) Tensor.ReduceFnResult(dims) {
            return a.reduceFn(.Add, dims);
        }
        pub fn max(comptime a: Tensor, comptime dims: anytype) Tensor.ReduceFnResult(dims) {
            return a.reduceFn(.Max, dims);
        }
        // Compounded operations
        pub fn div(comptime a: Tensor, comptime b: anytype) Tensor.BinaryFnResult(TensorType(b), .Add) {
            return a.mul(tensor.tensorOf(b).recip());
        }
        pub fn sub(comptime a: Tensor, comptime b: anytype) Tensor.BinaryFnResult(TensorType(b), .Add) {
            return a.add(b.neg());
        }
        pub fn sigmoid(comptime a: FloatTensor(Tensor)) FloatTensor(Tensor) {
            const x_pos = a.neg().exp2().add(1.0).recip();
            const x_neg = a.exp2().div(a.exp2().add(1.0));
            const mask = a.lessThan(0.0);
            return mask.where(x_neg, x_pos);
        }
        pub fn relu(comptime a: Tensor) Tensor {
            return a.maximum(0).cast(a.dtype);
        }
        pub fn softmax(comptime a: FloatTensor(Tensor), comptime dim: i16) FloatTensor(Tensor) {
            const minus_max_exp = a.sub(a.max({})).exp2();
            const sumexp = minus_max_exp.sum(dim);
            return minus_max_exp.div(sumexp);
        }
    };
}
