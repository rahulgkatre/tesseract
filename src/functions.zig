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
        pub fn exp(comptime a: Tensor) FloatTensor(Tensor) {
            return a.unaryFn(.Exp);
        }
        pub fn log(comptime a: Tensor) FloatTensor(Tensor) {
            return a.unaryFn(.Log);
        }
        pub fn neg(comptime a: Tensor) Tensor {
            return a.unaryFn(.Neg);
        }
        pub fn recip(comptime a: Tensor) FloatTensor(Tensor) {
            return a.unaryFn(.Recip);
        }
        pub fn sin(comptime a: Tensor) FloatTensor(Tensor) {
            return a.unaryFn(.Sin);
        }
        pub fn sqrt(comptime a: Tensor) FloatTensor(Tensor) {
            return a.unaryFn(.Sqrt);
        }
        pub fn copy(comptime a: Tensor) Tensor {
            return a.unaryFn(.Copy);
        }
        // ZipOps
        pub fn add(comptime a: Tensor, comptime b: anytype) Tensor.BinaryFnResult(TensorType(b), .Add) {
            return a.binaryFn(b, .Add);
        }
        pub fn mul(comptime a: Tensor, comptime b: anytype) Tensor.BinaryFnResult(TensorType(b), .Mul) {
            return a.binaryFn(b, .Mul);
        }
        pub fn maximum(comptime a: Tensor, comptime b: anytype) Tensor.BinaryFnResult(TensorType(b), .Maximum) {
            return a.binaryFn(b, .Maximum);
        }
        pub fn mod(comptime a: IntTensor(Tensor), comptime b: anytype) IntTensor(Tensor.Broadcast(TensorType(b).shape)) {
            return a.binaryFn(b, .Mod);
        }
        pub fn lessThan(comptime a: Tensor, comptime b: anytype) BoolTensor(Tensor.Broadcast(TensorType(b).shape)) {
            return a.binaryFn(b, .LessThan);
        }
        pub fn equals(comptime a: BoolTensor(Tensor), comptime b: anytype) BoolTensor(Tensor.Broadcast(TensorType(b).shape)) {
            return a.binaryFn(b, .Equals);
        }
        pub fn xor(comptime a: BoolTensor(Tensor), comptime b: anytype) BoolTensor(Tensor.Broadcast(TensorType(b).shape)) {
            return a.binaryFn(b, .Xor);
        }
        // ReduceOps
        pub fn sum(comptime a: Tensor, comptime dims: anytype) Tensor.ReduceFnResult(dims) {
            return a.reduceFn(.Sum, dims);
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
            const x_pos = a.neg().exp().add(1.0).recip();
            const x_neg = a.exp().div(a.exp().add(1.0));
            const mask = a.lessThan(0.0);
            return mask.where(x_neg, x_pos);
        }
        pub fn relu(comptime a: Tensor) Tensor {
            return a.maximum(0).asType(a.dtype);
        }
        pub fn softmax(comptime a: FloatTensor(Tensor), comptime dim: i16) FloatTensor(Tensor) {
            const minus_max_exp = a.sub(a.max({})).exp();
            const sumexp = minus_max_exp.sum(dim);
            return minus_max_exp.div(sumexp);
        }
    };
}
