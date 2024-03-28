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
        pub fn exp(a: Tensor) FloatTensor(Tensor) {
            return a.map(.Exp);
        }
        pub fn log(a: Tensor) FloatTensor(Tensor) {
            return a.map(.Log);
        }
        pub fn neg(a: Tensor) Tensor {
            return a.map(.Neg);
        }
        pub fn recip(a: Tensor) FloatTensor(Tensor) {
            return a.map(.Recip);
        }
        pub fn sin(a: Tensor) FloatTensor(Tensor) {
            return a.map(.Sin);
        }
        pub fn sqrt(a: Tensor) FloatTensor(Tensor) {
            return a.map(.Sqrt);
        }
        pub fn copy(a: Tensor) Tensor {
            return a.map(.Copy);
        }
        // ZipOps
        pub fn add(a: Tensor, b: anytype) Tensor.Zip(TensorType(b), .Add) {
            return a.zip(b, .Add);
        }
        pub fn mul(a: Tensor, b: anytype) Tensor.Zip(TensorType(b), .Mul) {
            return a.zip(b, .Mul);
        }
        pub fn maximum(a: Tensor, b: anytype) Tensor.Zip(TensorType(b), .Maximum) {
            return a.zip(b, .Maximum);
        }
        pub fn mod(a: Tensor, b: anytype) IntTensor(Tensor.Broadcast(TensorType(b).shape)) {
            return a.zip(b, .Mod);
        }
        pub fn lessThan(a: Tensor, b: anytype) BoolTensor(Tensor.Broadcast(TensorType(b).shape)) {
            return a.zip(b, .LessThan);
        }
        pub fn equals(a: Tensor, b: anytype) BoolTensor(Tensor.Broadcast(TensorType(b).shape)) {
            return a.zip(b, .Equals);
        }
        pub fn xor(a: Tensor, b: anytype) IntTensor(Tensor.Broadcast(TensorType(b).shape)) {
            return a.zip(b, .Xor);
        }
        // ReduceOps
        pub fn sum(a: Tensor, comptime dims: anytype) Tensor.Reduce(dims) {
            return a.reduce(.Sum, dims);
        }
        pub fn max(a: Tensor, comptime dims: anytype) Tensor.Reduce(dims) {
            return a.reduce(.Max, dims);
        }
        // Compounded operations
        pub fn div(comptime a: Tensor, comptime b: anytype) Tensor.Zip(TensorType(b), .Add) {
            return a.mul(tensor.tensorOf(b).recip());
        }
        pub fn sub(a: Tensor, b: anytype) Tensor.Zip(TensorType(b), .Add) {
            return a.add(b.neg());
        }
        pub fn sigmoid(a: Tensor) FloatTensor(Tensor) {
            const x_pos = a.neg().exp().add(1.0).recip();
            const x_neg = a.exp().div(a.exp().add(1.0));
            const mask = a.lessThan(0.0);
            return mask.where(x_neg, x_pos);
        }
        pub fn relu(a: Tensor) Tensor {
            return a.maximum(0).asType(a.dtype);
        }
        pub fn softmax(a: Tensor, comptime dim: i16) FloatTensor(Tensor) {
            const minus_max_exp = a.sub(a.max({})).exp();
            const sumexp = minus_max_exp.sum(dim);
            return minus_max_exp.div(sumexp);
        }
    };
}
