const std = @import("std");
const tensor = @import("tensor.zig");
const dtypes = @import("dtypes.zig");

const AsTensor = tensor.AsTensor;
const isTensor = tensor.isTensor;
const IntTensor = dtypes.IntTensor;
const BoolTensor = dtypes.BoolTensor;
const FloatTensor = dtypes.FloatTensor;

const LN_2 = tensor.asTensor(0.69314718056);
const INV_LN_2 = tensor.asTensor(1.4426950408888495760773985077695);

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
        pub fn add(comptime a: Tensor, comptime b: anytype) Tensor.BinaryFnResult(b, .Add) {
            return a.binaryFn(b, .Add);
        }
        pub fn mul(comptime a: Tensor, comptime b: anytype) Tensor.BinaryFnResult(b, .Mul) {
            return a.binaryFn(b, .Mul);
        }
        pub fn maximum(comptime a: Tensor, comptime b: anytype) Tensor.BinaryFnResult(b, .Max) {
            return a.binaryFn(b, .Max);
        }
        pub fn mod(comptime a: IntTensor(Tensor), comptime b: anytype) Tensor.BinaryFnResult(b, .Mod) {
            return a.binaryFn(b, .Mod);
        }
        pub fn lessThan(comptime a: Tensor, comptime b: anytype) Tensor.BinaryFnResult(b, .Lt) {
            return a.binaryFn(b, .Lt);
        }
        pub fn equals(comptime a: BoolTensor(Tensor), comptime b: anytype) Tensor.BinaryFnResult(b, .Eq) {
            return a.binaryFn(b, .Eq);
        }
        pub fn xor(comptime a: BoolTensor(Tensor), comptime b: anytype) Tensor.BinaryFnResult(b, .Xor) {
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
        pub fn div(comptime a: Tensor, comptime b: anytype) Tensor.BinaryFnResult(b, .Add) {
            return a.mul(tensor.asTensor(b).recip());
        }
        pub fn sub(comptime a: Tensor, comptime b: anytype) Tensor.BinaryFnResult(b, .Add) {
            return a.add(tensor.asTensor(b).neg());
        }
        pub fn exp(comptime a: FloatTensor(Tensor)) FloatTensor(Tensor) {
            return a.startGroup("exp").mul(INV_LN_2).exp2().endGroup();
        }
        pub fn log(comptime a: FloatTensor(Tensor)) FloatTensor(Tensor) {
            return a.startGroup("log").log2().mul(LN_2).endGroup();
        }
        pub fn sigmoid(comptime raw_a: FloatTensor(Tensor)) FloatTensor(Tensor) {
            const a = raw_a.startGroup("sigmoid");
            const x_pos = a.neg().exp().add(1.0).recip();
            const x_neg = a.exp().div(a.exp().add(1.0));
            const mask = a.lessThan(0.0);
            return mask.where(x_neg, x_pos).endGroup();
        }
        pub fn relu(comptime a: Tensor) Tensor {
            return a.startGroup("relu").maximum(0).cast(a.dtype).endGroup();
        }
        pub fn softmax(comptime raw_a: FloatTensor(Tensor), comptime dim: i16) FloatTensor(Tensor) {
            const a = raw_a.startGroup("softmax");
            const minus_max_exp = a.sub(a.max({})).exp();
            const sumexp = minus_max_exp.sum(dim);
            return minus_max_exp.div(sumexp).endGroup();
        }
    };
}
