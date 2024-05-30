const std = @import("std");
const tensor = @import("tensor.zig");
const dtypes = @import("dtypes.zig");

const TensorTypeOf = tensor.TensorTypeOf;
const isTensor = tensor.isTensor;
const IntTensor = dtypes.IntTensor;
const BoolTensor = dtypes.BoolTensor;
const FloatTensor = dtypes.FloatTensor;

const LN_2 = tensor.asTensor(0.69314718056);
const INV_LN_2 = tensor.asTensor(1.4426950408888495760773985077695);

pub fn Functions(comptime T: type) type {
    const Tensor = tensor.Tensor(T.ArrayType());
    return struct {
        // MapOps
        pub fn exp2(a: FloatTensor(Tensor)) FloatTensor(Tensor) {
            return a.unaryFn(.Exp2);
        }
        pub fn log2(a: FloatTensor(Tensor)) FloatTensor(Tensor) {
            return a.unaryFn(.Log2);
        }
        pub fn neg(a: Tensor) Tensor {
            return a.unaryFn(.Neg);
        }
        pub fn recip(a: FloatTensor(Tensor)) FloatTensor(Tensor) {
            return a.unaryFn(.Rcp);
        }
        pub fn sin(a: FloatTensor(Tensor)) FloatTensor(Tensor) {
            return a.unaryFn(.Sin);
        }
        pub fn sqrt(a: FloatTensor(Tensor)) FloatTensor(Tensor) {
            return a.unaryFn(.Sqrt);
        }
        // ZipOps
        pub fn add(a: Tensor, comptime b: anytype) Tensor.BinaryFnResultType(b, .Add) {
            return a.binaryFn(b, .Add);
        }
        pub fn mul(a: Tensor, comptime b: anytype) Tensor.BinaryFnResultType(b, .Mul) {
            return a.binaryFn(b, .Mul);
        }
        pub fn maximum(a: Tensor, comptime b: anytype) Tensor.BinaryFnResultType(b, .Max) {
            return a.binaryFn(b, .Max);
        }
        pub fn mod(a: IntTensor(Tensor), comptime b: anytype) Tensor.BinaryFnResultType(b, .Mod) {
            return a.binaryFn(b, .Mod);
        }
        pub fn lessThan(a: Tensor, comptime b: anytype) Tensor.BinaryFnResultType(b, .Lt) {
            return a.binaryFn(b, .Lt);
        }
        pub fn equals(a: BoolTensor(Tensor), comptime b: anytype) Tensor.BinaryFnResultType(b, .Eq) {
            return a.binaryFn(b, .Eq);
        }
        pub fn xor(a: BoolTensor(Tensor), comptime b: anytype) Tensor.BinaryFnResultType(b, .Xor) {
            return a.binaryFn(b, .Xor);
        }
        // ReduceOps
        pub fn sum(a: Tensor, dims: anytype) Tensor.ReduceFnResultType(dims) {
            return a.reduceFn(.Add, dims);
        }
        pub fn max(a: Tensor, comptime dims: anytype) Tensor.ReduceFnResultType(dims) {
            return a.reduceFn(.Max, dims);
        }
        // Compounded operations
        pub fn div(a: Tensor, b: anytype) Tensor.BinaryFnResultType(b, .Add) {
            return a.startGroup("div").mul(tensor.asTensor(b).recip()).endGroup();
        }
        pub fn sub(a: Tensor, b: anytype) Tensor.BinaryFnResultType(b, .Add) {
            return a.add(tensor.asTensor(b).neg());
        }
        pub fn exp(a: FloatTensor(Tensor)) FloatTensor(Tensor) {
            return a.startGroup("exp").mul(INV_LN_2).exp2().endGroup();
        }
        pub fn log(a: FloatTensor(Tensor)) FloatTensor(Tensor) {
            return a.startGroup("log").log2().mul(LN_2).endGroup();
        }
        pub fn sigmoid(_a: FloatTensor(Tensor)) FloatTensor(Tensor) {
            const a = _a.startGroup("sigmoid");
            const x_pos = a.neg().exp().add(1.0).recip();
            const x_neg = a.exp().div(a.exp().add(1.0));
            const mask = a.lessThan(0.0);
            return mask.where(x_neg, x_pos).endGroup();
        }
        pub fn relu(a: Tensor) Tensor {
            if (dtypes.isFloat(a.dtype)) {
                return a.startGroup("relu").maximum(0.0).endGroup();
            } else if (dtypes.isInt(a.dtype)) {
                return a.startGroup("relu").maximum(0).endGroup();
            } else {
                unreachable;
            }
        }
        pub fn softmax(_a: FloatTensor(Tensor), comptime dim: i16) FloatTensor(Tensor) {
            const a = _a.startGroup("softmax");
            const minus_max_exp = a.sub(a.max({})).exp();
            const sumexp = minus_max_exp.sum(dim);
            return minus_max_exp.div(sumexp).endGroup();
        }
        pub fn mean(a: Tensor, comptime dims: anytype) FloatTensor(Tensor.ReduceFnResultType(dims)) {
            return a.div(a.sum(dims));
        }
        pub fn variance(_a: Tensor, comptime dims: anytype) FloatTensor(Tensor.ReduceFnResultType(dims)) {
            const a = _a.startGroup("variance");
            const mu = a.mean(dims);
            const N: f64 = @floatFromInt(@divExact(a.num_entries, mu.num_entries));
            const a_minus_mu = a.sub(mu);
            return (a_minus_mu.mul(a_minus_mu)).sum(dims).div(N);
        }
    };
}
