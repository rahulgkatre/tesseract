const std = @import("std");
const tensor = @import("tensor.zig");
const dtypes = @import("dtypes.zig");

pub fn Functions(comptime T: type) type {
    comptime std.debug.assert(tensor.isTensor(T));
    const Tensor = tensor.Tensor(T.dtype, T.ndims, T.shape);
    return struct {
        fn ZipResult(comptime a: Tensor, comptime b: anytype) type {
            const Ta = tensor.TensorType(a);
            const Tb = tensor.TensorType(b).AsType(Ta.dtype);

            const default = dtypes.default;
            const result = dtypes.resultDType(Ta.dtype, Tb.dtype);
            if ((dtypes.isFloat(result) and dtypes.isFloat(default)) or (dtypes.isInt(result) and dtypes.isInt(default)) or (dtypes.isBool(result) and dtypes.isBool(default))) {
                return Ta.Broadcast(Tb.shape).AsType(result);
            } else {
                return Ta.Broadcast(Tb.shape).AsType(default);
            }
        }

        // MapOps
        pub fn exp(a: Tensor) tensor.FloatTensor(Tensor) {
            return a.map(.Exp);
        }
        pub fn log(a: Tensor) tensor.FloatTensor(Tensor) {
            return a.map(.Log);
        }
        pub fn neg(a: Tensor) Tensor {
            return a.map(.Neg);
        }
        pub fn recip(a: Tensor) tensor.FloatTensor(Tensor) {
            return a.map(.Recip);
        }
        pub fn sin(a: Tensor) tensor.FloatTensor(Tensor) {
            return a.map(.Sin);
        }
        pub fn sqrt(a: Tensor) tensor.FloatTensor(Tensor) {
            return a.map(.Sqrt);
        }
        pub fn copy(a: Tensor) Tensor {
            return a.map(.Copy);
        }

        pub fn add(a: Tensor, b: anytype) Tensor.Zip(tensor.TensorType(b).shape, tensor.TensorType(b).dtype, .Add) {
            return a.zip(b, .Add);
        }
        pub fn mul(a: Tensor, b: anytype) Tensor.Zip(tensor.TensorType(b).shape, tensor.TensorType(b).dtype, .Mul) {
            return a.zip(b, .Mul);
        }
        pub fn maximum(a: Tensor, b: anytype) Tensor.Zip(tensor.TensorType(b).shape, tensor.TensorType(b).dtype, .Maximum) {
            return a.zip(b, .Maximum);
        }
        pub fn mod(a: Tensor, b: anytype) tensor.IntTensor(Tensor.Broadcast(tensor.TensorType(b).shape)) {
            return a.zip(b, .Mod);
        }
        pub fn lessThan(a: Tensor, b: anytype) tensor.BoolTensor(Tensor.Broadcast(tensor.TensorType(b).shape)) {
            return a.zip(b, .LessThan);
        }
        pub fn equals(a: Tensor, b: anytype) tensor.BoolTensor(Tensor.Broadcast(tensor.TensorType(b).shape)) {
            return a.zip(b, .Equals);
        }
        pub fn xor(a: Tensor, b: anytype) tensor.IntTensor(Tensor.Broadcast(tensor.TensorType(b).shape)) {
            return a.zip(b, .Xor);
        }

        // ReduceOps
        pub fn sum(a: Tensor, comptime dims: anytype) Tensor.Reduce(dims) {
            return a.reduce(.Sum, dims);
        }
        pub fn max(a: Tensor, comptime dims: anytype) Tensor.Reduce(dims) {
            return a.reduce(.Max, dims);
        }

        // Compounded
        pub fn div(comptime a: Tensor, comptime b: anytype) Tensor.Zip(tensor.TensorType(b).shape, tensor.TensorType(b).dtype, .Add) {
            return a.mul(tensor.tensorOf(b).recip());
        }
        pub fn sub(a: Tensor, b: anytype) Tensor.Zip(tensor.TensorType(b).shape, tensor.TensorType(b).dtype, .Add) {
            return a.add(b.neg());
        }

        pub fn sigmoid(a: Tensor) tensor.FloatTensor(Tensor) {
            const x_pos = a.neg().exp().add(1.0).recip();
            const x_neg = a.exp().div(a.exp().add(1.0));
            const mask = a.lessThan(0.0);
            return mask.where(x_neg, x_pos);
        }

        pub fn relu(a: Tensor) Tensor {
            return a.maximum(0).asType(a.dtype);
        }

        pub fn softmax(a: Tensor, comptime dim: i16) tensor.FloatTensor(Tensor) {
            const minus_max_exp = a.sub(a.max({})).exp();
            const sumexp = minus_max_exp.sum(dim);
            return minus_max_exp.div(sumexp);
        }
    };
}
