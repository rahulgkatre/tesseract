const tensor = @import("tensor.zig");

// MapOps
pub fn exp(comptime a: anytype) tensor.TensorType(a) {
    return a.map(.Exp);
}
pub fn log(comptime a: anytype) tensor.TensorType(a) {
    return a.map(.Log);
}
pub fn neg(comptime a: anytype) tensor.TensorType(a) {
    return a.map(.Neg);
}
pub fn recip(comptime a: anytype) tensor.TensorType(a) {
    return a.map(.Recip);
}
pub fn sin(comptime a: anytype) tensor.TensorType(a) {
    return a.map(.Sin);
}
pub fn sqrt(comptime a: anytype) tensor.TensorType(a) {
    return a.map(.Sqrt);
}

// ZipOps
pub fn add(comptime a: anytype, comptime b: anytype) @TypeOf(a.zip(.Add, b)) {
    return a.zip(.Add, b);
}
pub fn mul(comptime a: anytype, comptime b: anytype) @TypeOf(a.zip(.Mul, b)) {
    return a.zip(.Mul, b);
}
pub fn maximum(comptime a: anytype, comptime b: anytype) @TypeOf(a.zip(.Maximum, b)) {
    return a.zip(.Maximum, b);
}
pub fn mod(comptime a: anytype, comptime b: anytype) @TypeOf(a.zip(.Mod, b)) {
    return a.zip(.Mod, b);
}
pub fn lessThan(comptime a: anytype, comptime b: anytype) @TypeOf(a.zip(.LessThan, b)) {
    return a.zip(.LessThan, b);
}
pub fn equals(comptime a: anytype, comptime b: anytype) @TypeOf(a.zip(.Equals, b)) {
    return a.zip(.Equals, b);
}
pub fn xor(comptime a: anytype, comptime b: anytype) @TypeOf(a.zip(.Xor, b)) {
    return a.zip(.Xor, b);
}

// ReduceOps
pub fn sum(comptime a: anytype, comptime dims: anytype) @TypeOf(a.*).Reduce(dims) {
    return a.reduce(.Sum, dims);
}
pub fn max(comptime a: anytype, comptime dims: anytype) @TypeOf(a.*).Reduce(dims) {
    return a.reduce(.Max, dims);
}

// Compounded
pub fn div(comptime a: anytype, comptime b: anytype) t: {
    const a_tensor = tensor.tensorOf(a);
    const b_tensor = tensor.tensorOf(b);
    break :t @TypeOf(a_tensor.mul(b_tensor.recip()));
} {
    const a_tensor = tensor.tensorOf(a);
    const b_tensor = tensor.tensorOf(b);
    return a_tensor.mul(b_tensor.recip());
}
pub fn sub(comptime a: anytype, comptime b: anytype) @TypeOf(a.add(b.neg())) {
    return a.add(b.neg());
}

pub fn sigmoid(comptime a: anytype) @TypeOf(a) {
    const x_pos = a.neg().exp().add(1.0).recip();
    const x_neg = a.exp().div(a.exp().add(1.0));
    const mask = a.lessThan(0.0);
    return mask.where(x_neg, x_pos);
}

pub fn relu(comptime a: anytype) @TypeOf(a) {
    return a.maximum(0);
}

pub fn abs(a: anytype) @TypeOf(a) {}
