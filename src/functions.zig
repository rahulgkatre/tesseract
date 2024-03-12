const tensor = @import("tensor.zig");

// MapOps
pub fn exp(comptime x: anytype) @TypeOf(x.*) {
    return x.map(.Exp);
}
pub fn log(comptime x: anytype) @TypeOf(x.*) {
    return x.map(.Log);
}
pub fn neg(comptime x: anytype) @TypeOf(x.*) {
    return x.map(.Neg);
}
pub fn recip(comptime x: anytype) @TypeOf(x.*) {
    return x.map(.Recip);
}
pub fn sin(comptime x: anytype) @TypeOf(x.*) {
    return x.map(.Sin);
}
pub fn sqrt(comptime x: anytype) @TypeOf(x.*) {
    return x.map(.Sqrt);
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
pub fn less_than(comptime a: anytype, comptime b: anytype) @TypeOf(a.zip(.LessThan, b)) {
    return a.zip(.LessThan, b);
}
pub fn equals(comptime a: anytype, comptime b: anytype) @TypeOf(a.zip(.Equals, b)) {
    return a.zip(.Equals, b);
}
pub fn xor(comptime a: anytype, comptime b: anytype) @TypeOf(a.zip(.Xor, b)) {
    return a.zip(.Xor, b);
}

// ReduceOps
pub fn sum(comptime x: anytype, comptime dims: anytype) @TypeOf(x.*).Reduce(dims) {
    return x.reduce(.Sum, dims);
}
pub fn max(comptime x: anytype, comptime dims: anytype) @TypeOf(x.*).Reduce(dims) {
    return x.reduce(.Max, dims);
}

// Compounded
pub fn div(comptime a: anytype, comptime b: anytype) @TypeOf(a.mul(b.recip())) {
    return a.mul(b.recip());
}
pub fn sub(comptime a: anytype, comptime b: anytype) @TypeOf(a.add(b.neg())) {
    return a.add(b.neg());
}

pub fn matmul(comptime a: anytype, comptime b: anytype) @TypeOf(a.*).MatMul(@TypeOf(b)) {
    const a_mul_b = a
        .unsqueeze(a.ndims - 1)
        .mul(b.transpose(b.ndims - 2, b.ndims - 1).copy().unsqueeze(b.ndims - 2));
    return a_mul_b
        .sum(a_mul_b.ndims - 1)
        .squeeze(a_mul_b.ndims - 1);
}
