const tensor = @import("tensor.zig");

// MapOps
pub fn exp2(comptime x: anytype) @TypeOf(x.*) {
    return x.map(.Exp2);
}
pub fn log2(comptime x: anytype) @TypeOf(x.*) {
    return x.map(.Log2);
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
pub fn sum(comptime x: anytype, comptime reduce_dims: anytype) @TypeOf(x.*).Reduce(reduce_dims) {
    return x.reduce(.Sum, reduce_dims);
}
pub fn max(comptime x: anytype, comptime reduce_dims: anytype) @TypeOf(x.*).Reduce(reduce_dims) {
    return x.reduce(.Max, reduce_dims);
}

// Compound functions that use the ops
pub fn exp(comptime x: anytype) @TypeOf(x.*) {
    // 1 / ln(2) = 1.44269504089
    // e^x = 2^(x / ln(2))
    const recip_ln2 = tensor.constant(@TypeOf(x.*).dtype, 1.44269504089);
    return x.mul(recip_ln2).exp2();
}
pub fn log(comptime x: anytype) @TypeOf(x.*) {
    // ln(2) = 0.69314718056
    // ln(x) = log2(x)ln(2)
    const ln2 = tensor.constant(@TypeOf(x.*).dtype, 0.69314718056);
    return x.log2().mul(ln2);
}
pub fn div(comptime a: anytype, comptime b: anytype) @TypeOf(a.mul(b.recip())) {
    return a.mul(b.recip());
}
pub fn sub(comptime a: anytype, comptime b: anytype) @TypeOf(a.add(b.neg())) {
    return a.add(b.neg());
}

pub fn matmul(comptime a: anytype, comptime b: anytype) @TypeOf(a.*).MatMul(@TypeOf(b)) {
    return a
        .unsqueeze(a.ndims - 1)
        .mul(b.transpose(b.ndims - 2, b.ndims - 1).copy().unsqueeze(b.ndims - 2))
        .sum(b.ndims)
        .squeeze(b.ndims);
}
