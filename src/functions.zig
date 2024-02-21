const tensor = @import("tensor.zig");

// MapOps
pub fn exp2(x: anytype) @TypeOf(x.*) {
    return x.map(.Exp2);
}
pub fn log2(x: anytype) @TypeOf(x.*) {
    return x.map(.Log2);
}
pub fn neg(x: anytype) @TypeOf(x.*) {
    return x.map(.Neg);
}
pub fn recip(x: anytype) @TypeOf(x.*) {
    return x.map(.Recip);
}
pub fn sin(x: anytype) @TypeOf(x.*) {
    return x.map(.Sin);
}
pub fn sqrt(x: anytype) @TypeOf(x.*) {
    return x.map(.Sqrt);
}

// ZipOps
pub fn add(a: anytype, b: anytype) @TypeOf(a.zip(.Add, b)) {
    return a.zip(.Add, b);
}
pub fn mul(a: anytype, b: anytype) @TypeOf(a.zip(.Mul, b)) {
    return a.zip(.Mul, b);
}
pub fn maximum(a: anytype, b: anytype) @TypeOf(a.zip(.Maximum, b)) {
    return a.zip(.Maximum, b);
}
pub fn mod(a: anytype, b: anytype) @TypeOf(a.zip(.Mod, b)) {
    return a.zip(.Mod, b);
}
pub fn less_than(a: anytype, b: anytype) @TypeOf(a.zip(.LessThan, b)) {
    return a.zip(.LessThan, b);
}
pub fn equals(a: anytype, b: anytype) @TypeOf(a.zip(.Equals, b)) {
    return a.zip(.Equals, b);
}
pub fn xor(a: anytype, b: anytype) @TypeOf(a.zip(.Xor, b)) {
    return a.zip(.Xor, b);
}

// ReduceOps
pub fn sum(x: anytype, comptime reduce_dims: anytype) @TypeOf(x.*).Reduce(reduce_dims) {
    return x.reduce(.Sum, reduce_dims);
}
pub fn max(x: anytype, comptime reduce_dims: anytype) @TypeOf(x.*).Reduce(reduce_dims) {
    return x.reduce(.Max, reduce_dims);
}

// Compound functions that use the ops
pub fn exp(x: anytype) @TypeOf(x.*) {
    // 1 / ln(2) = 1.44269504089
    // e^x = 2^(x / ln(2))
    const recip_ln2 = tensor.constant(@TypeOf(x.*).dtype, 1.44269504089);
    return x.mul(recip_ln2).exp2();
}
pub fn log(x: anytype) @TypeOf(x.*) {
    // ln(2) = 0.69314718056
    // ln(x) = log2(x)ln(2)
    const ln2 = tensor.constant(@TypeOf(x.*).dtype, 0.69314718056);
    return x.log2().mul(ln2);
}
pub fn div(a: anytype, b: anytype) @TypeOf(a.mul(b.recip())) {
    return a.mul(b.recip());
}
pub fn sub(a: anytype, b: anytype) @TypeOf(a.add(b.neg())) {
    return a.add(b.neg());
}

pub fn matmul(a: anytype, b: anytype) @TypeOf(a.*).MatMul(@TypeOf(b)) {
    const a_unsqueeze = a.unsqueeze(@TypeOf(a.*).ndims);
    const b_unsqueeze = b.unsqueeze(@TypeOf(b).ndims - 2);
    const outer_prod = a_unsqueeze.mul(b_unsqueeze);
    return outer_prod.sum(@TypeOf(outer_prod).ndims - 2).squeeze(@TypeOf(outer_prod).ndims - 2);
}
