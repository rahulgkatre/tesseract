const ops = @import("ops.zig");

pub const GraphTensor = struct {
    const Self = @This();
    permute_fn: *const fn (comptime ptr: *const Self, comptime perm: []u8) Self,
    map_fn: *const fn (comptime ptr: *const Self, comptime map_op: ops.MapOp) Self,
    zip_fn: *const fn (comptime ptr: *const Self, comptime zip_op: ops.ZipOp, comptime other_ptr: anytype) Self,
    reduce_fn: *const fn (comptime ptr: *const Self, comptime reduce_op: ops.ReduceOp, comptime reduce_dim: u8) Self,
    pub fn permute(comptime self: *const Self, comptime perm :[]u8) Self {
        return self.permute_fn(self, perm);
    }
    pub fn map(comptime self: *const Self, comptime map_op: ops.MapOp) Self {
        return self.map_fn(self, map_op);
    }
    pub fn zip(comptime self: *const Self, comptime zip_op: ops.ZipOp, comptime other_ptr: anytype) Self {
        return self.zip_fn(self, zip_op, other_ptr);
    }
    pub fn reduce(comptime self: *const Self, comptime reduce_op: ops.ReduceOp, comptime reduce_dim: u8) Self {
        return self.reduce_fn(self, reduce_op, reduce_dim);
    }
};


const Input = enum {
    Tensor,
    Array,
    Value,
};

const MAX_NDIMS = 8;
const OpInput = union(Input) { 
    Tensor: *const GraphTensor,
    Array: [MAX_NDIMS]usize, 
    Value: usize,
};

const History = struct { op: ops.Op, inputs: [2]OpInput };

pub fn extendShape(comptime in_ndims: u8, in_shape: [in_ndims]usize, comptime out_ndims: u8) [out_ndims]usize {
    var out_shape: [out_ndims]usize = undefined;
    @memset(&out_shape, 1);
    @memcpy(out_shape[(out_ndims - in_ndims)..], &in_shape);
    return out_shape;
}

pub fn extendStrides(comptime in_ndims: u8, in_strides: [in_ndims]usize, comptime out_ndims: u8) [out_ndims]usize {
    var out_strides: [out_ndims]usize = undefined;
    @memset(&out_strides, 0);
    @memcpy(out_strides[(out_ndims - in_ndims)..], &in_strides);
    return out_strides;
}
