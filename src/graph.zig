const ops = @import("ops.zig");

pub const GraphTensor = struct {
    const Self = @This();
    // TODO: Can we make the args of these not comptime?
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

// TODO: Revise this and improve typing
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

