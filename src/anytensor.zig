const dtypes = @import("dtypes.zig");
const record = @import("record.zig");
const tensor = @import("tensor.zig");

/// Strips away generic information to make it easier to work with pointers to tensors
/// with different shapes, dtypes, etc.
/// By making anytensor and generic tensor extern structs, they are guaranteed to have
/// the same layout.
pub const anytensor = extern struct {
    dtype: dtypes.DType,
    ndims: u8,
    shape: [*]const u64,
    strides: [*]const u64,
    offset: u64,
    record: *const record.Record,

    pub fn TensorType(comptime self: anytensor) type {
        return tensor.Tensor(self.dtype, self.ndims, self.shape[0..self.ndims][0..].*);
    }

    pub fn infer(comptime self: anytensor) TensorType(self) {
        return @as(*const TensorType(self), @ptrCast(&self)).*;
    }

    pub const JsonFormat = struct {
        uid: usize,
        dtype: dtypes.DType,
        ndims: u8,
        shape: []const u64,
        strides: []const u64,
        offset: u64,
    };

    pub fn toJsonFormat(self: *const anytensor) JsonFormat {
        return .{
            .uid = @intFromPtr(self),
            .dtype = self.dtype,
            .ndims = self.ndims,
            .shape = self.shape[0..self.ndims],
            .strides = self.strides[0..self.ndims],
            .offset = self.offset,
        };
    }
};
