const dtypes = @import("dtypes.zig");
const record = @import("record.zig");
const tensor = @import("tensor.zig");
const std = @import("std");

// anytensor and tensor need to have the exact same runtime layout for @ptrCast tricks to work
comptime {
    const t_info = @typeInfo(tensor.tensor(.bool, .{1})).Struct;
    const a_info = @typeInfo(anytensor).Struct;
    std.debug.assert(t_info.layout == .@"extern");
    std.debug.assert(t_info.layout == a_info.layout);
    for (t_info.fields, a_info.fields) |t_field, a_field| {
        std.debug.assert(t_field.alignment == a_field.alignment);
        std.debug.assert(t_field.is_comptime == a_field.is_comptime);
        std.debug.assert(t_field.type == a_field.type);
    }
}

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

    pub fn Narrow(comptime self: anytensor) type {
        return tensor.Tensor(self.dtype, self.ndims, self.shape[0..self.ndims][0..].*);
    }

    /// Performs type narrowing to get back the shapetyped Tensor
    pub fn narrow(comptime self: *const anytensor) *const Narrow(self.*) {
        return @ptrCast(self);
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
