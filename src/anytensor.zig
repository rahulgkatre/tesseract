const dtypes = @import("dtypes.zig");
const tracker = @import("tracker.zig");
const tensor = @import("tensor.zig");
const utils = @import("utils.zig");
const std = @import("std");

// AnyTensor and tensor need to have the exact same runtime layout for @ptrCast tricks to work
comptime {
    const t_info = @typeInfo(tensor.Tensor(f32)).Struct;
    const a_info = @typeInfo(AnyTensor).Struct;
    std.debug.assert(t_info.layout == .@"extern");
    std.debug.assert(t_info.layout == a_info.layout);
    for (t_info.fields, a_info.fields) |t_field, a_field| {
        std.debug.assert(t_field.alignment == a_field.alignment);
        std.debug.assert(@sizeOf(t_field.type) == @sizeOf(a_field.type));
    }
}

/// Strips away generic information to make it easier to work with pointers to tensors
/// with different shapes, dtypes, etc.
/// By making AnyTensor and generic tensor extern structs, they are guaranteed to have
/// the same layout.
pub const AnyTensor = extern struct {
    dtype: dtypes.DType,
    ndims: u8,
    shape: [*]const u64,
    strides: [*]const u64,
    offset: u64,
    meta: *const tensor.Metadata,

    pub fn Narrow(comptime self: AnyTensor) type {
        return tensor.AsTensor(self);
    }

    /// Performs type narrowing to get back the shapetyped Tensor
    pub fn narrow(comptime self: *const AnyTensor) *const Narrow(self.*) {
        return @ptrCast(self);
    }

    pub const Json = struct {
        uid: usize,
        dtype: dtypes.DType,
        ndims: u8,
        shape: []const u64,
        strides: []const u64,
        offset: u64,
    };

    pub fn toJson(self: *const AnyTensor) Json {
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
