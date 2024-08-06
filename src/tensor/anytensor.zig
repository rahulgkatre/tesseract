const std = @import("std");
const dtypes = @import("../dtypes.zig");
const utils = @import("../utils.zig");
const graph = @import("../graph.zig");

const shared = @import("shared.zig");
const tensor = @import("tensor.zig");
const tensor_typing = @import("tensor_typing.zig");
const TensorTypeOf = tensor_typing.AsTensorType;

// AnyTensor and tensor need to have the exact same runtime layout for @ptrCast tricks to work
comptime {
    const t_info = @typeInfo(tensor.Tensor(f32)).Struct;
    const a_info = @typeInfo(AnyTensor).Struct;
    std.debug.assert(t_info.layout == .@"extern");
    std.debug.assert(t_info.layout == a_info.layout);
    for (t_info.fields, a_info.fields) |t_field, a_field| {
        std.debug.assert(std.mem.eql(u8, t_field.name, a_field.name));
        std.debug.assert(t_field.alignment == a_field.alignment);
        std.debug.assert(@sizeOf(t_field.type) == @sizeOf(a_field.type));
    }
}

/// Strips away generic information to make it easier to work with pointers to tensors
/// with different shapes, dtypes, etc.
/// By making AnyTensor and generic tensor extern structs, they are guaranteed to have
/// the same layout.
pub const AnyTensor = extern struct {
    meta: *const shared.Metadata,
    dtype: dtypes.DType,
    ndims: u8,
    shape: [*]const u64,
    strides: [*]const u64,
    offset: u64,

    pub const Json = struct {
        ptr: usize,
        dtype: dtypes.DType,
        ndims: u8,
        shape: []const u64,
        strides: []const u64,
        offset: u64,
    };

    pub fn toJson(self: *const AnyTensor) Json {
        return .{
            .ptr = @intFromPtr(self),
            .dtype = self.dtype,
            .ndims = self.ndims,
            .shape = self.shape[0..self.ndims],
            .strides = self.strides[0..self.ndims],
            .offset = self.offset,
        };
    }
};
