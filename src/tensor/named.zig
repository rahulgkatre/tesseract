const std = @import("std");
const tensor = @import("tensor.zig");
const shared = @import("shared.zig");
const dtypes = @import("../dtypes.zig");

pub fn NamedTensor(comptime Array: type, maybe_dim_names: ?[]const ?[]const u8) type {
    return struct {
        pub usingnamespace tensor.Tensor(Array);
        const Self = @This();

        meta: *const shared.Metadata,
        dtype: dtypes.DType = Self.dtype,
        ndims: u8 = Self.ndims,
        shape: *const [Self.ndims]u64 = &Self.shape,
        strides: *const [Self.ndims]u64 = &Self.contiguous_strides,
        offset: u64 = 0,

        pub const NamedDim = blk: {
            var dim_enum_fields: [Self.ndims]std.builtin.Type.EnumField = undefined;
            var enum_idx: usize = 0;
            if (maybe_dim_names) |dim_names| {
                for (dim_names, 0..) |maybe_name, dim_idx| {
                    if (maybe_name) |name| {
                        dim_enum_fields[enum_idx] = std.builtin.Type.EnumField{ .name = name[0.. :0], .value = dim_idx };
                        enum_idx += 1;
                    }
                }
                break :blk @Type(std.builtin.Type{ .Enum = .{ .fields = dim_enum_fields[0..enum_idx], .is_exhaustive = false, .tag_type = u8, .decls = &.{} } });
            } else {
                break :blk void;
            }
        };

        pub fn setDimNames(
            self: *const Self,
            comptime dim_names: std.meta.Tuple(&[_]type{?[]const u8} ** Self.ndims),
            allocator: std.mem.Allocator,
        ) *const NamedTensor(Array, dim_names) {
            const meta = allocator.create(shared.Metadata) catch unreachable;
            meta.* = self.meta.*;
            meta.dim_names = &dim_names;

            const Out = comptime NamedTensor(Array, dim_names);
            const out: *Out = allocator.create(Out) catch unreachable;
            out.* = .{
                .meta = meta,
                .strides = self.strides,
                .offset = self.offset,
            };
            return out;
        }

        pub fn namedDim(_: *const Self, dim: NamedDim) u64 {
            return @intFromEnum(dim);
        }
    };
}
