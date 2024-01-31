const std = @import("std");
const Allocator = std.mem.Allocator;
const comptimePrint = std.fmt.comptimePrint;
const utils = @import("utils.zig");
const ops = @import("ops.zig");
const Graph = @import("Graph.zig");

pub var debug = false;

// TensorArena provides an allocator for the tensor metadata
// No actual elements of the tensor are stored by this allocator

pub fn constant(comptime dtype: type, comptime value: dtype) Tensor(dtype, .{1}) {
    return Tensor(dtype, .{1}).full(value);
}

pub fn range(comptime dtype: type, comptime start: dtype, comptime stop: dtype) Tensor(dtype, .{stop - start}) {
    @setEvalBranchQuota(@as(u32, 2 * stop));
    const data: [stop - start]dtype = std.simd.iota(dtype, stop - start) + @as(@Vector(stop - start, dtype), @splat(start));
    return Tensor(dtype, .{stop - start}).fromData(data[0..]);
}

pub fn Tensor(comptime dtype: type, comptime shape: anytype) type {
    return AsStrided(dtype, shape, utils.stridesFromShape(shape));
}

fn AsStrided(comptime dtype: type, comptime shape: anytype, comptime strides: anytype) type {
    if (shape.len + 1 != strides.len) {
        @compileError("Provided shape ndims not compatible with provided strides ndims, you may be missing the storage offset (strides[ndims])");
    }
    return TensorView(dtype, shape.len, shape, strides);
}

// A Tensor is actually a TensorView, this is probably the best name for it because
// its generic parameters directly affect how data is accessed (viewed)
// While TensorView provides the API, the constructor is not the friendliest
// hence there is a simpler Tensor constructor
fn TensorView(comptime _dtype: type, comptime _ndims: u8, comptime _shape: [_ndims]usize, comptime _strides: [_ndims + 1]usize) type {
    return struct {
        const Self = @This();
        pub const dtype: type = _dtype;
        pub const ndims: u8 = _ndims;
        pub const shape: [ndims]usize = _shape;
        pub const strides: [ndims + 1]usize = _strides;
        pub const size = utils.storageSizeForTensor(ndims, shape, strides);

        ndims: u8 = ndims,
        shape: [ndims]usize = shape,
        size: usize = size,
        strides: [ndims + 1]usize = strides,
        traceFn: *const fn (self: *const Self) void,

        pub fn init(
            comptime traceFn: *const fn (self: *const Self) void,
        ) Self {
            return .{ .traceFn = traceFn };
        }

        // Load the tensor's data from an array pointer
        // Not a slice because this guarantees that the size requirement is met and verified in comptime
        pub fn fromData(data: *const [size]dtype) Self {
            _ = data;
            const impl = struct {
                fn trace(self: *const Self) void {
                    Graph.new_node(self, .{ .InitOp = .{ .op = .FromData } }, Self);
                }
            };
            return init(impl.trace);
        }

        // Fill a tensor with a value
        pub fn full(comptime value: dtype) Self {
            _ = value;
            const impl = struct {
                fn trace(self: *const Self) void {
                    Graph.new_node(self, .{ .InitOp = .{ .op = .Full } }, Self);
                }
            };
            return init(impl.trace);
        }

        pub fn trace(comptime self: *const Self) void {
            self.traceFn(self);
        }

        pub fn isContiguous(_: *const Self) bool {
            return comptime utils.isContiguous(ndims, strides);
        }
        pub inline fn broadcastIndex(_: *const Self, bc_index: anytype) [ndims]usize {
            // Determine the index in the current tensor given an index in the broadcasted tensor
            // If the current tensor has size of 1 in a dimension, then the index must be 0
            // Otherwise it will be what the broadcasted index is
            const bc_ndims = bc_index.len;
            var index: [ndims]usize = undefined;
            inline for (0..ndims) |d| {
                index[ndims - d - 1] = if (shape[ndims - d - 1] == 1) 0 else bc_index[bc_ndims - d - 1];
            }
            return index;
        }
        pub inline fn idxToPos(_: anytype, index: [ndims]usize) usize {
            const index_vec: @Vector(ndims, usize) = index;
            const strides_vec: @Vector(ndims, usize) = strides[0..ndims].*;
            return @reduce(.Add, index_vec * strides_vec) + strides[ndims];
        }
        pub inline fn posToIdx(_: anytype, flat_index: usize) [ndims]usize {
            var index: [ndims]usize = undefined;
            var remainder = flat_index - strides[ndims];
            inline for (0..ndims) |d| {
                if (strides[d] == 0) {
                    index[d] = 0;
                } else {
                    index[d] = @divTrunc(remainder, strides[d]);
                    remainder = @mod(remainder, strides[d]);
                }
            }
            return index;
        }
        pub fn Permute(comptime perm: [ndims]u8) type {
            var strides_perm: [ndims + 1]u8 = undefined;
            @memcpy(strides_perm[0..ndims], &perm);
            strides_perm[ndims] = ndims;
            return AsStrided(
                dtype,
                utils.permuteArray(ndims, shape, perm),
                utils.permuteArray(ndims + 1, strides, strides_perm),
            );
        }
        pub fn permute(self: *const Self, comptime perm: [ndims]u8) Permute(perm) {
            const Out = Permute(perm);
            const impl = struct {
                fn trace(out: *const Out) void {
                    self.trace();
                    Graph.new_node(out, .{ .TypeOp = .{ .op = .Permute, .x = Graph.get_node(self) } }, Out);
                }
            };
            return Out.init(impl.trace);
        }
        pub fn view(self: *const Self, comptime new_shape: anytype) Tensor(dtype, new_shape) {
            const Out = Tensor(dtype, new_shape);
            std.debug.assert(Out.size == size);
            if (self.isContiguous()) {
                const impl = struct {
                    fn trace(out: *const Out) void {
                        self.trace();
                        Graph.new_node(out, .{ .TypeOp = .{ .op = .View, .x = Graph.get_node(self) } }, Out);
                    }
                };
                return Out.init(impl.trace);
            } else {
                @compileError("Must be contiguous to view");
            }
        }

        pub fn asStrided(self: *const Self, comptime new_shape: anytype, comptime new_strides: anytype) AsStrided(dtype, new_shape, new_strides) {
            const Out = AsStrided(dtype, new_shape, new_strides);
            const impl = struct {
                fn trace(out: *const Out) void {
                    self.trace();
                    Graph.new_node(out, .{ .TypeOp = .{ .op = .AsStrided, .x = Graph.get_node(self) } }, Out);
                }
            };
            return init(impl.trace);
        }

        pub fn Cast(comptime new_dtype: type) type {
            return TensorView(new_dtype, ndims, shape, strides);
        }
        pub fn cast(self: *const Self, comptime new_dtype: type) Cast(new_dtype) {
            return self.graph.?.cast(new_dtype, self);
        }

        // TODO: Zip op can sometimes change the type (e.g. EQ and LT) so the new dtype needs to be specified
        pub fn Broadcast(comptime Other: type) type {
            // Gets the broadcast shape between two tensors if one exists
            // If the two tensors do not broadcast, the code won't compile
            if (dtype != Other.dtype) {
                @compileError("Cannot broadcast tensors as they do not have the same dtype, please cast first");
            }
            const bc_ndims = @max(ndims, Other.ndims);
            var bc_shape: [bc_ndims]usize = undefined;
            for (0..bc_ndims) |i| {
                const dim1 = if (i >= ndims) 1 else shape[ndims - i - 1];
                const dim2 = if (i >= Other.ndims) 1 else Other.shape[Other.ndims - i - 1];
                if (dim1 != 1 and dim2 != 1 and dim1 != dim2) {
                    @compileError(comptimePrint(
                        "Cannot broadcast tensors of shapes {any} and {any}",
                        .{ shape, Other.shape },
                    ));
                }
                bc_shape[bc_ndims - i - 1] = if (dim1 == dim2 or dim2 == 1) dim1 else dim2;
            }
            return Tensor(dtype, bc_shape);
        }

        pub fn Reduce(comptime dim: ?u8) type {
            if (dim == null) {
                return Tensor(dtype, [_]usize{1} ** ndims);
            }
            if (dim.? >= ndims) {
                @compileError(comptimePrint(
                    "Reduce dim {d} is out of bounds for tensor with ndims={d} ",
                    .{ dim.?, ndims },
                ));
            }
            var reduced_shape: [ndims]usize = undefined;
            @memcpy(&reduced_shape, &shape);
            reduced_shape[dim.?] = 1;
            return Tensor(dtype, reduced_shape);
        }

        // We can add the tensor functions using "pub usingnamespace"
        // That way the tensor struct definition is cleaner
        pub usingnamespace @import("functions.zig");
    };
}
