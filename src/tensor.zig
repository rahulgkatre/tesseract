const std = @import("std");
const Allocator = std.mem.Allocator;
const comptimePrint = std.fmt.comptimePrint;
const utils = @import("utils.zig");
const ops = @import("ops.zig");
const Backend = @import("backend.zig").Backend;
const LazyBuffer = @import("buffer.zig").LazyBuffer;

pub fn Tensor(comptime dtype: type, comptime shape: anytype) type {
    return DefaultStridedTensor(dtype, shape.len, shape);
}

pub fn StridedTensor(comptime dtype: type, comptime shape: anytype, comptime strides: anytype) type {
    if (shape.len != strides.len) @compileError("Provided shape ndims != provided strides ndims");
    return BaseTensor(dtype, shape.len, shape, strides);
}

fn BaseTensor(comptime _dtype: type, comptime _ndims: u8, comptime _shape: [_ndims]usize, comptime _strides: [_ndims]usize) type {
    return struct {
        const Self = @This();
        pub const dtype: type = _dtype;
        pub const ndims: u8 = _ndims;
        pub const shape: [ndims]usize = _shape;
        pub const strides: [ndims]usize = _strides;
        pub const size = utils.bufferSizeForTensor(ndims, shape, strides);
        pub const str = comptimePrint(
            "Tensor({any},.{any})",
            .{ dtype, shape },
        );
        ndims: u8 = ndims,
        shape: [ndims]usize = shape,
        size: usize = size,
        strides: [ndims]usize = strides,
        str: @TypeOf(str) = str,

        backend: *const Backend,
        buffer: ?*LazyBuffer(dtype),
        eval_fn: *const fn (self: *const Self) void,
        pub fn init(backend: *const Backend) Self {
            return .{
                .backend = backend,
                .buffer = null,
                .eval_fn = struct {
                    // TODO: The default eval_fn must check if the tensor is initialiazed and panic if it is not
                    fn eval(self: *const Self) void {
                        if (!@inComptime()) {
                            std.debug.print(
                                "\n{s}@{d} = {s}.init()",
                                .{ self.str, @intFromPtr(self), self.str },
                            );
                        } else {
                            @compileLog(comptimePrint(
                                "{s} = {s}.init()",
                                .{ self.str, self.str },
                            ));
                        }
                    }
                }.eval,
            };
        }
        pub fn eval(self: *const Self) void {
            self.eval_fn(self);
        }
        // TODO: Don't deinit individual tensors, the backend should deinit everything it allocated
        // pub fn deinit(self: *const Self) void {
        //     _ = self;
        // }
        pub inline fn isContiguous(_: anytype) bool {
            return comptime utils.isContiguous(ndims, strides);
        }
        pub fn broadcastIndex(comptime bc_ndims: u8, bc_index: [bc_ndims]usize) [ndims]usize {
            // Determine the index in the current tensor given an index in the broadcasted tensor
            // If the current tensor has size of 1 in a dimension, then the index must be 0
            // Otherwise it will be what the broadcasted index is
            const index: [ndims]usize = undefined;
            inline for (0..ndims) |d| {
                index[bc_ndims - d - 1] = if (shape[ndims - d - 1] == 1) 0 else bc_index[bc_ndims - d - 1];
            }
            return index;
        }
        pub fn flattenIndex(index: [ndims]usize) usize {
            // Convert a multidimensional index into a single dimensional index
            var flat_index: usize = 0;
            for (0..ndims) |d| {
                flat_index += index[d] * strides[d];
            }
            return flat_index;
        }
        pub fn unflattenIndex(flat_index: usize) [ndims]usize {
            var index: [ndims]usize = undefined;
            var remainder = flat_index;
            for (0..ndims) |d| {
                index[d] = @divTrunc(remainder, strides[d]);
                remainder = @mod(remainder, strides[d]);
            }
            return index;
        }
        pub fn permute(self: *const Self, comptime perm: [ndims]u8) PermutedTensor(Self, perm) {
            return PermutedTensor(Self, perm).init(self.backend);
        }
        pub fn map(self: *const Self, op: ops.MapOp) Self {
            return self.backend.lazy_map(op, self.*);
        }
        pub fn zip(self: *const Self, op: ops.ZipOp, other: anytype) BroadcastedTensor(Self, @TypeOf(other)) {
            return self.backend.lazy_zip(op, self.*, other);
        }
        pub fn reduce(self: *const Self, op: ops.ReduceOp, comptime reduce_dim: usize) ReducedTensor(Self, reduce_dim) {
            return self.backend.lazy_reduce(op, self.*, reduce_dim);
        }
        // We can add the tensor functions using "pub usingnamespace"
        // That way the tensor struct definition is cleaner
    };
}

fn DefaultStridedTensor(comptime dtype: type, comptime ndims: u8, comptime shape: [ndims]usize) type {
    var stride: usize = undefined;
    var offset: usize = 1;
    var strides: [ndims]usize = undefined;
    for (0..ndims - 1) |i| {
        stride = shape[ndims - i - 1] * offset;
        strides[ndims - i - 2] = stride;
        offset = stride;
    }
    strides[ndims - 1] = 1;
    return StridedTensor(dtype, shape, strides);
}

pub fn ReducedTensor(comptime tensor_t: type, comptime reduce_dim: u8) type {
    const ndims = @field(tensor_t, "ndims");
    const shape = @field(tensor_t, "shape");
    if (reduce_dim >= ndims) {
        @compileError(comptimePrint(
            "Reduce dim {d} is out of bounds for tensor {s} with ndims={d} ",
            .{
                reduce_dim,
                @field(tensor_t, "str"),
                ndims,
            },
        ));
    }
    var reduced_shape: [ndims]usize = undefined;
    @memcpy(&reduced_shape, &shape);
    reduced_shape[reduce_dim] = 1;
    return DefaultStridedTensor(
        @field(tensor_t, "dtype"),
        ndims,
        reduced_shape,
    );
}

pub fn BroadcastedTensor(comptime tensor1_t: type, comptime tensor2_t: type) type {
    // Gets the broadcast shape between two tensors if one exists
    // If the two tensors do not broadcast, the code won't compile
    const tensor1_dtype = @field(tensor1_t, "dtype");
    const tensor2_dtype = @field(tensor2_t, "dtype");
    if (tensor1_dtype != tensor2_dtype) {
        @compileError("Cannot broadcast tensors as they do not have the same dtype, please cast first");
    }

    const tensor1_ndims = @field(tensor1_t, "ndims");
    const tensor1_shape = @field(tensor1_t, "shape");
    const tensor2_ndims = @field(tensor2_t, "ndims");
    const tensor2_shape = @field(tensor2_t, "shape");

    const bc_ndims = @max(tensor1_ndims, tensor2_ndims);
    var bc_shape: [bc_ndims]usize = undefined;
    var dim1: usize = undefined;
    var dim2: usize = undefined;
    inline for (0..bc_ndims) |i| {
        dim1 = if (i >= tensor1_ndims) 1 else tensor1_shape[tensor1_ndims - i - 1];
        dim2 = if (i >= tensor2_ndims) 1 else tensor2_shape[tensor2_ndims - i - 1];
        if (dim1 != 1 and dim2 != 1 and dim1 != dim2) {
            @compileError(comptimePrint(
                "Cannot broadcast tensors of shapes {any} and {any}",
                .{ tensor1_shape, tensor2_shape },
            ));
        }
        bc_shape[bc_ndims - i - 1] = if (dim1 == dim2 or dim2 == 1) dim1 else dim2;
    }
    return Tensor(tensor1_dtype, bc_shape);
}

pub fn PermutedTensor(comptime tensor_t: type, comptime perm: [@field(tensor_t, "ndims")]u8) type {
    const ndims = @field(tensor_t, "ndims");
    const shape = @field(tensor_t, "shape");
    const strides = @field(tensor_t, "strides");
    return StridedTensor(
        @field(tensor_t, "dtype"),
        utils.permuteArray(ndims, shape, perm),
        utils.permuteArray(ndims, strides, perm),
    );
}
