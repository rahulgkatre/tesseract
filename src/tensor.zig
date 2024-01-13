const std = @import("std");
const Allocator = std.mem.Allocator;
const comptimePrint = std.fmt.comptimePrint;
const utils = @import("utils.zig");
const ops = @import("ops.zig");
const Backend = @import("backend.zig").Backend;

const InitType = enum { Input, Constant, Result };

pub fn ConstantTensor(comptime dtype: type) type {
    return Tensor(dtype, .{1});
}

pub fn Tensor(comptime dtype: type, comptime shape: anytype) type {
    return DefaultStridedTensor(dtype, shape);
}

fn DefaultStridedTensor(comptime dtype: type, comptime shape: anytype) type {
    return StridedTensor(dtype, shape, utils.stridesFromShape(shape));
}

pub fn StridedTensor(comptime dtype: type, comptime shape: anytype, comptime strides: anytype) type {
    if (shape.len + 1 != strides.len) {
        @compileError("Provided shape ndims not compatible provided strides ndims");
    }
    return BaseTensor(dtype, shape.len, shape, strides);
}

pub fn BaseTensor(comptime _dtype: type, comptime _ndims: u8, comptime _shape: [_ndims]usize, comptime _strides: [_ndims + 1]usize) type {
    return struct {
        const Self = @This();
        pub const dtype: type = _dtype;
        pub const ndims: u8 = _ndims;
        pub const shape: [ndims]usize = _shape;
        pub const strides: [ndims + 1]usize = _strides;
        pub const size = utils.storageSizeForTensor(ndims, shape, strides);
        pub const str = comptimePrint(
            "Tensor({any},.{any})",
            .{ dtype, shape },
        );
        ndims: u8 = ndims,
        shape: [ndims]usize = shape,
        size: usize = size,
        strides: [ndims + 1]usize = strides,
        str: @TypeOf(str) = str,
        init_type: InitType,
        backend: *const Backend,
        storage: Backend.Storage(dtype),

        // Callbacks during recursive traversal of compute graph
        evalFn: *const fn (self: *Self) Self,
        loadDataFn: *const fn (self: *Self) void,
        graphFn: *const fn (self: *const Self) void,

        fn init(
            backend: *const Backend,
            init_type: InitType,
            storage: ?Backend.Storage(dtype),
            comptime evalFn: ?*const fn (self: *Self) Self,
            comptime loadDataFn: ?*const fn (self: *Self) void,
            comptime graphFn: ?*const fn (self: *const Self) void,
        ) Self {
            const Impl = struct {
                fn eval(self: *Self) Self {
                    self.initStorage();
                    return self.*;
                }
                fn graph(self: *const Self) void {
                    if (@inComptime()) {
                        @compileLog(comptimePrint("{s} := {s} {s}", .{ self.str, @tagName(self.init_type), self.str }));
                    } else {
                        std.debug.print("{s}@{d} := {s} {s}\n", .{ self.str, @intFromPtr(self), @tagName(self.init_type), self.str });
                    }
                }
                fn loadData(_: *Self) void {}
            };
            return .{
                .init_type = init_type,
                .backend = backend,
                .storage = storage orelse backend.storage(dtype, size),
                .evalFn = evalFn orelse Impl.eval,
                .loadDataFn = loadDataFn orelse Impl.loadData,
                .graphFn = graphFn orelse Impl.graph,
            };
        }

        // Exposed functions for initializing
        pub fn input(backend: *const Backend, storage: ?Backend.Storage(dtype)) Self {
            return init(
                backend,
                .Input,
                storage,
                null,
                null,
                null,
            );
        }
        // TODO: A tensor of constants does not need to have a real shape as broadcasting is always possible
        // While a tensor with specific size is useful for things like tables, a ConstantTensor with size 1 will be better
        pub fn constant(backend: *const Backend, comptime value: dtype) Self {
            const funcs = struct {
                const val: dtype = value;
                fn graph(self: *const Self) void {
                    if (@inComptime()) {
                        @compileLog(comptimePrint("{s} := {s}({}) {s}", .{ self.str, @tagName(self.init_type), val, self.str }));
                    } else {
                        std.debug.print("{s}@{d} := {s}({}) {s}\n", .{ self.str, @intFromPtr(self), @tagName(self.init_type), val, self.str });
                    }
                }
                fn loadData(self: *Self) void {
                    self.storage.fill(val);
                }
            };
            return init(
                backend,
                .Constant,
                null,
                null,
                funcs.loadData,
                funcs.graph,
            );
        }
        pub fn result(
            backend: *const Backend,
            storage: ?Backend.Storage(dtype),
            comptime evalFn: ?*const fn (self: *Self) Self,
            comptime graphFn: ?*const fn (self: *const Self) void,
        ) Self {
            return init(
                backend,
                .Result,
                storage,
                evalFn,
                null,
                graphFn,
            );
        }

        pub fn eval(self: *const Self) Self {
            return self.evalFn(@constCast(self));
        }
        pub fn graph(self: *const Self) void {
            self.graphFn(self);
        }
        pub fn initStorage(self: *Self) void {
            self.storage.init();
            self.loadDataFn(self);
        }
        pub fn isContiguous(_: *const Self) bool {
            return comptime utils.isContiguous(ndims, strides);
        }
        pub fn broadcastIndex(_: anytype, bc_index: anytype) [ndims]usize {
            // Determine the index in the current tensor given an index in the broadcasted tensor
            // If the current tensor has size of 1 in a dimension, then the index must be 0
            // Otherwise it will be what the broadcasted index is
            const bc_ndims = bc_index.len;
            var index: [ndims]usize = undefined;
            for (0..ndims) |d| {
                index[ndims - d - 1] = if (shape[ndims - d - 1] == 1) 0 else bc_index[bc_ndims - d - 1];
            }
            return index;
        }
        pub fn flattenIndex(_: anytype, index: [ndims]usize) usize {
            // Convert a multidimensional index into a single dimensional index
            // Start by adding the storage offset
            var flat_index: usize = strides[ndims];
            for (0..ndims) |d| {
                flat_index += index[d] * strides[d];
            }
            return flat_index;
            // return @reduce(.Sum, @mulAdd([ndims]usize, index, strides, [_]usize{0} ** ndims));
        }
        pub fn unflattenIndex(_: anytype, flat_index: usize) [ndims]usize {
            var index: [ndims]usize = undefined;
            // Subtract storage offset first
            var remainder = flat_index - strides[ndims];
            for (0..ndims) |d| {
                index[d] = @divTrunc(remainder, strides[d]);
                remainder = @mod(remainder, strides[d]);
            }
            return index;
        }
        pub fn permute(self: *const Self, comptime perm: [ndims]u8) PermutedTensor(Self, perm) {
            return PermutedTensor(Self, perm).result(self.backend, self.storage, null, null);
        }
        pub fn view(self: *const Self, comptime new_shape: anytype) Tensor(dtype, new_shape) {
            if (self.isContiguous()) {
                return Tensor(dtype, new_shape).result(self.backend, self.storage, null, null);
            } else {
                @compileError("Must be contiguous to view");
            }
        }
        pub fn asStrided(self: *const Self, comptime new_shape: anytype, comptime new_strides: anytype) StridedTensor(dtype, new_shape, new_strides) {
            return StridedTensor(dtype, new_shape, new_strides).result(self.backend, self.storage, null, null);
        }
        // We can add the tensor functions using "pub usingnamespace"
        // That way the tensor struct definition is cleaner
        pub usingnamespace @import("functions.zig");
    };
}

pub fn ReducedTensor(comptime tensor_t: type, comptime dim: ?u8) type {
    const ndims = @field(tensor_t, "ndims");
    const shape = @field(tensor_t, "shape");
    if (dim == null) {
        return DefaultStridedTensor(@field(tensor_t, "dtype"), [_]usize{1} ** ndims);
    }

    if (dim.? >= ndims) {
        @compileError(comptimePrint(
            "Reduce dim {d} is out of bounds for tensor {s} with ndims={d} ",
            .{
                dim.?,
                @field(tensor_t, "str"),
                ndims,
            },
        ));
    }
    var reduced_shape: [ndims]usize = undefined;
    @memcpy(&reduced_shape, &shape);
    reduced_shape[dim.?] = 1;
    return DefaultStridedTensor(
        @field(tensor_t, "dtype"),
        reduced_shape,
    );
}

// TODO: The broadcasted tensor's dtype might be different based on the operation
// It will always be one of the two input tensor's dtypes
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
    for (0..bc_ndims) |i| {
        const dim1 = if (i >= tensor1_ndims) 1 else tensor1_shape[tensor1_ndims - i - 1];
        const dim2 = if (i >= tensor2_ndims) 1 else tensor2_shape[tensor2_ndims - i - 1];
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
    const shape = @field(tensor_t, "shape");
    const strides = @field(tensor_t, "strides");

    var strides_perm: [strides.len]u8 = undefined;
    @memcpy(strides_perm[0 .. strides.len - 1], &perm);
    strides_perm[strides.len - 1] = strides.len - 1;
    return StridedTensor(
        @field(tensor_t, "dtype"),
        utils.permuteArray(shape.len, shape, perm),
        utils.permuteArray(strides.len, strides, strides_perm),
    );
}

pub fn CastedTensor(comptime tensor_t: type, comptime new_dtype: type) type {
    const ndims = @field(tensor_t, "ndims");
    const shape = @field(tensor_t, "shape");
    const strides = @field(tensor_t, "strides");
    return BaseTensor(new_dtype, ndims, shape, strides);
}
