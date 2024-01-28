const std = @import("std");
const Allocator = std.mem.Allocator;
const comptimePrint = std.fmt.comptimePrint;
const utils = @import("utils.zig");
const ops = @import("ops.zig");
const Backend = @import("backend.zig").Backend;

pub var debug = true;

// TensorArena provides an allocator for the tensor metadata
// No actual elements of the tensor are stored by this allocator
pub const TensorArena = struct {
    var global_arena: ?std.heap.ArenaAllocator = null;
    var runtime_tensor_cache: std.AutoHashMap(usize, usize) = undefined;
    fn init(arena: std.heap.ArenaAllocator) void {
        global_arena = arena;
        runtime_tensor_cache = std.AutoHashMap(usize, usize).init(global_arena.?.allocator());
    }
    fn deinit() void {
        global_arena.?.deinit();
        global_arena = null;
    }
    fn allocator() std.mem.Allocator {
        return global_arena.?.allocator();
    }
    fn convert_to_runtime(comptime tensor_type: type, comptime_tensor_ptr: anytype, id: usize) *tensor_type {
        const runtime_tensor_ptr = runtime_tensor_cache.get(@intFromPtr(comptime_tensor_ptr));
        if (runtime_tensor_ptr != null) {
            return @ptrFromInt(runtime_tensor_ptr.?);
        }
        const runtime_tensor: *tensor_type = TensorArena.allocator().create(tensor_type) catch unreachable;
        if (debug) {
            std.debug.print("Comptime tensor {s}@{d} -> Runtime tensor {s}@{d}\n", .{ comptime_tensor_ptr.str, @intFromPtr(comptime_tensor_ptr), comptime_tensor_ptr.str, @intFromPtr(runtime_tensor) });
        }
        runtime_tensor.* = .{
            .id = id,
            .backend = comptime_tensor_ptr.backend,
            .storage = comptime_tensor_ptr.backend.storage(tensor_type.dtype, tensor_type.size),
            .evalFn = comptime_tensor_ptr.evalFn,
        };
        runtime_tensor_cache.putNoClobber(@intFromPtr(comptime_tensor_ptr), @intFromPtr(runtime_tensor)) catch unreachable;
        return runtime_tensor;
    }
};

pub fn runtime() void {
    TensorArena.init(std.heap.ArenaAllocator.init(std.heap.page_allocator));
}

pub fn finished() void {
    TensorArena.deinit();
}

pub fn constant(backend: *const Backend, comptime dtype: type, comptime value: dtype) Tensor(dtype, .{1}) {
    return Tensor(dtype, .{1}).full(backend, value);
}

pub fn range(backend: *const Backend, comptime dtype: type, comptime start: dtype, comptime stop: dtype) Tensor(dtype, .{stop - start}) {
    @setEvalBranchQuota(@as(u32, 2 * stop));
    const data: [stop - start]dtype = std.simd.iota(dtype, stop - start) + @as(@Vector(stop - start, dtype), @splat(start));
    return Tensor(dtype, .{stop - start}).fromData(backend, data[0..]);
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
        pub const str = comptimePrint(
            "Tensor{{{any},{any}}}",
            .{ dtype, shape },
        );

        id: ?usize = null,
        evaluated: bool = false,
        ndims: u8 = ndims,
        shape: [ndims]usize = shape,
        size: usize = size,
        strides: [ndims + 1]usize = strides,
        str: @TypeOf(str) = str,
        backend: *const Backend,
        storage: ?*Backend.Storage(dtype),

        // Callbacks for recursive traversal of compute graph
        evalFn: *const fn (self: *Self) *Self,

        fn init(
            backend: *const Backend,
            storage: ?*Backend.Storage(dtype),
            comptime evalFn: ?*const fn (self: *Self) *Self,
        ) Self {
            const impl = struct {
                fn eval(self: *Self) *Self {
                    if (!self.evaluated) {
                        if (debug) {
                            std.debug.print("t{d} = {s}\n", .{ self.id.?, self.str });
                        }
                    }
                    self.evaluated = true;
                    return self;
                }
            };
            return .{
                .backend = backend,
                .storage = storage,
                .evalFn = evalFn orelse impl.eval,
            };
        }

        // Load the tensor's data from an array pointer
        // Not a slice because this guarantees that the size requirement is met and verified in comptime
        pub fn fromData(backend: *const Backend, data: *const [size]dtype) Self {
            const impl = struct {
                fn eval(self: *Self) *Self {
                    if (!self.evaluated) {
                        self.storage.?.load(data);
                        if (debug) {
                            std.debug.print("t{d} = FromData {s}\n", .{ self.id.?, self.str });
                        }
                    }
                    self.evaluated = true;
                    return self;
                }
            };
            return init(backend, null, impl.eval);
        }

        // Fill a tensor with a value
        pub fn full(backend: *const Backend, comptime value: dtype) Self {
            const impl = struct {
                fn eval(self: *Self) *Self {
                    if (!self.evaluated) {
                        self.storage.?.fill(value);
                        if (debug) {
                            std.debug.print("t{d} = Full({any}) {s}\n", .{ self.id.?, value, self.str });
                        }
                    }
                    self.evaluated = true;
                    return self;
                }
            };
            return init(backend, null, impl.eval);
        }

        // Utility function for initializing a tensor which is the result of an operation
        // It will share the same backend but has callbacks provided by the backend
        // that link it to the rest of the computation graph
        pub fn result(
            backend: *const Backend,
            storage: ?*Backend.Storage(dtype),
            comptime evalFn: ?*const fn (self: *Self) *Self,
        ) Self {
            return init(backend, storage, evalFn);
        }

        // TODO: Might not be necessary if codegen is the only way to run the tensor code
        pub fn runtime(self: *const Self, id: usize) *Self {
            return TensorArena.convert_to_runtime(Self, self, id);
        }

        pub fn eval(comptime self: *const Self) *Self {
            return self.evalFn(self.runtime(0));
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
                fn eval(out: *Out) *Out {
                    if (!out.evaluated) {
                        const self_eval = self.eval();
                        out.storage = self_eval.storage;
                        out.evaluated = true;
                        if (debug) {
                            std.debug.print("t{d} = Permute({any}) t{d}\n", .{ out.id.?, perm, self_eval.id.? });
                        }
                    }
                    return out;
                }
            };
            return Out.result(self.backend, null, impl.eval);
        }
        pub fn view(self: *const Self, comptime new_shape: anytype) Tensor(dtype, new_shape) {
            const Out = Tensor(dtype, new_shape);
            std.debug.assert(Out.size == size);
            if (self.isContiguous()) {
                const impl = struct {
                    fn eval(out: *Out) *Out {
                        if (!out.evaluated) {
                            const self_eval = self.eval();
                            out.storage = self_eval.storage;
                            out.evaluated = true;
                            if (debug) {
                                std.debug.print("t{d} = View({any}) t{d}\n", .{ out.id.?, new_shape, self_eval.id.? });
                            }
                        }
                        return out;
                    }
                };
                return Out.result(self.backend, null, impl.eval);
            } else {
                @compileError("Must be contiguous to view");
            }
        }

        pub fn asStrided(self: *const Self, comptime new_shape: anytype, comptime new_strides: anytype) AsStrided(dtype, new_shape, new_strides) {
            return AsStrided(dtype, new_shape, new_strides).result(self.backend, self.storage, null);
        }

        pub fn Cast(comptime new_dtype: type) type {
            return TensorView(new_dtype, ndims, shape, strides);
        }
        pub fn cast(self: *const Self, comptime new_dtype: type) Cast(new_dtype) {
            return self.backend.cast(new_dtype, self);
        }

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
            // Broadcasting can sometimes change the type so the new dtype needs to be specified
            return Tensor(dtype, bc_shape);
        }

        pub fn Reduce(comptime dim: ?u8) type {
            if (dim == null) {
                return Tensor(dtype, [_]usize{1} ** ndims);
            }
            if (dim.? >= ndims) {
                @compileError(comptimePrint(
                    "Reduce dim {d} is out of bounds for tensor {s} with ndims={d} ",
                    .{ dim.?, str, ndims },
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
