const std = @import("std");
const Allocator = std.mem.Allocator;
const comptimePrint = std.fmt.comptimePrint;
const utils = @import("utils.zig");
const ops = @import("ops.zig");
const Backend = @import("backend.zig").Backend;

const InitType = enum { Input, Constant, Result };

pub fn Constant(comptime dtype: type) type {
    return Tensor(dtype, .{1});
}

pub fn Tensor(comptime dtype: type, comptime shape: anytype) type {
    return DefaultStridedTensor(dtype, shape);
}

pub fn Range(backend: *const Backend, comptime dtype: type, comptime start: dtype, comptime stop: dtype) Tensor(dtype, .{stop - start}) {
    const data: [stop - start]dtype = std.simd.iota(dtype, stop - start) + @as(@Vector(stop - start, dtype), @splat(start));
    return Tensor(dtype, .{stop - start}).fromData(backend, @constCast(data[0..]));
}

fn DefaultStridedTensor(comptime dtype: type, comptime shape: anytype) type {
    return AsStrided(dtype, shape, utils.stridesFromShape(shape));
}

fn AsStrided(comptime dtype: type, comptime shape: anytype, comptime strides: anytype) type {
    if (shape.len + 1 != strides.len) {
        @compileError("Provided shape ndims not compatible with provided strides ndims");
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
            "Tensor({any},{any})",
            .{ dtype, shape },
        );

        ndims: u8 = ndims,
        shape: [ndims]usize = shape,
        size: usize = size,
        strides: [ndims + 1]usize = strides,
        str: @TypeOf(str) = str,
        // init_type: InitType,
        backend: *const Backend,
        storage: ?*Backend.Storage(dtype),

        // Callbacks during recursive traversal of compute graph
        evalFn: *const fn (self: *Self) Self,
        loadDataFn: *const fn (self: *Self) void,
        graphFn: *const fn (self: *const Self) void,

        fn init(
            backend: *const Backend,
            // init_type: InitType,
            storage: ?*Backend.Storage(dtype),
            comptime evalFn: ?*const fn (self: *Self) Self,
            comptime loadDataFn: ?*const fn (self: *Self) void,
            comptime graphFn: ?*const fn (self: *const Self) void,
        ) Self {
            const Impl = struct {
                fn eval(self: *Self) Self {
                    self.initStorage(null);
                    return self.*;
                }
                fn graph(self: *const Self) void {
                    if (@inComptime()) {
                        @compileLog(comptimePrint("{s} = {s}", .{ self.str, self.str }));
                    } else {
                        std.debug.print("{s}@{d} = {s}\n", .{ self.str, @intFromPtr(self), self.str });
                    }
                }
                fn loadData(_: *Self) void {}
            };
            return .{
                // .init_type = init_type,
                .backend = backend,
                .storage = storage, // orelse backend.storage(dtype, size),
                .evalFn = evalFn orelse Impl.eval,
                .loadDataFn = loadDataFn orelse Impl.loadData,
                .graphFn = graphFn orelse Impl.graph,
            };
        }

        // Exposed functions for initializing
        pub fn input(backend: *const Backend, storage: ?*Backend.Storage(dtype)) Self {
            return init(
                backend,
                // .Input,
                storage,
                null,
                null,
                null,
            );
        }

        pub fn fromData(backend: *const Backend, data: anytype) Self {
            const Impl = struct {
                fn eval(self: *Self) Self {
                    self.initStorage(null);
                    return self.*;
                }
                fn graph(self: *const Self) void {
                    if (@inComptime()) {
                        @compileLog(comptimePrint("{s} = {s}", .{ self.str, self.str }));
                    } else {
                        std.debug.print("{s}@{d} = {s}\n", .{ self.str, @intFromPtr(self), self.str });
                    }
                }
                fn loadData(self: *Self) void {
                    self.storage.?.load(@as([]dtype, data));
                }
            };
            return .{
                // .init_type = init_type,
                .backend = backend,
                .storage = null, // orelse backend.storage(dtype, size),
                .evalFn = Impl.eval,
                .loadDataFn = Impl.loadData,
                .graphFn = Impl.graph,
            };
        }

        // TODO: A tensor of constants does not need to have a real shape as broadcasting is always possible
        // While a tensor with specific size is useful for things like tables, a ConstantTensor with size 1 will be better
        pub fn full(backend: *const Backend, comptime value: dtype) Self {
            const Impl = struct {
                const val: dtype = value;
                fn graph(self: *const Self) void {
                    if (@inComptime()) {
                        @compileLog(comptimePrint("{s} = Full({d}) {s}", .{ self.str, val, self.str }));
                    } else {
                        std.debug.print("{s}@{d} = Full({any}) {s}\n", .{ self.str, @intFromPtr(self), val, self.str });
                    }
                }
                fn loadData(self: *Self) void {
                    self.storage.?.fill(val);
                }
            };
            return init(
                backend,
                // .Constant,
                null,
                null,
                Impl.loadData,
                Impl.graph,
            );
        }
        pub fn result(
            backend: *const Backend,
            storage: ?*Backend.Storage(dtype),
            comptime evalFn: ?*const fn (self: *Self) Self,
            comptime graphFn: ?*const fn (self: *const Self) void,
        ) Self {
            return init(
                backend,
                // .Result,
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
        pub fn initStorage(self: *Self, comptime data: ?[size]dtype) void {
            if (self.storage == null) {
                self.storage = self.backend.storage(dtype, size, data);
                self.loadDataFn(self);
            }
        }
        pub fn isContiguous(_: *const Self) bool {
            return comptime utils.isContiguous(ndims, strides);
        }
        pub inline fn broadcastIndex(_: anytype, bc_index: anytype) [ndims]usize {
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
        pub inline fn flattenIndex(_: anytype, index: [ndims]usize) usize {
            const index_vec: @Vector(ndims, usize) = index;
            const strides_vec: @Vector(ndims, usize) = strides[0..ndims].*;
            return @reduce(.Add, index_vec * strides_vec) + strides[ndims];
        }
        pub inline fn unflattenIndex(_: anytype, flat_index: usize) [ndims]usize {
            var index: [ndims]usize = undefined;
            // Subtract storage offset first
            var remainder = flat_index - strides[ndims];
            inline for (0..ndims) |d| {
                index[d] = @divTrunc(remainder, strides[d]);
                remainder = @mod(remainder, strides[d]);
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
            const Output = Permute(perm);
            const Impl = struct {
                fn eval(ptr: *Output) Output {
                    const self_eval = self.eval();
                    ptr.storage = self_eval.storage;
                    return ptr.*;
                }
                fn graph(ptr: *const Output) void {
                    self.graph();
                    if (@inComptime()) {
                        @compileLog(comptimePrint("{s} = Permute({any}) {s}", .{ ptr.str, perm, self.str }));
                    } else {
                        std.debug.print("{s}@{d} = Permute({any}) {s}@{d}\n", .{ ptr.str, @intFromPtr(ptr), perm, self.str, @intFromPtr(self) });
                    }
                }
            };

            return Output.result(self.backend, self.storage, Impl.eval, Impl.graph);
        }
        pub fn view(self: *const Self, comptime new_shape: anytype) Tensor(dtype, new_shape) {
            const Output = Tensor(dtype, new_shape);
            if (self.isContiguous()) {
                const Impl = struct {
                    fn eval(ptr: *Output) Output {
                        const self_eval = self.eval();
                        ptr.storage = self_eval.storage;
                        return ptr.*;
                    }
                    fn graph(ptr: *const Output) void {
                        self.graph();
                        if (@inComptime()) {
                            @compileLog(comptimePrint("{s} = View({any}) {s}", .{ ptr.str, new_shape, self.str }));
                        } else {
                            std.debug.print("{s}@{d} = View({any}) {s}@{d}\n", .{ ptr.str, @intFromPtr(ptr), new_shape, self.str, @intFromPtr(self) });
                        }
                    }
                };
                return Output.result(self.backend, self.storage, Impl.eval, Impl.graph);
            } else {
                @compileError("Must be contiguous to view");
            }
        }

        pub fn asStrided(self: *const Self, comptime new_shape: anytype, comptime new_strides: anytype) AsStrided(dtype, new_shape, new_strides) {
            return AsStrided(dtype, new_shape, new_strides).result(self.backend, self.storage, null, null);
        }

        pub fn AsType(comptime new_dtype: type) type {
            return BaseTensor(new_dtype, ndims, shape, strides);
        }
        pub fn asType(self: *const Self, comptime new_dtype: type) AsType(new_dtype) {
            return self.backend.asType(new_dtype, self.*);
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
            return Tensor(dtype, bc_shape);
        }

        pub fn Reduce(comptime dim: ?u8) type {
            if (dim == null) {
                return DefaultStridedTensor(dtype, [_]usize{1} ** ndims);
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
