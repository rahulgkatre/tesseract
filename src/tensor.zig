const std = @import("std");
const comptimePrint = std.fmt.comptimePrint;
const anytensor = @import("anytensor.zig").anytensor;
const utils = @import("utils.zig");
const ops = @import("ops.zig");
const dtypes = @import("dtypes.zig");
const Record = @import("record.zig").Record;

pub fn AsTensor(comptime val: anytype) type {
    if (isTensor(@TypeOf(val))) {
        return Tensor(val.dtype, val.ndims, val.shape[0..val.ndims].*);
    } else {
        return Scalar(dtypes.inferDType(val));
    }
}

pub fn range(
    comptime start: comptime_int,
    comptime stop: comptime_int,
) tensor(.i32, .{stop - start}) {
    return tensor(.i32, .{stop - start}){ .record = &Record.init(.InitOp, .Range, {}, .{ .Range = .{
        .start = std.fmt.comptimePrint("{d}", .{start}),
        .stop = std.fmt.comptimePrint("{d}", .{stop}),
    } }) };
}

fn Scalar(comptime dtype: dtypes.DType) type {
    return tensor(dtype, .{1});
}

pub fn isTensor(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Pointer => |ptr| Tensor(ptr.child.dtype, ptr.child.ndims, ptr.child.shape) == T,
        .Struct => Tensor(T.dtype, T.ndims, T.shape) == T,
        else => false,
    };
}

/// Used for wrapping immediate values in single size tensors with the same dtype as the current tensor
pub fn asTensor(comptime val: anytype) AsTensor(val) {
    return if (!isTensor(@TypeOf(val))) AsTensor(val).full(val) else val;
}

pub fn randLike(comptime other: anytype) @TypeOf(other) {
    std.debug.assert(isTensor(@TypeOf(other)));
    return @TypeOf(other).rand();
}

pub fn fullLike(comptime other: anytype, value: dtypes.ZigType(other.dtype)) @TypeOf(other) {
    std.debug.assert(isTensor(@TypeOf(other)));
    return @TypeOf(other).full(value);
}

pub fn tensor(comptime dtype: dtypes.DType, comptime shape: anytype) type {
    return Tensor(dtype, shape.len, shape);
}

pub fn Tensor(
    // Generic parameters are private so they will be redeclare as pub conts in the result type
    comptime tensor_dtype: dtypes.DType,
    comptime tensor_ndims: u8,
    comptime tensor_shape: [tensor_ndims]u64,
) type {
    return extern struct {
        const Self = @This();
        const shaper = @import("zig").Shaper(Self);

        // All the functions for operations are implemented separately
        pub usingnamespace @import("functions.zig").Functions(Self);

        // Type level constants for comptime shape logic (e.g. @TypeOfa.ndims)
        pub const dtype: dtypes.DType = tensor_dtype;
        pub const ndims: u8 = tensor_ndims;
        pub const shape: [ndims]u64 = tensor_shape;
        pub const contiguous_strides: [ndims]u64 = utils.contiguousStrides(ndims, shape);
        pub const num_entries = utils.numEntries(ndims, shape);

        dtype: dtypes.DType = dtype,
        ndims: u8 = ndims,
        shape: [*]const u64 = &shape,
        strides: [*]const u64,
        offset: u64,
        record: *const Record = &Record.init(.InitOp, .Empty, {}, .{ .Empty = {} }),
        block_tracker: *const utils.BlockTracker,

        pub fn widen(comptime self: Self) anytensor {
            return @as(*const anytensor, @ptrCast(&self)).*;
        }

        /// Allows for negative dimension indexing to work by normalizing it to ndims
        pub fn normalize(dim: i16) u8 {
            const normalized = if (dim < 0) ndims + dim else dim;
            if (normalized < 0 or normalized > ndims) {
                @compileError(comptimePrint(
                    "Dimension index {d} is out of bounds {d}",
                    .{ normalized, ndims },
                ));
            }
            return @intCast(normalized);
        }

        /// Determine if the stride pattern of the tensor defines a fully contiguous section of memory at runtime
        pub fn isContiguous(self: Self) bool {
            // Strides need to be decreasing unless its a broadcasted stride (0)
            var prev: u64 = std.math.maxInt(u64);
            for (self.strides, 0..self.ndims) |s, _| {
                if (s > 0) {
                    if (s > prev) {
                        return false;
                    }
                    prev = s;
                }
            }
            return true;
        }

        pub fn storage(self: Self) u128 {
            // The storage size is 1 + last index calculated by the strides and shape
            // This is different from size() because there can be elements that are not
            // accessed by the stride pattern but are still contained in the underlying memory
            // An example is padding for alignment
            // Conversely, a convolution using stride tricks will index the same element multiple times
            // but the underlying memory has not increased in size
            // shape[d] - 1 is the last index in dimension d
            // Also incorporate the storage offset
            var _size: u128 = self.offset + 1;
            for (0..ndims) |d| {
                _size += (shape[d] - 1) * self.strides[d];
            }
            // The result is the size of the storage needed to visit all indices of the tensor
            return _size;
        }

        pub fn size(_: Self) u128 {
            return num_entries;
        }

        /// Supports negative indexing sugar (e.g. -1 = ndims - 1)
        pub fn dimsize(_: Self, d: i16) u64 {
            return shape[normalize(d)];
        }

        /// Supports negative indexing sugar (e.g. -1 = ndims - 1)
        pub fn dimstride(self: Self, d: i16) u64 {
            return self.strides[normalize(d)];
        }

        /// This is used to determine which higher level function the tensor is part of
        /// which is useful for finding a better gradient implementation if one exists
        /// than the one that would be found through simple backtracking of the graph.
        /// By default the block is null, so setting the block isn't necessary
        /// for simple functions with trivial gradients (e.g subtraction)
        pub fn startBlock(self: Self, block_name: []const u8) Self {
            // Block will not modify the computation graph so pass through
            // the other tensor fields unmodified
            // @compileLog("Starting block", &self.block_tracker.enterBlock(block_name).next_block.?.name);
            return .{
                .record = self.record,
                .strides = self.strides,
                .offset = self.offset,
                .block_tracker = &self.block_tracker.enterBlock(block_name),
            };
        }

        pub fn joinBlock(self: Self, block: ?*const utils.BlockTracker.Block) Self {
            // @compileLog("Joining block", other_block_tracker.next_block.?.name);
            return .{
                .record = self.record,
                .strides = self.strides,
                .offset = self.offset,
                .block_tracker = &self.block_tracker.joinBlock(block),
            };
        }

        /// End the current block by setting it to the outer block.
        /// Compile error if the current block is null.
        pub fn endBlock(self: *const Self) Self {
            // @compileLog("Exiting block", &self.block_tracker.next_block.?.name);
            return .{
                .record = self.record,
                .strides = self.strides,
                .offset = self.offset,
                .block_tracker = &self.block_tracker.exitBlock(),
            };
        }

        /// Used to mark a tensor as an input to a graph,
        /// codegen will make this an argument of the function
        pub fn input() Self {
            return .{
                .record = &Record.init(
                    .InitOp,
                    .Input,
                    {},
                    .{ .Input = {} },
                ),
                .strides = &contiguous_strides,
                .offset = 0,
                .block_tracker = &utils.BlockTracker{},
            };
        }

        /// Used to mark a tensor as a learnable parameter,
        /// codegen will make this an argument of the function,
        /// gradients can be accumulated for it,
        /// and optimizers can detect it,
        pub fn param() Self {
            return .{
                .record = &Record.init(
                    .InitOp,
                    .Parameter,
                    {},
                    .{ .Parameter = {} },
                ),
                .strides = &contiguous_strides,
                .offset = 0,
                .block_tracker = &utils.BlockTracker{},
            };
        }

        /// Fill a tensor with a value
        /// By default, full tensors will be constant folded in codegen
        /// unless they are marked as requires_grad
        pub fn full(comptime value: dtypes.ZigType(dtype)) Self {
            return .{
                .record = &Record.init(
                    .InitOp,
                    .Full,
                    {},
                    .{ .Full = std.fmt.comptimePrint("{}", .{value}) },
                ),
                .strides = &contiguous_strides,
                .offset = 0,
                .block_tracker = &utils.BlockTracker{},
            };
        }

        /// Fill a tensor with random generated numbers
        /// By default, random tensors will be constant folded in codegen
        /// unless they are marked as requires_grad
        /// Do not use this for random initialization of parameters!
        /// Note that some device backends do not support this
        pub fn random() Self {
            std.debug.assert(dtypes.isFloat(dtype));
            return .{
                .record = &Record.init(
                    .InitOp,
                    .Rand,
                    {},
                    .{ .Rand = {} },
                ),
                .strides = &contiguous_strides,
                .offset = 0,
                .block_tracker = &utils.BlockTracker{},
            };
        }

        pub fn Permute(comptime perm: [ndims]u8) type {
            return Reshape(utils.arrayPermute(u64, ndims, shape, perm));
        }
        /// Permute the dimensions of the tensor. A valid permutation must contain
        /// values from 0 to ndims and each value must appear exactly once.
        pub fn permute(comptime a: Self, comptime perm: [ndims]u8) (Permute(perm)) {
            return a.view(
                Permute(perm).shape,
                utils.arrayPermute(u64, ndims, a.strides[0..ndims].*, perm),
                a.offset,
            );
        }

        pub fn Transpose(comptime dim1: i16, comptime dim2: i16) type {
            const norm1 = normalize(dim1);
            const norm2 = normalize(dim2);
            var new_shape = shape;
            new_shape[norm1] = shape[norm2];
            new_shape[norm2] = shape[norm1];
            return Reshape(new_shape);
        }
        /// Transpose two dimensions of the tensor. Similar to permute, but only for two dimensions.
        pub fn transpose(comptime a: Self, comptime dim1: i16, comptime dim2: i16) (Transpose(dim1, dim2)) {
            const norm1 = normalize(dim1);
            const norm2 = normalize(dim2);
            if (norm1 != norm2) {
                var new_strides = a.strides[0..a.ndims].*;
                new_strides[norm1] = a.strides[norm2];
                new_strides[norm2] = a.strides[norm1];
                return a.view(
                    Transpose(norm1, norm2).shape,
                    new_strides,
                    a.offset,
                );
            } else {
                return a;
            }
        }

        /// Shorthand for transposing rightmost dimensions of tensor
        pub fn T(comptime a: Self) (Transpose(-2, -1)) {
            return a.transpose(-2, -1);
        }

        pub fn Reshape(comptime new_shape: anytype) type {
            return tensor(dtype, new_shape);
        }
        /// Change the shape of the tensor. This changes the type too.
        pub fn reshape(comptime a: Self, comptime new_shape: anytype) (Reshape(new_shape)) {
            if (!isContiguous(a)) {
                return a.contiguous().reshape(new_shape);
            } else {
                return a.view(new_shape, Reshape(new_shape).contiguous_strides, a.offset);
            }
        }
        pub fn contiguous(comptime a: Self) Self {
            return a.id();
        }

        pub fn Flatten(comptime start_dim: i16, comptime end_dim: i16) type {
            const norm_start = normalize(start_dim);
            const norm_end = normalize(end_dim);
            if (norm_start == norm_end) {
                return @This();
            } else {
                var new_shape: [ndims - (norm_end - norm_start)]u64 = undefined;
                new_shape[norm_start] = 1;
                for (0..ndims) |d| {
                    if (d < norm_start or d > norm_end) {
                        new_shape[d] = shape[d];
                    } else {
                        new_shape[norm_start] *= shape[d];
                    }
                }
                return Reshape(new_shape);
            }
        }
        pub fn flatten(comptime a: Self, comptime start_end_dims: struct {
            start_dim: i16 = 0,
            end_dim: i16 = -1,
        }) Flatten(start_end_dims.start_dim, start_end_dims.end_dim) {
            return a.reshape(Flatten(start_end_dims.start_dim, start_end_dims.end_dim).shape);
        }

        pub fn Unsqueeze(comptime dim: i16) type {
            return Reshape(utils.arrayInsert(ndims, shape, normalize(dim), 1));
        }
        /// Insert a dim of size 1 into the shape of the tensor.
        pub fn unsqueeze(comptime a: Self, comptime dim: i16) Unsqueeze(dim) {
            return a.view(
                Unsqueeze(dim).shape,
                utils.arrayInsert(ndims, a.strides[0..ndims].*, normalize(dim), 0),
                a.offset,
            );
        }

        pub fn Squeeze(comptime dim: i16) type {
            if (shape[normalize(dim)] != 1) {
                @compileError("Cannot squeeze as dimension size is not 1");
            }
            return Reshape(utils.arrayDelete(ndims, shape, normalize(dim)));
        }
        /// Remove a dim of size 1 from the shape of the tensor.
        pub fn squeeze(comptime a: Self, comptime dim: i16) Squeeze(dim) {
            return a.view(
                Squeeze(dim).shape,
                utils.arrayDelete(ndims, a.strides[0..ndims].*, normalize(dim)),
                a.offset,
            );
        }

        /// Changes the shape and stride of the tensor to change how the underlying memory is accessed.
        /// Powerful enough to be used to implement any reshaping or windowing operation on a tensor.
        /// There are guardrails to prevent out of bounds access into underlying memory!
        pub fn view(comptime a: Self, comptime new_shape: anytype, comptime new_strides: [new_shape.len]u64, new_offset: u64) (Reshape(new_shape)) {
            var out = Reshape(new_shape){
                .record = &Record.init(.ArrayOp, .View, .{&a.widen()}, {}),
                .strides = &new_strides,
                .offset = new_offset,
                .block_tracker = &a.block_tracker.update(),
            };
            if (out.storage() > a.storage()) {
                @compileError(comptimePrint(
                    \\View indexes elements outside defined storage
                    \\Old shape: {any}
                    \\Old strides: {any}
                    \\Old storage offset: {}
                    \\Old storage size: {}
                    \\New shape: {any}
                    \\New strides: {any}
                    \\New storage offset: {}
                    \\New storage size: {}
                , .{
                    a.shape[0..a.ndims],
                    a.strides[0..a.ndims],
                    a.offset,
                    a.storage(),
                    out.shape[0..out.ndims],
                    out.strides[0..out.ndims],
                    out.offset,
                    out.storage(),
                }));
            }
            return out;
        }

        ///Cast an array of a datatype to another datatype
        pub fn cast(comptime a: Self, comptime new_dtype: dtypes.DType) tensor(new_dtype, shape) {
            if (new_dtype != a.dtype) {
                return .{
                    .record = &Record.init(.ArrayOp, .Cast, .{&a.widen()}, {}),
                    .strides = a.strides,
                    .offset = a.offset,
                    .block_tracker = &a.block_tracker.update(),
                };
            } else {
                return a;
            }
        }

        ///Apply an elementwise unary operation
        pub fn unaryFn(comptime a: Self, comptime op: ops.UnaryOp) Self {
            return .{
                .record = &Record.init(.UnaryOp, op, .{&a.widen()}, {}),
                .strides = a.strides,
                .offset = a.offset,
                .block_tracker = &a.block_tracker.update(),
            };
        }

        /// Expand a tensor along 1 or more dimensions with size 1 and stride 0
        /// The new shape must broadcast with the old shape
        pub fn expand(comptime a: Self, comptime new_shape: anytype) (Broadcast(new_shape.len, new_shape)) {
            const Out = Broadcast(new_shape.len, new_shape);
            if (Self == Out) {
                return a;
            }
            var bc_strides: [new_shape.len]u64 = undefined;
            for (0..new_shape.len) |i| {
                bc_strides[new_shape.len - i - 1] = if (i >= ndims) 0 else a.strides[ndims - i - 1];
            }
            return .{
                .record = &Record.init(
                    .ArrayOp,
                    .Expand,
                    .{&a.widen()},
                    {},
                ),
                .strides = &bc_strides,
                .block_tracker = &a.block_tracker.update(),
            };
        }

        pub fn Broadcast(comptime other_ndims: u8, comptime other_shape: [other_ndims]u64) type {
            if (std.mem.eql(u64, &shape, &other_shape)) {
                return @This();
            }
            const bc_ndims = @max(ndims, other_shape.len);
            var bc_shape: [bc_ndims]u64 = undefined;
            for (0..bc_ndims) |i| {
                const dim1 = if (i >= ndims) 1 else shape[ndims - i - 1];
                const dim2 = if (i >= other_shape.len) 1 else other_shape[other_shape.len - i - 1]; // orelse dim1;
                if (dim1 != 1 and dim2 != 1 and dim1 != dim2) {
                    @compileError(comptimePrint(
                        \\Tensor shapes are not comaptible for broadcasting
                        \\Tensor A shape: {any}
                        \\Tensor B shape: {any}
                    ,
                        .{ shape, other_shape },
                    ));
                }
                bc_shape[bc_ndims - i - 1] = if (dim1 == dim2 or dim2 == 1) dim1 else dim2;
            }
            return Reshape(bc_shape);
        }

        pub fn BinaryFnResult(comptime other: anytype, comptime op: ops.BinaryOp) type {
            const OtherTensor: type = AsTensor(other);
            const bc_shape = Broadcast(OtherTensor.ndims, OtherTensor.shape).shape;
            const new_dtype: dtypes.DType = switch (op) {
                .Eq, .Lt => .bool,
                else => dtypes.resultDType(
                    Self.dtype,
                    OtherTensor.dtype,
                ),
            };
            return tensor(new_dtype, bc_shape);
        }
        /// Apply an elementwise binary operation on two arrays, with broadcasting
        pub fn binaryFn(comptime a: Self, _b: anytype, comptime op: ops.BinaryOp) BinaryFnResult(_b, op) {
            const b = (asTensor(_b)).joinBlock(a.block_tracker.next_block);
            const Result = BinaryFnResult(b, op);
            return Result{
                .record = &Record.init(
                    .BinaryOp,
                    op,
                    .{ &a.widen(), &b.widen() },
                    {},
                ),
                .strides = &Result.contiguous_strides,
                .offset = 0,
                .block_tracker = &a.block_tracker.update(),
            };
        }

        pub fn ReduceFnResult(comptime reduce_dims: anytype) type {
            switch (@typeInfo(@TypeOf(reduce_dims))) {
                .ComptimeInt, .Int => {
                    const dim = normalize(reduce_dims);
                    if (dim < 0 or dim >= ndims) {
                        @compileError("Dimension index for single dimension reduce is out of bounds");
                    }
                    var reduced_shape: [ndims]u64 = undefined;
                    @memcpy(&reduced_shape, &shape);
                    reduced_shape[dim] = 1;
                    return Reshape(reduced_shape);
                },
                .Null, .Void => {
                    return Reshape(.{1});
                },
                else => {
                    const dims = reduce_dims;
                    if (dims.len > ndims) {
                        @compileError("Length of dimension index array for multi dimension reduce is out of bounds");
                    }
                    var reduce_dim_mask: [ndims]bool = [_]bool{false} ** ndims;
                    var reduced_shape: [ndims]u64 = undefined;
                    @memcpy(&reduced_shape, &shape);
                    for (0..dims.len) |d| {
                        const norm = normalize(d);
                        if (reduce_dim_mask[norm]) {
                            @compileError("Cannot reuse dimension index for multi dimensional reduce");
                        }
                        reduce_dim_mask[d] = true;
                        reduced_shape[d] = 1;
                    }
                    return Reshape(reduced_shape);
                },
            }
        }
        /// Perform a reduction across 1 or more (or all) dimensions of a tensor.
        /// Dimensions to reduce can be passed as a int for 1 dim, tuple for multiple dims, or null/void for all dims
        pub fn reduceFn(
            a: Self,
            comptime op: ops.ReduceOp,
            comptime reduce_dims: anytype,
        ) (ReduceFnResult(reduce_dims)) {
            const reduce_dim_mask: [ndims]bool = switch (@typeInfo(@TypeOf(reduce_dims))) {
                .ComptimeInt, .Int => blk: {
                    var tmp_mask: [ndims]bool = [_]bool{false} ** ndims;
                    const dim = reduce_dims;
                    tmp_mask[normalize(dim)] = true;
                    break :blk tmp_mask;
                },
                .Null, .Void => [_]bool{true} ** ndims,
                else => blk: {
                    var tmp_mask: [ndims]bool = [_]bool{false} ** ndims;
                    for (reduce_dims) |dim| {
                        tmp_mask[normalize(dim)] = true;
                    }
                    break :blk tmp_mask;
                },
            };

            const Result = ReduceFnResult(reduce_dims);
            return Result{
                .record = &Record.init(.ReduceOp, op, .{&a.widen()}, &reduce_dim_mask),
                .strides = &Result.contiguous_strides,
                .offset = 0,
                .block_tracker = &a.block_tracker.update(),
            };
        }

        pub fn MatMul(comptime other: anytype) type {
            // Matrix multiplication invariant
            // (n x m1) matmul (m2 x p) -> (n x p) iff m1 = m2
            // otherwise matmul is invalid, compile error
            const n = if (ndims == 1) 1 else shape[ndims - 2];
            const m = shape[ndims - 1];
            const other_m = if (other.ndims == 1) 1 else other.shape[other.ndims - 2];
            const p = other.shape[other.ndims - 1];

            if (m == other_m) {
                const mm_ndims = @max(ndims, other.ndims);
                var mm_shape: [mm_ndims]u64 = undefined;
                // Expanding check, look only at batch dimensions (everything before last 2 dimensions)
                for (0..mm_ndims - 2) |i| {
                    const dim1 = if (i >= ndims - 2) 1 else shape[ndims - i - 3];
                    const dim2 = if (i >= other.ndims - 2) 1 else other.shape[other.ndims - i - 3];
                    if (dim1 == dim2 or dim1 == 1 or dim2 == 1) {
                        mm_shape[mm_ndims - i - 3] = if (dim1 == dim2 or dim2 == 1) dim1 else dim2;
                    } else {
                        @compileError(comptimePrint(
                            \\Tensors cannot be broadcasted for batch matrix multiplication
                            \\Tensor A: {any}
                            \\Tensor B: {any}
                            \\Shapes incompatible at dimension {d}
                        , .{ shape, other.shape, i }));
                    }
                }
                mm_shape[mm_ndims - 2] = n;
                mm_shape[mm_ndims - 1] = p;
                return Reshape(mm_shape);
            }
            @compileError(comptimePrint(
                \\Tensors have incompatible shapes for batch matrix multiplication
                \\Tensor A: {any}
                \\Tensor B: {any}
                \\n = {d}, m = {d}, other_m = {d}, p = {d}
            , .{ shape, @TypeOf(other).shape, n, m, other_m, p }));
        }
        pub fn matmul(comptime _a: Self, comptime _b: anytype) MatMul(_b) {
            const a = _a.startBlock("matmul");
            const b = _b.joinBlock(a.block_tracker.next_block);
            return a.unsqueeze(a.ndims - 1)
                .mul(b.transpose(b.ndims - 2, b.ndims - 1).unsqueeze(b.ndims - 2))
                .sum(a.ndims)
                .squeeze(a.ndims)
                .endBlock();
        }

        pub fn Where(comptime true_value: anytype, comptime false_value: anytype) type {
            const true_tensor = asTensor(true_value);
            const false_tensor = asTensor(false_value);
            std.debug.assert(true_tensor.dtype == false_tensor.dtype);
            std.debug.assert(dtypes.isBool(Self.dtype));
            const True = @TypeOf(true_tensor);
            const False = @TypeOf(false_tensor);
            const Bc = True.Broadcast(False.ndims, False.shape);
            return tensor(Bc.dtype, Broadcast(Bc.ndims, Bc.shape).shape);
        }
        /// Conditional elementwise operator
        /// out[i] = if (mask[i]) true_value[i] else false_value[i]
        /// Supports broadcasting between all 3 tensors, but true value and false value are broadcasted together first and must also have the same dtype
        pub fn where(mask: Self, true_value: anytype, false_value: anytype) (Where(true_value, false_value)) {
            const Out = Where(true_value, false_value);
            const mask_expand = mask.expand(Out.shape);
            const true_expand = asTensor(true_value).joinBlock(mask.block_tracker.next_block).expand(Out.shape);
            const false_expand = asTensor(false_value).joinBlock(mask.block_tracker.next_block).expand(Out.shape);
            return .{
                .record = &Record.init(.TernaryOp, .Where, .{ &mask_expand.widen(), &true_expand.widen(), &false_expand.widen() }, {}),
                .strides = mask.strides,
                .offset = mask.offset,
                .block_tracker = &mask.block_tracker.update(),
            };
        }

        // TODO: Need to implement padding to get conv2d to work
        // Might want to practice with Conv1d first
        // pub fn Conv2d(comptime Filter: type, _stride: anytype, _) type {
        //     const stride: [2]u64 = switch (@typeInfo(@TypeOf(_stride)) {
        //         .ComptimeInt, .Int => [2]u64{ _stride, _stride },
        //         .Array => blk: {
        //             if (_stride.len != 2) {
        //                 @compileError("2D convolution stride must be a 2 element tuple");
        //             }
        //             break :blk _stride;
        //         },
        //         else => {
        //             @compileError("2D convolution stride must be 1 number of a 2 element tuple");
        //         },
        //     };

        //     if (ndims == 4) {
        //         const batch_size = shape[0];
        //         const in_channels = shape[1];
        //         const in_height = shape[2];
        //         const in_width = shape[3];

        //         const out_height = @divFloor(in_height + 2 * , denominator: T)
        //     }

        // }
    };
}

test "same tensors assignable" {
    // This test catches regressions caused by comptime slices with the same values not being
    // equal to teach other, which would cause this test to not compile
    // Note that the fill value is different: this should have no effect
    comptime {
        const tensor1 = tensor(.i32, .{ 2, 3, 4 }).full(0);
        var tensor2 = tensor(.i32, .{ 2, 3, 4 }).full(1);
        var tensor3 = tensor(tensor2.dtype, tensor2.shape[0..tensor2.ndims].*).full(2);
        tensor2 = tensor1;
        tensor3 = tensor2;
    }
}

test "permute" {
    const tensor1 = comptime tensor(.i32, .{ 2, 3, 4 }).full(0);
    const tensor2 = comptime tensor1.permute(.{ 0, 2, 1 });
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 4, 3 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 12, 1, 4 }, tensor2.strides[0..tensor2.ndims]);
}

test "view" {
    const tensor1 = comptime tensor(.i32, .{ 2, 3, 4 }).full(0);
    const tensor2 = comptime tensor1.reshape(.{ 12, 2 });
    const tensor3 = comptime tensor2.reshape(.{24});
    try std.testing.expectEqualSlices(u64, &[_]u64{ 12, 2 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 1 }, tensor2.strides[0..tensor2.ndims]);
    try std.testing.expectEqualSlices(u64, &[_]u64{24}, tensor3.shape[0..tensor3.ndims]);
    try std.testing.expectEqualSlices(u64, &[_]u64{1}, tensor3.strides[0..tensor3.ndims]);
}

test "as strided" {
    // Based on example from https://pytorch.org/docs/stable/generated/torch.as_strided.html
    const tensor1 = comptime tensor(.i32, .{ 3, 3 }).full(0);
    const tensor2 = comptime tensor1.view(.{ 2, 2 }, .{ 1, 2 }, 0);

    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 2 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expectEqual(false, tensor2.isContiguous());

    const test_indices = [_][2]u64{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 } };
    const expected_flat_indices1 = &[_]u64{ 0, 2, 1, 3 };
    for (expected_flat_indices1, test_indices) |expected_flat_i, test_i| {
        try std.testing.expectEqual(expected_flat_i, utils.ravelMultiIndex(tensor2.ndims, tensor2.strides[0..tensor2.ndims].*, tensor2.offset, test_i));
    }

    const tensor3 = comptime tensor1.view(.{ 2, 2 }, .{ 1, 2 }, 1);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 2 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expectEqual(false, tensor2.isContiguous());

    const expected_flat_indices2 = &[_]u64{ 1, 3, 2, 4 };
    for (expected_flat_indices2, test_indices) |expected_flat_i, test_i| {
        try std.testing.expectEqual(expected_flat_i, utils.ravelMultiIndex(tensor3.ndims, tensor3.strides[0..tensor3.ndims].*, tensor3.offset, test_i));
    }
}

test "unaryFn" {
    const tensor1 = comptime tensor(.i32, .{ 2, 3, 4 }).full(3);
    const tensor2 = comptime tensor1.neg();
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 4 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expect(tensor2.record.UnaryOp.op == .Neg);
    try std.testing.expectEqual(tensor2.record.UnaryOp.a.narrow().*, tensor1);
}

test "binaryFn" {
    const tensor1 = comptime tensor(.i32, .{ 2, 1, 4 }).full(2);
    const tensor2 = comptime tensor(.i32, .{ 3, 1 }).full(3);
    const tensor3 = comptime tensor1.add(tensor2);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 4 }, tensor3.shape[0..tensor3.ndims]);
    try std.testing.expect(tensor3.record.BinaryOp.op == .Add);
    try std.testing.expectEqual(tensor3.record.BinaryOp.a.narrow().*, tensor1);
    try std.testing.expectEqual(tensor3.record.BinaryOp.b.narrow().*, tensor2);
}

test "reduce" {
    const tensor1 = comptime tensor(.i32, .{ 2, 3, 4 }).full(5);
    const tensor2 = comptime tensor1.sum(1);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 1, 4 }, tensor2.shape[0..tensor1.ndims]);
    try std.testing.expect(tensor2.record.ReduceOp.op == .Add);
    try std.testing.expectEqual(tensor2.record.ReduceOp.a.narrow().*, tensor1);
    try std.testing.expectEqual(tensor2.record.ReduceOp.dims[0..tensor2.ndims].*, ([_]bool{ false, true, false }));
}

test "multiple dim reduce" {
    const tensor1 = comptime tensor(.i32, .{ 2, 3, 4 }).full(5);
    const tensor2 = comptime tensor1.sum(.{ 0, 1 });
    try std.testing.expectEqualSlices(u64, &[_]u64{ 1, 1, 4 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expect(tensor2.record.ReduceOp.op == .Add);
    try std.testing.expectEqual(tensor2.record.ReduceOp.a.narrow().*, tensor1);
    try std.testing.expectEqualDeep(tensor2.record.ReduceOp.dims[0..tensor2.ndims], &[_]bool{ true, true, false });
}

test "binaryFn reduce" {
    const tensor1 = comptime tensor(.i32, .{ 2, 1, 4 }).full(2);
    const tensor2 = comptime tensor(.i32, .{ 2, 3, 1 }).full(3);
    const tensor3 = comptime tensor1.add(tensor2).sum(1);
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 1, 4 }, tensor3.shape[0..tensor3.ndims]);
    try std.testing.expect(tensor3.record.ReduceOp.op == .Add);
    // Anonymous intermediate tensor that stores tensor1 + tensor2
    const anon = tensor3.record.ReduceOp.a;
    try std.testing.expectEqual(anon.record.BinaryOp.a.narrow().*, tensor1);
    try std.testing.expectEqual(anon.record.BinaryOp.b.narrow().*, tensor2);
}

test "as_type" {
    const tensor1 = comptime tensor(.bool, .{3}).full(true);
    try std.testing.expect(tensor1.dtype == .bool);
    const tensor2 = comptime tensor1.cast(.i32);
    try std.testing.expect(tensor2.dtype == .i32);
    const tensor3 = comptime tensor2.cast(.i8);
    try std.testing.expect(tensor3.dtype == .i8);
    const tensor4 = comptime tensor3.cast(.f16);
    try std.testing.expect(tensor4.dtype == .f16);
    const tensor5 = comptime tensor4.cast(.f32);
    try std.testing.expect(tensor5.dtype == .f32);
}

fn fn1() tensor(.i32, .{ 2, 1, 4 }) {
    const tensor1 = tensor(.i32, .{ 2, 1, 4 }).full(1);
    const tensor2 = tensor(.i32, .{ 2, 3, 1 }).full(2);
    const tensor3 = tensor1.add(tensor2).sum(1);
    return tensor3;
}

fn fn2(input: anytype) tensor(.i32, .{ 2, 3, 4 }) {
    return comptime blk: {
        const tensor4 = tensor(.i32, .{ 2, 1, 4 }).full(4);
        const tensor5 = tensor(.i32, .{ 2, 3, 1 }).full(5);
        const tensor6 = tensor4.mul(input).sum(1).add(tensor5);
        break :blk tensor6;
    };
}

test "tensors from functions" {
    _ = comptime blk: {
        const tensor3 = fn1();
        const tensor6 = fn2(tensor3);
        break :blk tensor6;
    };
}

test "transpose" {
    const tensor1 = comptime tensor(.i32, .{ 2, 1, 4 }).full(1);
    const tensor2 = comptime tensor1.T();
    const ndims = tensor1.ndims;
    try std.testing.expectEqualDeep(tensor2, comptime tensor1.transpose(-2, -1));
    try std.testing.expectEqualDeep(tensor1.shape[0..ndims], comptime tensor2.T().shape[0..ndims]);
    try std.testing.expectEqualDeep(tensor1.strides[0..ndims], comptime tensor2.T().strides[0..ndims]);
}

test "runtime" {
    const tensor1 = comptime tensor(.i32, .{ 2, 1, 4 }).full(1);
    const tensor2 = comptime tensor1.T();
    _ = tensor2;
}
