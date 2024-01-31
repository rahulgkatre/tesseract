const std = @import("std");
const Allocator = std.mem.Allocator;
const comptimePrint = std.fmt.comptimePrint;
const utils = @import("utils.zig");
const ops = @import("ops.zig");
const Graph = @import("Graph.zig");

pub fn constant(comptime dtype: type, comptime value: dtype) Tensor(dtype, .{1}) {
    return Tensor(dtype, .{1}).full(value);
}

pub fn range(comptime dtype: type, comptime start: dtype, comptime stop: dtype) Tensor(dtype, .{stop - start}) {
    @setEvalBranchQuota(@as(u32, 2 * stop));
    const data: [stop - start]dtype = std.simd.iota(dtype, stop - start) + @as(@Vector(stop - start, dtype), @splat(start));
    return Tensor(dtype, .{stop - start}).from(data[0..]);
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
        pub fn from(data: *const [size]dtype) Self {
            _ = data;
            const traceFn = struct {
                fn trace(self: *const Self) void {
                    Graph.Node.new(self, .{ .InitOp = .{ .op = .From } }, Self);
                }
            }.trace;
            return init(traceFn);
        }

        // Fill a tensor with a value
        pub fn full(comptime value: dtype) Self {
            _ = value;
            const traceFn = struct {
                fn trace(self: *const Self) void {
                    Graph.Node.new(self, .{ .InitOp = .{ .op = .Full } }, Self);
                }
            }.trace;
            return init(traceFn);
        }

        // Fill a tensor with a value
        pub fn rand(comptime value: dtype) Self {
            _ = value;
            const traceFn = struct {
                fn trace(self: *const Self) void {
                    Graph.Node.new(self, .{ .InitOp = .{ .op = .Rand } }, Self);
                }
            }.trace;
            return init(traceFn);
        }

        pub fn trace(comptime self: *const Self) void {
            self.traceFn(self);
        }

        pub fn isContiguous(_: *const Self) bool {
            return comptime utils.isContiguous(ndims, strides);
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
        pub fn permute(self: *const Self, comptime perm: [ndims]u8) permute(perm) {
            const Out = Permute(perm);
            const traceFn = struct {
                fn trace(out: *const Out) void {
                    self.trace();
                    Graph.Node.new(out, .{ .TypeOp = .{ .op = .Permute, .x = Graph.Node.get(self) } }, Out);
                }
            }.trace;
            return Out.init(traceFn);
        }

        pub fn view(self: *const Self, comptime new_shape: anytype) Tensor(dtype, new_shape) {
            const Out = Tensor(dtype, new_shape);
            std.debug.assert(Out.size == size);
            if (self.isContiguous()) {
                const traceFn = struct {
                    fn trace(out: *const Out) void {
                        self.trace();
                        Graph.Node.new(out, .{ .TypeOp = .{ .op = .View, .x = Graph.Node.get(self) } }, Out);
                    }
                }.trace;
                return Out.init(traceFn);
            } else {
                @compileError("Must be contiguous to view");
            }
        }

        pub fn as_strided(self: *const Self, comptime new_shape: anytype, comptime new_strides: anytype) AsStrided(dtype, new_shape, new_strides) {
            const Out = AsStrided(dtype, new_shape, new_strides);
            const traceFn = struct {
                fn trace(out: *const Out) void {
                    self.trace();
                    Graph.Node.new(out, .{ .TypeOp = .{ .op = .AsStrided, .x = Graph.Node.get(self) } }, Out);
                }
            }.trace;
            return Out.init(traceFn);
        }

        pub fn AsType(comptime new_dtype: type) type {
            return TensorView(new_dtype, ndims, shape, strides);
        }
        pub fn as_type(self: *const Self, comptime new_dtype: type) AsType(new_dtype) {
            const Out: type = AsType(new_dtype);
            const traceFn = struct {
                fn trace(out: *const Out) void {
                    self.trace();
                    Graph.Node.new(out, .{ .TypeOp = .{ .op = .AsType, .x = Graph.Node.get(self) } }, Out);
                }
            }.trace;
            return Out.init(traceFn);
        }

        pub fn map(x: *const Self, comptime op: ops.MapOp) Self {
            const Out: type = @TypeOf(x.*);
            const traceFn = struct {
                fn trace(out: *const Out) void {
                    x.trace();
                    Graph.Node.new(out, .{ .MapOp = .{ .op = op, .x = Graph.Node.get(x) } }, Out);
                }
            }.trace;
            return Out.init(traceFn);
        }
        pub fn exp2(x: *const Self) Self {
            return x.map(.Exp2);
        }
        pub fn log2(x: *const Self) Self {
            return x.map(.Log2);
        }
        pub fn neg(x: *const Self) Self {
            return x.map(.Neg);
        }
        pub fn recip(x: *const Self) Self {
            return x.map(.Recip);
        }
        pub fn sin(x: *const Self) Self {
            return x.map(.Sin);
        }
        pub fn sqrt(x: *const Self) Self {
            return x.map(.Sqrt);
        }

        pub fn Broadcast(comptime OtherTensorType: type, comptime new_dtype: type) type {
            // Gets the broadcast shape between two tensors if one exists
            // If the two tensors do not broadcast, the code won't compile
            if (dtype != OtherTensorType.dtype) {
                @compileError("Cannot broadcast tensors as they do not have the same dtype, please cast first");
            }
            const bc_ndims = @max(ndims, OtherTensorType.ndims);
            var bc_shape: [bc_ndims]usize = undefined;
            for (0..bc_ndims) |i| {
                const dim1 = if (i >= ndims) 1 else shape[ndims - i - 1];
                const dim2 = if (i >= OtherTensorType.ndims) 1 else OtherTensorType.shape[OtherTensorType.ndims - i - 1];
                if (dim1 != 1 and dim2 != 1 and dim1 != dim2) {
                    @compileError(comptimePrint(
                        "Cannot broadcast tensors of shapes {any} and {any}",
                        .{ shape, OtherTensorType.shape },
                    ));
                }
                bc_shape[bc_ndims - i - 1] = if (dim1 == dim2 or dim2 == 1) dim1 else dim2;
            }
            return Tensor(new_dtype, bc_shape);
        }
        fn ZipNewDtype(comptime op: ops.ZipOp) type {
            return switch (op) {
                .Equals, .LessThan => bool,
                else => dtype,
            };
        }
        pub fn zip(a: *const Self, comptime op: ops.ZipOp, b: anytype) Broadcast(@TypeOf(b), ZipNewDtype(op)) {
            const Out: type = Broadcast(@TypeOf(b), ZipNewDtype(op));
            const traceFn = struct {
                fn trace(out: *const Out) void {
                    a.trace();
                    b.trace();
                    Graph.Node.new(out, .{ .ZipOp = .{ .op = op, .a = Graph.Node.get(a), .b = Graph.Node.get(&b) } }, Out);
                }
            }.trace;
            return Out.init(traceFn);
        }
        // ZipOps
        pub fn add(a: *const Self, b: anytype) Broadcast(@TypeOf(b), dtype) {
            return a.zip(.Add, b);
        }
        pub fn mul(a: *const Self, b: anytype) Broadcast(@TypeOf(b), dtype) {
            return a.zip(.Mul, b);
        }
        pub fn maximum(a: *const Self, b: anytype) Broadcast(@TypeOf(b), dtype) {
            return a.zip(.Maximum, b);
        }
        pub fn mod(a: *const Self, b: anytype) Broadcast(@TypeOf(b)) {
            return a.zip(.Mod, b);
        }
        pub fn less_than(a: *const Self, b: anytype) Broadcast(@TypeOf(b), bool) {
            return a.zip(.LessThan, b);
        }
        pub fn equals(a: *const Self, b: anytype) Broadcast(@TypeOf(b), bool) {
            return a.zip(.Equals, b);
        }
        pub fn xor(a: *const Self, b: anytype) Broadcast(@TypeOf(b), dtype) {
            return a.zip(.Xor, b);
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
        pub fn reduce(x: *const Self, comptime op: ops.ReduceOp, comptime dim: ?u8) Reduce(dim) {
            const Out: type = Reduce(dim);
            const traceFn = struct {
                fn trace(out: *const Out) void {
                    x.trace();
                    Graph.Node.new(out, .{ .ReduceOp = .{ .op = op, .x = Graph.Node.get(x), .dim = dim } }, Out);
                }
            }.trace;
            return Out.init(traceFn);
        }

        // ReduceOps
        pub fn sum(x: *const Self, comptime dim: ?u8) Reduce(dim) {
            return x.reduce(.Sum, dim);
        }
        pub fn max(x: *const Self, comptime dim: ?u8) Reduce(dim) {
            return x.reduce(.Max, dim);
        }

        // Compound functions that use the ops
        pub fn exp(x: *const Self) Self {
            // 1 / log(2) = 1.44269504089
            // e^x = 2^(x / log(2))
            return x.mul(constant(dtype, 1.44269504089)).exp2();
        }
        pub fn ln(x: *const Self) Self {
            // log(2) = 0.69314718056
            // log(x) = log2(x)log(2)
            return x.log2().mul(constant(dtype, 0.69314718056));
        }
        pub fn div(a: *const Self, b: anytype) Broadcast(@TypeOf(b), dtype) {
            return a.mul(b.recip());
        }
        pub fn sub(a: *const Self, b: anytype) Broadcast(@TypeOf(b), dtype) {
            return a.add(b.neg());
        }
    };
}
