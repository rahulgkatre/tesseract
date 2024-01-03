const std = @import("std");
const Allocator = std.mem.Allocator;
const comptimePrint = std.fmt.comptimePrint;
const utils = @import("utils.zig");
const TensorStorage = @import("storage.zig").TensorStorage;
const ops = @import("ops.zig");

pub fn Tensor(comptime dtype: type, comptime shape: anytype) type {
    // Utility function to create a tensor from an input shape (tuple or array of usize)
    // Most of the time, this is what you want to use
    // Infers strides from shape so it will be contiguous
    // Tensors created using this function must be realized manually as they are endpoints of the compute graph
    return DefaultStridedTensor(dtype, shape.len, shape);
}

pub fn StridedTensor(comptime dtype: type, comptime shape: anytype, comptime strides: anytype) type {
    if (shape.len != strides.len) {
        @compileError("Provided shape != provided strides");
    }
    return BaseTensor(dtype, shape.len, shape, strides);
}

// TODO: Add a device type param here
// Should be easy to add this type param everywhere as the device will remain the same unless a to_device() method is called
fn BaseTensor(comptime _dtype: type, comptime _ndims: u8, comptime _shape: [_ndims]usize, comptime _strides: [_ndims]usize) type {
    return struct {
        const Self = @This();
        // These just take on the value of the generic arguments
        // Save the dtype here as it is needed for some comptime functions, accessed via @field
        pub const dtype: type = _dtype;
        pub const ndims: u8 = _ndims;
        pub const shape: [ndims]usize = _shape;
        pub const strides: [ndims]usize = _strides;
        pub const size = utils.bufferSizeForTensor(ndims, shape, strides);
        ndims: u8 = ndims,
        shape: [ndims]usize = shape,
        strides: [ndims]usize = strides,
        size: usize = size,

        storage: ?*TensorStorage(dtype, size),
        allocator: ?Allocator,
        eval_fn: *const fn (self: *const Self) void,

        pub fn init() Self {
            return .{
                .storage = null,
                .allocator = null,
                .eval_fn = struct {
                    fn eval(self: *const Self) void {
                        std.debug.print("\n{s}@{d} = init()", .{
                            self.info(),
                            @intFromPtr(self),
                        });
                    }
                }.eval,
            };
        }
        pub fn info(_: *const Self) @TypeOf(comptimePrint("tensor<{any}, {any}>", .{ shape, dtype })) {
            return comptimePrint("tensor<{any}, {any}>", .{ shape, dtype });
        }
        pub fn eval(self: *const Self) void {
            self.eval_fn(self);
        }
        pub fn deinit(self: *const Self) void {
            _ = self;
            // if (self.real and self.owns_storage) {
            //     self.storage.?.deinit();
            // }
        }
        pub inline fn isContiguous(_: *const Self) bool {
            // The information for contiguous is in the type itself
            return comptime utils.isContiguous(ndims, strides);
        }
        pub fn permute(comptime _: *const Self, comptime perm: [ndims]u8) PermutedTensor(Self, perm) {
            return PermutedTensor(Self, perm).init();
        }
        pub fn map(self: *const Self, op: ops.MapOp) Self {
            var out = init();
            out.eval_fn = struct {
                fn eval(ptr: *const @TypeOf(out)) void {
                    self.eval();
                    std.debug.print("\n{s}@{d} = {any} {s}@{d} {s}@{d}", .{
                        ptr.info(),
                        @intFromPtr(ptr),
                        op,
                        self.info(),
                        @intFromPtr(self),
                    });
                    return;
                }
            }.eval;
            return out;
        }
        pub fn zip(self: *const Self, op: ops.ZipOp, other: anytype) BroadcastedTensor(Self, @TypeOf(other)) {
            var out = BroadcastedTensor(Self, @TypeOf(other)).init();
            out.eval_fn = struct {
                fn eval(ptr: *const @TypeOf(out)) void {
                    self.eval();
                    other.eval();
                    std.debug.print("\n{s}@{d} = {any} {s}@{d} {s}@{d}", .{
                        ptr.info(),
                        @intFromPtr(ptr),
                        op,
                        self.info(),
                        @intFromPtr(self),
                        other.info(),
                        @intFromPtr(&other),
                    });
                }
            }.eval;
            return out;
        }
        pub fn reduce(self: *const Self, op: ops.ReduceOp, comptime reduce_dim: usize) ReducedTensor(Self, reduce_dim) {
            var out = ReducedTensor(Self, reduce_dim).init();
            out.eval_fn = struct {
                fn eval(ptr: *const @TypeOf(out)) void {
                    self.eval();
                    std.debug.print("\n{s}@{d} = {any} {s}@{d} {d}", .{
                        ptr.info(),
                        @intFromPtr(ptr),
                        op,
                        self.info(),
                        @intFromPtr(self),
                        reduce_dim,
                    });
                }
            }.eval;
            return out;
        }
    };
}

fn DefaultStridedTensor(comptime dtype: type, comptime ndims: u8, comptime shape: [ndims]usize) type {
    return BaseTensor(dtype, ndims, shape, utils.defaultStrides(ndims, shape));
}

fn ReducedTensor(comptime tensor_t: type, comptime reduce_dim: usize) type {
    const dtype = @field(tensor_t, "dtype");
    const ndims = @field(tensor_t, "ndims");
    const shape = utils.reducedShape(ndims, @field(tensor_t, "shape"), reduce_dim);
    return DefaultStridedTensor(dtype, ndims, shape);
}

fn BroadcastedTensor(comptime tensor1_t: type, comptime tensor2_t: type) type {
    return Tensor(@field(tensor1_t, "dtype"), utils.shapeBroadcast(tensor1_t, tensor2_t));
}

fn PermutedTensor(comptime tensor_t: type, comptime perm: [@field(tensor_t, "ndims")]u8) type {
    const dtype = @field(tensor_t, "dtype");
    const ndims = @field(tensor_t, "ndims");
    const shape = @field(tensor_t, "shape");
    const strides = @field(tensor_t, "strides");
    const permute_shape = utils.permuteArray(ndims, shape, perm);
    const permute_strides = utils.permuteArray(ndims, strides, perm);
    return BaseTensor(dtype, ndims, permute_shape, permute_strides);
}
