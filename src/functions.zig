const ops = @import("ops.zig");
const tensor = @import("tensor.zig");

// Higher order functions
pub inline fn map(x_ptr: anytype, op: ops.MapOp) @TypeOf(x_ptr.*) {
    return x_ptr.backend.map(op, x_ptr);
}

pub inline fn zip(a_ptr: anytype, op: ops.ZipOp, b_ptr: anytype) @TypeOf(a_ptr.*).Broadcast(@TypeOf(b_ptr.*)) {
    return a_ptr.backend.zip(op, a_ptr, b_ptr);
}

pub inline fn reduce(x_ptr: anytype, op: ops.ReduceOp, comptime dim: ?u8) @TypeOf(x_ptr.*).Reduce(dim) {
    return x_ptr.backend.reduce(op, x_ptr, dim);
}

fn FuncType(comptime op: ops.Op) type {
    return @TypeOf(switch (op) {
        .MapOp => struct {
            inline fn mapFn(self: anytype) @TypeOf(self.*) {
                unreachable;
            }
        }.mapFn,
        .ZipOp => struct {
            inline fn zipFn(self: anytype, other: anytype) @TypeOf(self.*).Broadcast(@TypeOf(other)) {
                unreachable;
            }
        }.zipFn,
        .ReduceOp => struct {
            inline fn reduceFn(self: anytype, comptime dim: ?u8) @TypeOf(self.*).Reduce(dim) {
                unreachable;
            }
        }.reduceFn,
    });
}

inline fn Func(comptime op: ops.Op) FuncType(op) {
    return switch (op) {
        .MapOp => struct {
            inline fn f(self: anytype) @TypeOf(self.*) {
                return map(self, op.MapOp);
            }
        },
        .ZipOp => struct {
            inline fn f(self: anytype, other: anytype) @TypeOf(self.*).Broadcast(@TypeOf(other)) {
                return zip(self, op.ZipOp, &other);
            }
        },
        .ReduceOp => struct {
            inline fn f(self: anytype, comptime dim: ?u8) @TypeOf(self.*).Reduce(dim) {
                return reduce(self, op.ReduceOp, dim);
            }
        },
    }.f;
}

// KEEPING THIS HERE FOR NOW
// Pre-defined User Functions
inline fn mapZipFunc(comptime map_op: ops.MapOp, comptime zip_op: ops.ZipOp) FuncType(.{ .ZipOp = zip_op }) {
    return struct {
        inline fn f(a_ptr: anytype, b: anytype) @TypeOf(a_ptr.*).Broadcast(@TypeOf(b)) {
            return zip(&map(a_ptr, map_op), zip_op, b);
        }
    }.f;
}

inline fn zipMapFunc(comptime map_op: ops.MapOp, comptime zip_op: ops.ZipOp) FuncType(.{ .MapOp = map_op }) {
    return struct {
        inline fn f(a_ptr: anytype, b: anytype) @TypeOf(a_ptr.*).Broadcast(@TypeOf(b)) {
            return map(&zip(a_ptr, zip_op, b), map_op);
        }
    }.f;
}

inline fn composeFunc(comptime map_op_f: ops.MapOp, comptime map_op_g: ops.MapOp) FuncType(.{ .MapOp = map_op_f }) {
    return struct {
        inline fn f(a_ptr: anytype) @TypeOf(a_ptr.*) {
            return map(&map(a_ptr, map_op_g), map_op_f);
        }
    }.f;
}

// This is where all the actual functions of a tensor are
// TODO: Make each of these functions differentiable

pub const exp2 = Func(.{ .MapOp = .Exp2 });
pub const log2 = Func(.{ .MapOp = .Log2 });

pub inline fn exp(self: anytype) @TypeOf(self.*) {
    // 1 / log(2) = 1.44269504089
    // e^x = 2^(x / log(2))
    return self.mul(tensor.constant(self.backend, @TypeOf(self.*).dtype, 1.44269504089)).exp2();
}

pub inline fn ln(comptime self: anytype) @TypeOf(self.*) {
    // log(2) = 0.69314718056
    // log(x) = log2(x)log(2)
    return self.log2().mul(tensor.constant(self.backend, self.dtype, 0.69314718056));
}

pub const neg = Func(.{ .MapOp = .Neg });
pub const add = Func(.{ .ZipOp = .Add });
pub const mul = Func(.{ .ZipOp = .Mul });
pub const sum = Func(.{ .ReduceOp = .Sum });
pub const max = Func(.{ .ReduceOp = .Max });
pub const recip = Func(.{ .MapOp = .Recip });

pub inline fn div(self: anytype, other: anytype) @TypeOf(self.*).Broadcast(@TypeOf(other)) {
    return self.mul(other.recip());
}

pub inline fn sub(self: anytype, other: anytype) @TypeOf(self.*).Broadcast(@TypeOf(other)) {
    return self.add(other.neg());
}
