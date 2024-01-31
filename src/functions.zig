const ops = @import("ops.zig");
const tensor = @import("tensor.zig");
const Graph = @import("Graph.zig");

// Higher order functions
pub inline fn map(x_ptr: anytype, op: ops.MapOp) @TypeOf(x_ptr.*) {
    return Graph.map(op, x_ptr);
}

pub inline fn zip(a_ptr: anytype, op: ops.ZipOp, b_ptr: anytype) @TypeOf(a_ptr.*).Broadcast(@TypeOf(b_ptr.*)) {
    return Graph.zip(op, a_ptr, b_ptr);
}

pub inline fn reduce(x_ptr: anytype, op: ops.ReduceOp, comptime dim: ?u8) @TypeOf(x_ptr.*).Reduce(dim) {
    return Graph.reduce(op, x_ptr, dim);
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
        else => @compileError("This op does not run in runtime"),
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
        else => @compileError("This op does not run in runtime"),
    }.f;
}

// This is where all the actual functions of a tensor are
// TODO: Make each of these functions differentiable

pub const exp2 = Func(.{ .MapOp = .Exp2 });
pub const log2 = Func(.{ .MapOp = .Log2 });

pub inline fn exp(self: anytype) @TypeOf(self.*) {
    // 1 / log(2) = 1.44269504089
    // e^x = 2^(x / log(2))
    return self.mul(tensor.constant(@TypeOf(self.*).dtype, 1.44269504089)).exp2();
}

pub inline fn ln(comptime self: anytype) @TypeOf(self.*) {
    // log(2) = 0.69314718056
    // log(x) = log2(x)log(2)
    return self.log2().mul(tensor.constant(self.dtype, 0.69314718056));
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
