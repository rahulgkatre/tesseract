const ops = @import("ops.zig");
const tensor = @import("tensor.zig");

// Higher order functions
pub inline fn map(x: anytype, op: ops.MapOp) @TypeOf(x) {
    return x.backend.map(op, x);
}

pub inline fn zip(a: anytype, op: ops.ZipOp, b: anytype) @TypeOf(a).Broadcast(@TypeOf(b)) {
    return a.backend.zip(op, a, b);
}

pub inline fn reduce(x: anytype, op: ops.ReduceOp, comptime dim: ?u8) @TypeOf(x).Reduce(dim) {
    return x.backend.reduce(op, x, dim);
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
        else => @compileError("Operation cannot run lazily"),
    });
}

inline fn Func(comptime op: ops.Op) FuncType(op) {
    return switch (op) {
        .MapOp => struct {
            inline fn f(self: anytype) @TypeOf(self.*) {
                return map(self.*, op.MapOp);
            }
        },
        .ZipOp => struct {
            inline fn f(self: anytype, other: anytype) @TypeOf(self.*).Broadcast(@TypeOf(other)) {
                return zip(self.*, op.ZipOp, other);
            }
        },
        .ReduceOp => struct {
            inline fn f(self: anytype, comptime dim: ?u8) @TypeOf(self.*).Reduce(dim) {
                return reduce(self.*, op.ReduceOp, dim);
            }
        },
        else => @compileError("Operation cannot run lazily"),
    }.f;
}

pub const exp2 = Func(.{ .MapOp = .Exp2 });
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

// KEEPING THIS HERE FOR NOW
// Pre-defined User Functions
inline fn mapZipFunc(comptime map_op: ops.MapOp, comptime zip_op: ops.ZipOp) FuncType(.{ .ZipOp = zip_op }) {
    return struct {
        inline fn f(a_ptr: anytype, b: anytype) tensor.BroadcastedTensor(@TypeOf(a_ptr.*), @TypeOf(b)) {
            return zip(&map(a_ptr, map_op), zip_op, b);
        }
    }.f;
}

inline fn zipMapFunc(comptime map_op: ops.MapOp, comptime zip_op: ops.ZipOp) FuncType(.{ .MapOp = map_op }) {
    return struct {
        inline fn f(a_ptr: anytype, b: anytype) tensor.BroadcastedTensor(@TypeOf(a_ptr.*), @TypeOf(b)) {
            return map(&zip(a_ptr, zip_op, b), map_op);
        }
    }.f;
}

inline fn composeFunc(comptime map_op_f: ops.MapOp, comptime map_op_g: ops.MapOp) FuncType(.{ .MapOp = map_op_f }) {
    return struct {
        inline fn f(a_ptr: anytype) tensor.BroadcastedTensor(@TypeOf(a_ptr.*)) {
            return map(&map(a_ptr, map_op_g), map_op_f);
        }
    }.f;
}
