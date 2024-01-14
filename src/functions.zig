const ops = @import("ops.zig");
const tensor = @import("tensor.zig");

pub fn asType(x_ptr: anytype, comptime dtype: type) tensor.CastedTensor(@TypeOf(x_ptr.*), dtype) {
    return x_ptr.backend.asType(dtype, x_ptr.*);
}

// Higher order functions
pub fn map(x: anytype, op: ops.MapOp) @TypeOf(x) {
    return x.backend.map(op, x);
}

pub fn zip(a: anytype, op: ops.ZipOp, b: anytype) tensor.BroadcastedTensor(@TypeOf(a), @TypeOf(b)) {
    return a.backend.zip(op, a, b);
}

pub fn reduce(x: anytype, op: ops.ReduceOp, comptime dim: ?u8) tensor.ReducedTensor(@TypeOf(x), dim) {
    return x.backend.reduce(op, x, dim);
}

fn FuncType(comptime op: ops.Op) type {
    return @TypeOf(switch (op) {
        .MapOp => struct {
            fn f(x_ptr: anytype) @TypeOf(x_ptr.*) {
                @panic("Not implemented");
            }
        },
        .ZipOp => struct {
            fn f(a_ptr: anytype, b: anytype) tensor.BroadcastedTensor(@TypeOf(a_ptr.*), @TypeOf(b)) {
                @panic("Not implemented");
            }
        },
        .ReduceOp => struct {
            fn f(x_ptr: anytype, comptime dim: ?u8) tensor.ReducedTensor(@TypeOf(x_ptr.*), dim) {
                @panic("Not implemented");
            }
        },
        else => @compileError("Operation cannot run lazily"),
    }.f);
}

fn Func(comptime op: ops.Op) FuncType(op) {
    return switch (op) {
        .MapOp => struct {
            fn f(x_ptr: anytype) @TypeOf(x_ptr.*) {
                return map(x_ptr.*, op.MapOp);
            }
        },
        .ZipOp => struct {
            fn f(a_ptr: anytype, b: anytype) tensor.BroadcastedTensor(@TypeOf(a_ptr.*), @TypeOf(b)) {
                return zip(a_ptr.*, op.ZipOp, b);
            }
        },
        .ReduceOp => struct {
            fn f(x_ptr: anytype, comptime dim: ?u8) tensor.ReducedTensor(@TypeOf(x_ptr.*), dim) {
                return reduce(x_ptr.*, op.ReduceOp, dim);
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

pub fn div(a_ptr: anytype, b: anytype) tensor.BroadcastedTensor(@TypeOf(a_ptr.*), @TypeOf(b)) {
    return a_ptr.mul(b.recip());
}

pub fn sub(a_ptr: anytype, b: anytype) tensor.BroadcastedTensor(@TypeOf(a_ptr.*), @TypeOf(b)) {
    return a_ptr.add(b.neg());
}

// KEEPING THIS HERE FOR NOW
// Pre-defined User Functions
fn mapZipFunc(comptime map_op: ops.MapOp, comptime zip_op: ops.ZipOp) FuncType(.{ .ZipOp = zip_op }) {
    return struct {
        fn f(a_ptr: anytype, b: anytype) tensor.BroadcastedTensor(@TypeOf(a_ptr.*), @TypeOf(b)) {
            return zip(&map(a_ptr, map_op), zip_op, b);
        }
    }.f;
}

fn zipMapFunc(comptime map_op: ops.MapOp, comptime zip_op: ops.ZipOp) FuncType(.{ .MapOp = map_op }) {
    return struct {
        fn f(a_ptr: anytype, b: anytype) tensor.BroadcastedTensor(@TypeOf(a_ptr.*), @TypeOf(b)) {
            return map(&zip(a_ptr, zip_op, b), map_op);
        }
    }.f;
}

fn composeFunc(comptime map_op_f: ops.MapOp, comptime map_op_g: ops.MapOp) FuncType(.{ .MapOp = map_op_f }) {
    return struct {
        fn f(a_ptr: anytype) tensor.BroadcastedTensor(@TypeOf(a_ptr.*)) {
            return map(&map(a_ptr, map_op_g), map_op_f);
        }
    }.f;
}
