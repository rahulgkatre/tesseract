const ops = @import("ops.zig");
const tensor = @import("tensor.zig");

// Higher order functions
pub fn map(x_ptr: anytype, op: ops.MapOp) @TypeOf(x_ptr.*) {
    return x_ptr.backend.map(op, x_ptr.*);
}

pub fn zip(a_ptr: anytype, op: ops.ZipOp, b: anytype) tensor.BroadcastedTensor(@TypeOf(a_ptr.*), @TypeOf(b)) {
    return a_ptr.backend.zip(op, a_ptr.*, b);
}

pub fn reduce(x_ptr: anytype, op: ops.ReduceOp, comptime dim: ?u8) tensor.ReducedTensor(@TypeOf(x_ptr.*), dim) {
    return x_ptr.backend.reduce(op, x_ptr.*, dim);
}

fn FuncType(comptime op: ops.Op) type {
    return @TypeOf(switch (op) {
        .MapOp => struct {
            fn f(x_ptr: anytype) @TypeOf(x_ptr.*) {
                return map(x_ptr, op.MapOp);
            }
        },
        .ZipOp => struct {
            fn f(a_ptr: anytype, b: anytype) tensor.BroadcastedTensor(@TypeOf(a_ptr.*), @TypeOf(b)) {
                return zip(a_ptr, op.ZipOp, b);
            }
        },
        .ReduceOp => struct {
            fn f(x_ptr: anytype, comptime dim: ?u8) tensor.ReducedTensor(@TypeOf(x_ptr.*), dim) {
                return reduce(x_ptr, op.ReduceOp, dim);
            }
        },
        else => @compileError("Operation cannot run lazily"),
    }.f);
}

fn MapZipFunc(comptime map_op: ops.MapOp, comptime zip_op: ops.ZipOp) FuncType(.{ .ZipOp = zip_op }) {
    return struct {
        fn f(a_ptr: anytype, b: anytype) tensor.BroadcastedTensor(@TypeOf(a_ptr.*), @TypeOf(b)) {
            return zip(&map(a_ptr, map_op), zip_op, b);
        }
    }.f;
}

fn Func(comptime op: ops.Op) FuncType(op) {
    return switch (op) {
        .MapOp => struct {
            fn f(x_ptr: anytype) @TypeOf(x_ptr.*) {
                return map(x_ptr, op.MapOp);
            }
        },
        .ZipOp => struct {
            fn f(a_ptr: anytype, b: anytype) tensor.BroadcastedTensor(@TypeOf(a_ptr.*), @TypeOf(b)) {
                return zip(a_ptr, op.ZipOp, b);
            }
        },
        .ReduceOp => struct {
            fn f(x_ptr: anytype, comptime dim: ?u8) tensor.ReducedTensor(@TypeOf(x_ptr.*), dim) {
                return reduce(x_ptr, op.ReduceOp, dim);
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
// pub const div = MapZipFunc(.Recip, .Mul);
pub const recip = Func(.{ .MapOp = .Recip });
