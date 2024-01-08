pub const MapOp = enum { Neg, Log2, Exp2, Sqrt, Recip };
pub const ZipOp = enum { Add, Mul, Maximum, Mod, Lt, Eq, Xor };
pub const ReduceOp = enum { Sum, Max };
pub const TypeOp = enum { Reshape, Permute, Expand, Pad, Shrink, Stride, AsStrided, AsType };
pub const OpTypes = enum { MapOp, ZipOp, ReduceOp, TypeOp };
pub const Op = union(OpTypes) { MapOp: MapOp, ZipOp: ZipOp, ReduceOp: ReduceOp, TypeOp: TypeOp };

fn ScalarMapOpReturnType(comptime map_op: MapOp, comptime x: anytype) type {
    return switch (map_op) {
        .Neg => @TypeOf(x), // Neg can apply to any numeric type (or boolean)
        else => @TypeOf(x + 0.0), // Other
    };
}

fn scalarMapOpEval(comptime map_op: MapOp, x: anytype) ScalarMapOpReturnType(map_op, x) {
    return comptime switch (map_op) {
        .Neg => if (@typeInfo(@TypeOf(x)) == .Bool) !x else -x,
        .Log2 => @log2(x + 0.0),
        .Exp2 => @exp2(x + 0.0),
        .Sqrt => @sqrt(x + 0.0),
        .Recip => @divExact(1.0, x + 0.0),
    };
}

fn ScalarZipOpReturnType(comptime zip_op: ZipOp, comptime a: anytype, comptime b: anytype) type {
    return switch (zip_op) {
        .Lt, .Eq => bool,
        .Xor => @TypeOf(a ^ b),
        else => @TypeOf(a + b),
    };
}

fn scalarZipOpEval(comptime zip_op: ZipOp, a: anytype, b: anytype) ScalarZipOpReturnType(zip_op, a, b) {
    return comptime switch (zip_op) {
        .Add => a + b,
        .Mul => a * b,
        .Maximum => @max(a, b),
        .Lt => a < b,
        .Eq => a == b,
        .Xor => a ^ b,
        else => @compileError("Not implemented"),
    };
}

fn ScalarOpReturnType(comptime op: Op) type {
    return @TypeOf(switch (op) {
        .MapOp => |map_op| struct {
            inline fn f(x: anytype) ScalarMapOpReturnType(map_op, x) {
                return comptime scalarMapOpEval(map_op, x);
            }
        },
        .ZipOp => |zip_op| struct {
            inline fn f(a: anytype, b: anytype) ScalarZipOpReturnType(zip_op, a, b) {
                return comptime scalarZipOpEval(zip_op, a, b);
            }
        },
        else => @compileError("Not implemented"),
    }.f);
}

pub fn EvalFunc(comptime op: Op) ScalarOpReturnType(op) {
    return comptime switch (op) {
        .MapOp => |map_op| struct {
            inline fn f(x: anytype) ScalarMapOpReturnType(map_op, x) {
                return comptime scalarMapOpEval(map_op, x);
            }
        },
        .ZipOp => |zip_op| struct {
            inline fn f(a: anytype, b: anytype) ScalarZipOpReturnType(zip_op, a, b) {
                return comptime scalarZipOpEval(zip_op, a, b);
            }
        },
        else => @compileError("Not implemented"),
    }.f;
}

const exp2 = EvalFunc(.{ .MapOp = .Exp2 });
const add = EvalFunc(.{ .ZipOp = .Add });
const neg = EvalFunc(.{ .MapOp = .Neg });

const print = @import("std").debug.print;
// TODO: Test fails because Zig does not automatically cast during arithmetic
// We can handle this by doing the following:
// To keep the size requirements the same, use @sizeOf
// to find the number of bits for the int data, and use the same number of bits
// for the floating point representation
test "test impl" {
    const a: i32 = 2;
    const b: i32 = 3;
    print("\n-a = {any}", .{neg(a)});
    print("\nexp2(b) = {any}", .{exp2(b)});
    print("\na+b = {any}\n", .{add(a, b)});
}
