const MapOp = enum { Neg, Log2, Exp2 };
const ZipOp = enum { Add, Mul };
// const ReduceOp = enum { Sum, Max };
const OpTypes = enum { MapOp, ZipOp };
const Op = union(OpTypes) {
    MapOp: MapOp,
    ZipOp: ZipOp,
    // ReduceOp: ReduceOp,
};

fn MapOpReturn(comptime map_op: MapOp, comptime x: anytype) type {
    return switch (map_op) {
        .Neg => @TypeOf(x), // Neg can apply to any numeric type (or boolean)
        else => @TypeOf(x + 0.0), // Other
    };
}

fn OpImpl(comptime op: Op) type {
    return @TypeOf(switch (op) {
        .MapOp => |map_op| struct {
            inline fn f(x: anytype) MapOpReturn(map_op, x) {
                return x;
            }
        },
        .ZipOp => struct {
            // Use a +
            inline fn f(a: anytype, b: anytype) @TypeOf(a + b) {
                return a + b;
            }
        },
    }.f);
}

fn impl(comptime op: Op) OpImpl(op) {
    // TODO: Will comptime switch collapse to just a function body containing the chosen branch?
    return switch (op) {
        .MapOp => |map_op| struct {
            inline fn f(x: anytype) MapOpReturn(map_op, x) {
                return comptime switch (map_op) {
                    .Neg => -x,
                    .Exp2 => @exp2(x + 0.0),
                    .Log2 => @log2(x + 0.0),
                };
            }
        },
        .ZipOp => |zip_op| struct {
            inline fn f(a: anytype, b: anytype) @TypeOf(a + b) {
                return comptime switch (zip_op) {
                    .Add => a + b,
                    .Mul => a * b,
                };
            }
        },
    }.f;
    // Alternative: a closure for every op
    //     switch (map_op) {
    //         .Log2 => struct {
    //             fn f(x: anytype) @TypeOf(x) {
    //                 return @log2(x);
    //             }
    //         },
    //         .Exp2 => struct {
    //             fn f(x: anytype) @TypeOf(x) {
    //                 return @exp2(x);
    //             }
    //         },
    //     }.f,
    //     .ZipOp => |zip_op| switch (zip_op) {
    //         .Add => struct {
    //             fn f(a: anytype, b: anytype) @TypeOf(a + b) {
    //                 return a + b;
    //             }
    //         },
    //         .Mul => struct {
    //             fn f(a: anytype, b: anytype) @TypeOf(a + b) {
    //                 return a * b;
    //             }
    //         },
    //     }.f,
    // };
}

const exp2 = impl(.{ .MapOp = .Exp2 });
const log2 = impl(.{ .MapOp = .Log2 });
const add = impl(.{ .ZipOp = .Add });
const mul = impl(.{ .ZipOp = .Mul });
const neg = impl(.{ .MapOp = .Neg });

test "test impl" {
    const a = 2;
    const b = 3;
    _ = neg(a);
    _ = exp2(b);
    _ = log2(a);
    _ = add(a, b);
    _ = mul(a, b);
}
