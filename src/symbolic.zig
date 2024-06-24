const std = @import("std");
const utils = @import("utils.zig");

const Const = struct {
    value: i64,
    pub fn of(value: i64) Expr {
        return .{ .Const = .{ .value = value } };
    }

    pub fn add(const1: Const, const2: Const) Expr {
        return of(const1.value + const2.value);
    }
    pub fn mul(const1: Const, const2: Const) Expr {
        return of(const1.value * const2.value);
    }
    pub fn divExact(const1: Const, const2: Const) Expr {
        std.debug.assert(@mod(const1.value, const2.value) == 0);
        return of(@divExact(const1.value, const2.value));
    }
    pub fn mod(const1: Const, const2: Const) Expr {
        return of(@mod(const1.value, const2.value));
    }
};

const Op = enum {
    divExact,
    mod,
    add,
    mul,

    pub fn expr(op: Op, arg1: Expr, arg2: Expr) Expr {
        return .{ .Expr = .{ .op = op, .arg1 = arg1, .arg2 = arg2 } };
    }

    pub fn mul(arg1: Expr, arg2: Expr) Expr {
        return expr(.mul, arg1, arg2);
    }

    pub fn add(arg1: Expr, arg2: Expr) Expr {
        return expr(.add, arg1, arg2);
    }

    pub fn divExact(arg1: Expr, arg2: Expr) Expr {
        return expr(.divExact, arg1, arg2);
    }

    pub fn mod(arg1: Expr, arg2: Expr) Expr {
        return expr(.mod, arg1, arg2);
    }
};

const Guard = union(enum) {
    lessThan,
    greaterThan,
    equals,
    geq,
    leq,
};

const Var = struct {
    name: @TypeOf(.variable),
    scalar: Expr = Const.of(1),
    divisor: Expr = Const.of(1),
    addend: Expr = Const.of(0),

    min: u64 = 0,
    max: ?u64 = null,

    pub fn of(name: @TypeOf(.variable)) Expr {
        return .{ .Var = .{ .name = name } };
    }

    pub fn zeroAddend(self: Var) Expr {
        var copy = self;
        copy.addend = Const.of(0);
        return Expr{ .Var = copy };
    }

    pub fn setAddend(self: Var, addend: Expr) Expr {
        var copy = self;
        copy.addend = addend;
        return Expr{ .Var = copy };
    }

    pub fn addVar(var1: Var, var2: Var) Expr {
        return Op.expr(
            .add,
            Op.add(var1.zeroAddend(), var2.zeroAddend()),
            Op.add(var1.addend, var2.addend).eval(),
        );
    }

    pub fn mulVar(var1: Var, var2: Var) Expr {
        return Op.expr(
            .add,
            Op.expr(
                .add,
                Op.mul(var1.zeroAddend(), var2.zeroAddend()),
                Op.mul(var1.zeroAddend(), var2.addend).eval(),
            ),
            Op.expr(
                .add,
                Op.mul(var2.zeroAddend(), var1.addend).eval(),
                Op.mul(var1.addend, var2.addend).eval(),
            ),
        );
    }

    pub fn addConst(var1: Var, const2: Const) Expr {
        return (Expr{
            .Var = .{
                .name = var1.name,
                .addend = Op.add(var1.addend, Const.of(const2.value)).eval(),
                .divisor = var1.divisor,
                .scalar = var1.scalar,
            },
        }).eval();
    }

    pub fn mulConst(var1: Var, const2: Const) Expr {
        return (Expr{
            .Var = .{
                .name = var1.name,
                .addend = Op.mul(var1.addend, Const.of(const2.value)).eval(),
                .divisor = var1.divisor,
                .scalar = Op.mul(var1.scalar, Const.of(const2.value)).eval(),
            },
        }).eval();
    }

    pub fn divConst(var1: Var, const2: Const) Expr {
        return (Expr{
            .Var = .{
                .name = var1.name,
                .addend = Op.divExact(var1.addend, Const.of(const2.value)).eval(),
                .divisor = Op.mul(var1.divisor, Const.of(const2.value)).eval(),
                .scalar = var1.scalar,
            },
        }).eval();
    }

    pub fn addExpr(var1: Var, expr2: Expr) Expr {
        return Op.add(var1.zeroAddend(), Op.add(var1.addend, expr2).eval());
    }

    pub fn mulExpr(var1: Var, expr2: Expr) Expr {
        return Op.add(Op.mul(var1.zeroAddend(), Expr), Op.mul(var1.addend, expr2).eval());
    }
};

const Expr = union(enum) {
    Var: Var,
    Const: Const,
    Expr: struct { op: Op, arg1: Expr, arg2: Expr },

    pub fn flip(expr: Expr) Expr {
        return switch (expr) {
            .Expr => |ex| .{ .Expr = .{ .op = ex.op, .arg1 = ex.arg2, .arg2 = ex.arg1 } },
            else => expr,
        };
    }

    pub fn eval(expr: Expr) Expr {
        return switch (expr) {
            .Expr => |ex| switch (ex.arg1.eval()) {
                .Var => |var1| switch (ex.arg2.eval()) {
                    .Var => |var2| switch (ex.op) {
                        .add => var1.addVar(var2),
                        .mul => var1.mulVar(var2),
                        else => Op.expr(ex.op, ex.arg1.eval(), ex.arg2.eval()),
                    },
                    .Const => |const2| switch (ex.op) {
                        .add => var1.addConst(const2),
                        .mul => var1.mulConst(const2),
                        .divExact => var1.divConst(const2),
                        .mod => var1.setMinMax(var1.min, if (var1.max) |old_max| @min(old_max, const2.value) else const2.value),
                    },
                    .Expr => |expr2| switch (ex.op) {
                        .add => var1.addExpr(expr2),
                        .mul => var1.mulExpr(expr2),
                        else => Op.expr(ex.op, ex.arg1.eval(), ex.arg2.eval()),
                    },
                },
                .Const => |const1| switch (ex.arg2.eval()) {
                    .Const => |const2| switch (ex.op) {
                        .add => const1.add(const2),
                        .mul => const1.mul(const2),
                        .divExact => const1.divExact(const2),
                        .mod => const1.mod(const2),
                    },
                    else => expr.flip().eval(),
                },
                else => Op.expr(ex.op, ex.arg1.eval(), ex.arg2.eval()),
            },
            .Var => |v| blk: {
                var new = expr;
                if (v.max != null and v.min == v.max.?) {
                    if (v.max.? <= 0) {
                        @compileError("Var values cannot be 0 or negative");
                    }
                    new = Const.of(v.min);
                } else {
                    switch (v.scalar) {
                        .Const => |const1| switch (v.divisor) {
                            .Const => |const2| {
                                const gcd = utils.gcd(@intCast(@abs(const1.value)), @intCast(@abs(const2.value)));
                                new.Var.scalar = Const.of(@divExact(const1.value, gcd));
                                new.Var.divisor = Const.of(gcd);
                            },
                            else => {},
                        },
                        .Var => |var1| switch (v.divisor) {
                            .Const => |const2| {
                                new.Var.scalar = var1.divConst(const2);
                            },
                            else => {},
                        },
                        else => {},
                    }
                }
                break :blk new;
            },
            .Const => expr,
        };
    }
};

test Expr {
    comptime var expr = Op.add(Const.of(-1), Var.of(.x)).eval();
    try std.testing.expect(expr.Var.addend.Const.value == -1);
    expr = Op.mul(Const.of(2), expr).eval();
    try std.testing.expect(expr.Var.scalar.Const.value == 2);
    try std.testing.expect(expr.Var.addend.Const.value == -2);
    expr = Op.divExact(Const.of(2), expr).eval();
    try std.testing.expect(expr.Var.scalar.Const.value == 1);
    try std.testing.expect(expr.Var.addend.Const.value == -1);
}

const Constraint = struct {
    lhs: Var,
    guard: Guard,
    rhs: Expr,
};
