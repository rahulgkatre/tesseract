const std = @import("std");
const utils = @import("utils.zig");

pub const Const = struct {
    value: i64,
    min: i64,
    max: i64,

    pub fn of(value: i64) Expr {
        return .{ .Const = .{ .value = value, .min = value, .max = value } };
    }

    pub fn add(const1: Const, const2: Const) Expr {
        return of(const1.value + const2.value);
    }
    pub fn mul(const1: Const, const2: Const) Expr {
        return of(const1.value * const2.value);
    }
    pub fn div(const1: Const, const2: Const) Expr {
        std.debug.assert(@mod(const1.value, const2.value) == 0);
        return of(@divExact(const1.value, const2.value));
    }
    pub fn mod(const1: Const, const2: Const) Expr {
        return of(@mod(const1.value, const2.value));
    }
};

const Op = enum {
    div,
    mod,
    add,
    mul,
};
const Expr = struct {
    op: Op,
    input: *const SymInt,
    other: *const SymInt,
    min: i64,
    max: i64,

    pub fn of(op: Op, input: *const SymInt, other: *const SymInt, min: i64, max: i64) SymInt {
        return .{
            .Expr = .{
                .op = op,
                .input = input,
                .other = other,
                .min = min,
                .max = max,
            },
        };
    }

    pub fn div(input: SymInt, other: SymInt) SymInt {
        const in_expr = input.Expr;
        switch (other) {
            .Const => switch (in_expr.op) {
                .mul => if (std.meta.activeTag(in_expr.other) == .Const and std.meta.activeTag(input.Const) == .Const) {},
            },
        }
    }
};

pub const SymInt = union(enum) {
    Var: Var,
    Const: Const,
    Expr: Expr,

    pub fn wrap(x: anytype) SymInt {
        var T: type = @TypeOf(x);
        var data = x;
        if (std.meta.activeTag(@typeInfo(T)) == .Pointer) {
            T = @TypeOf(x.*);
            data = x.*;
        }
        if (Var == T or Const == T or Expr == T) {
            return @unionInit(SymInt, utils.rawTypeName(T), data);
        }
        if (@TypeOf(.name) == T) {
            return Var.of(data);
        }
        return switch (@typeInfo(T)) {
            .Int, .ComptimeInt => Const.of(data),
            else => unreachable,
        };
    }

    pub fn constantFold(x: SymInt) SymInt {
        return switch (x) {
            inline else => |s| if (s.min == s.max) Const.of(s.min) else x,
        };
    }

    pub fn add(input: anytype, other: anytype) SymInt {
        return switch (wrap(input)) {
            inline else => |sym_input| sym_input.add(wrap(b)),
        };
    }

    pub fn mul(input: anytype, other: anytype) SymInt {
        return switch (wrap(input)) {
            inline else => |sym_input| @TypeOf(sym_input).mul(wrap(input), wrap(other)),
        };
    }

    pub fn div(input: anytype, other: anytype) SymInt {
        return switch (wrap(input)) {
            inline else => |sym_input| @TypeOf(sym_input).mul(wrap(input), wrap(other)),
        };
    }

    pub fn mod(input: anytype, other: anytype) SymInt {
        return switch (wrap(input)) {
            inline else => |sym_input| @TypeOf(sym_input).mul(wrap(input), wrap(other)),
        };
    }
};

pub const Guard = union(enum) {
    lessThan,
    greaterThan,
    equals,
    geq,
    leq,
};

pub const Constraint = struct {
    target: *const Var,
    guard: Guard,
    rhs: *const Expr,
};

pub const Var = struct {
    name: []const u8,
    min: i64 = 0,
    max: i64 = std.math.maxInt(i64),

    pub fn of(name: @TypeOf(.variable)) SymInt {
        return .{ .Var = .{ .name = std.fmt.comptimePrint("{any}", .{name})[1..] } };
    }

    pub fn add(input: SymInt, other: SymInt) SymInt {
        const in_min, const in_max = switch (input) {
            inline else => |sym_in| .{ sym_in.min, sym_in.max },
        };
        const other_min, const other_max = switch (other) {
            inline else => |sym_other| .{ sym_other.min, sym_other.max },
        };
        return Expr.of(.add, &input, &other, in_min + other_min, @min(in_max + other_max, std.math.maxInt(i64))).constantFold();
    }

    pub fn mul(input: SymInt, other: SymInt) SymInt {
        const in_var = input.Var;
        return switch (other) {
            .Const => |sym_other| Expr.of(
                .add,
                &input,
                &other,
                in_var.max * sym_other.min,
                @min(in_var.max * sym_other.max, std.math.maxInt(i64)),
            ).constantFold(),
            else => unreachable,
        };
    }

    pub fn div(input: SymInt, other: SymInt) SymInt {}

    pub fn mod(input: SymInt, other: SymInt) SymInt {}
};
