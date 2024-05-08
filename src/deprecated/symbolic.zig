const std = @import("std");

fn isSymbolic(comptime ndims: comptime_int, comptime dims: [ndims]Dim) bool {
    for (dims) |dim| {
        switch (dim) {
            .constant => {},
            else => return true,
        }
    }
    return false;
}

fn symbolicDims(comptime shape: anytype) [shape.len]Dim {
    var symbolic_shape: [shape.len]Dim = undefined;
    for (shape, 0..) |shape_d, d| {
        symbolic_shape[d] = switch (@typeInfo(@TypeOf(shape_d))) {
            .Struct => {
                //TODO: Support for named dimensions
                @compileError("Named dimensions are not supported yet");
            },
            // Strings will get converted to an enum field for eventual named dimension support
            // Enum field value is the dim number, so it can be passed into any function that requires a dim by index
            .Pointer => .{ .variable = shape_d },
            .ComptimeInt, .Int => .{
                .constant = shape_d,
            },
            else => .{ .variable = std.fmt.comptimePrint("{any}", .{shape_d}) },
        };
    }
    return symbolic_shape;
}

fn symbolicStrides(comptime ndims: u8, comptime symbolic_shape: [ndims]Dim) [ndims + 1]Dim {
    var strides: [ndims + 1]Dim = undefined;
    var offset: Dim = .{ .constant = 1 };
    for (0..ndims - 1) |d| {
        const stride = Dim.mul(symbolic_shape[ndims - d - 1], offset);
        strides[ndims - d - 2] = stride;
        offset = stride;
    }
    strides[ndims - 1] = .{ .constant = 1 };
    strides[ndims] = .{ .constant = 0 };
    for (0..ndims) |d| {
        if (symbolic_shape[d].equalsConstant(0) or symbolic_shape[d].equalsConstant(1)) {
            strides[d] = .{ .constant = 0 };
        }
    }
    return strides;
}

pub const Dim = union(enum) {
    // Plan for symbolic:
    // val represents a constant dimension size
    // variable represents a purely variable size
    // symbolic represents a symbolic combination of the two but can be resolved to val or variable
    // if all symbolic variables are vals or if identity or zero axioms apply

    constant: i64,
    variable: []const u8,
    symbolic: Symbolic,

    pub fn equalsConstant(self: Dim, val: i64) bool {
        return switch (self) {
            .constant => self.constant == val,
            else => false,
        };
    }

    pub fn equals(a: Dim, b: Dim) bool {
        return switch (a) {
            .constant => |lhs| switch (b) {
                .constant => |rhs| lhs == rhs,
                else => false,
            },
            .variable => |lhs| switch (b) {
                .variable => |rhs| std.mem.eql(u8, lhs, rhs),
                else => false,
            },
            .symbolic => |lhs| switch (b) {
                .symbolic => |rhs| (lhs.op == rhs.op) and ((lhs.a.equals(rhs.a) and lhs.b.equals(rhs.b)) or (lhs.a.equals(rhs.b) and lhs.b.equals(rhs.a))),
                else => false,
            },
        };
    }

    pub fn variables(self: Dim) []const []const u8 {
        return switch (self) {
            .constant => &[0].{},
            .variable => |variable| &[1][]const u8{variable},
            .symbolic => |symbolic| symbolic.variables(),
        };
    }

    const Symbolic = struct {
        a: *const Dim,
        b: *const Dim,
        op: enum {
            add,
            mul,
        },

        fn variables(self: Symbolic) [][]const u8 {
            const a_vars: [][]const u8 = @constCast(self.a.variables());
            std.sort.block(u8, a_vars, {}, std.ascii.lessThanIgnoreCase);
            const b_vars: [][]const u8 = @constCast(self.b.variables());
            std.sort.block(u8, b_vars, {}, std.ascii.lessThanIgnoreCase);
            var c_vars: [a_vars.len + b_vars.len]std.builtin.Type.StructField = undefined;
            var a_i: usize = 0;
            var b_i: usize = 0;
            var c_i: usize = 0;
            while (a_i < a_vars.len and b_i < b_vars.len) {
                const sf1 = a_vars[a_i];
                const sf2 = b_vars[b_i];
                if (sf1.name.len != sf2.name.len or !std.comptime_string_map.eqlAsciiIgnoreCase(sf1.name, sf2.name)) {
                    c_vars[c_i] = sf1;
                    c_vars[c_i + 1] = sf2;
                    c_i += 2;
                } else {
                    c_vars[c_i] = sf1;
                    c_i += 1;
                }
                a_i += 1;
                b_i += 1;
            }
            return c_vars[0..c_i][0..];
        }
    };

    pub fn add(comptime a: Dim, comptime b: Dim) Dim {
        switch (a) {
            .constant => |d1| switch (b) {
                .constant => |d2| {
                    std.debug.assert(d1 + d2 >= 0);
                    return .{ .constant = d1 + d2 };
                },
                else => if (d1 == 0) return b,
            },
            else => switch (b) {
                .constant => |d2| if (d2 == 0) return a,
                else => {},
            },
        }
        return .{ .symbolic = .{ .a = &a, .b = &b, .op = .add } };
    }

    pub fn mul(comptime a: Dim, comptime b: Dim) Dim {
        // If possible, distribute the multiplication which makes it easier to check symbolic equality later
        switch (a) {
            .constant => |d1| {
                switch (b) {
                    .constant => |d2| return .{ .constant = d1 * d2 },
                    .variable => if (d1 == 0) return .{ .constant = 0 } else if (d1 == 1) return b,
                    .symbolic => |d2| switch (d2.op) {
                        .add => return add(a.mul(d2.a), a.mul(d2.b)),
                        else => {},
                    },
                }
            },
            .variable => {
                switch (b) {
                    .constant => |d2| if (d2 == 0) return .{ .constant = 0 } else if (d2 == 1) return a,
                    else => {},
                }
            },
            .symbolic => |d1| {
                switch (b) {
                    .constant => |d2| if (d2 == 0) return .{ .constant = 0 } else if (d2 == 1) return a,
                    .variable => switch (d1.op) {
                        .add => return add(d1.a.mul(b), d1.b.mul(b)),
                        else => {},
                    },
                    .symbolic => |d2| switch (d1.op) {
                        .add => switch (d2.op) {
                            .add => return add(add(d1.a.mul(d2.a), d1.a.mul(d2.b)), add(d1.b.mul(d2.a), d1.b.mul(d2.b))),
                            .mul => return add(d1.a.mul(b), d1.b.mul(b)),
                        },
                        .mul => switch (d2.op) {
                            .add => return add(a.mul(d2.a), a.mul(d2.b)),
                            .mul => {},
                        },
                    },
                }
            },
        }
        // Base case: the simple way
        return .{ .symbolic = .{ .a = &a, .b = &b, .op = .mul } };
    }
};
