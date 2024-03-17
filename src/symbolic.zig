const std = @import("std");

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

    pub fn equals(self: Dim, other: Dim) bool {
        return switch (self) {
            .constant => switch (other) {
                .constant => self.constant == other.constant,
                else => false,
            },
            .variable => switch (other) {
                .variable => std.mem.eql(u8, self.variable, other.variable),
                else => false,
            },
            .symbolic => switch (other) {
                .symbolic => self.symbolic.dim1.equals(other.symbolic.dim1) and self.symbolic.dim2.equals(other.symbolic.dim2),
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
        dim1: *const Dim,
        dim2: *const Dim,
        op: enum {
            sum,
            product,
        },

        fn variables(self: Symbolic) [][]const u8 {
            const dim1_vars: [][]const u8 = @constCast(self.dim1.variables());
            std.sort.block(u8, dim1_vars, {}, std.ascii.lessThanIgnoreCase);
            const dim2_vars: [][]const u8 = @constCast(self.dim2.variables());
            std.sort.block(u8, dim2_vars, {}, std.ascii.lessThanIgnoreCase);
            var self_vars: [dim1_vars.len + dim2_vars.len]std.builtin.Type.StructField = undefined;
            var src1_i: usize = 0;
            var src2_i: usize = 0;
            var dst_i: usize = 0;
            while (src1_i < dim1_vars.len and src2_i < dim2_vars.len) {
                const sf1 = dim1_vars[src1_i];
                const sf2 = dim2_vars[src2_i];
                if (sf1.name.len != sf2.name.len or !std.comptime_string_map.eqlAsciiIgnoreCase(sf1.name, sf2.name)) {
                    self_vars[dst_i] = sf1;
                    self_vars[dst_i + 1] = sf2;
                    dst_i += 2;
                } else {
                    self_vars[dst_i] = sf1;
                    dst_i += 1;
                }
                src1_i += 1;
                src2_i += 1;
            }
            return self_vars[0..dst_i][0..];
        }
    };

    pub fn add(comptime dim1: Dim, comptime dim2: Dim) Dim {
        switch (dim1) {
            .constant => |d1| switch (dim2) {
                .constant => |d2| {
                    std.debug.assert(d1 + d2 >= 0);
                    return .{ .constant = d1 + d2 };
                },
                else => if (d1 == 0) return dim2,
            },
            else => switch (dim2) {
                .constant => |d2| if (d2 == 0) return dim1,
                else => {},
            },
        }
        return .{ .symbolic = .{ .dim1 = &dim1, .dim2 = &dim2, .op = .product } };
    }

    pub fn mul(comptime dim1: Dim, comptime dim2: Dim) Dim {
        switch (dim1) {
            .constant => |d1| {
                switch (dim2) {
                    .constant => |d2| return .{ .constant = d1 * d2 },
                    else => if (d1 == 0) return .{ .constant = 0 } else if (d1 == 1) return dim2,
                }
            },
            else => {
                switch (dim2) {
                    .constant => |d2| if (d2 == 0) return .{ .constant = 0 } else if (d2 == 1) return dim1,
                    else => {},
                }
            },
        }
        return .{ .symbolic = .{ .dim1 = &dim1, .dim2 = &dim2, .op = .product } };
    }
};
