const std = @import("std");
const ops = @import("ops.zig");
const dtypes = @import("dtypes.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;
const tensor = @import("tensor.zig");
const symbolic = @import("symbolic.zig");

pub fn arrayPermute(comptime T: type, comptime len: u8, array: [len]u64, comptime perm: [len]u8) [len]T {
    var used: [len]bool = [_]bool{false} ** len;
    for (perm) |p| {
        if (p < len and !used[p]) {
            used[p] = true;
        } else {
            @compileError(std.fmt.comptimePrint("Invalid permutation {any}", .{perm}));
        }
    }
    for (used) |u| {
        if (!u) @compileError("Not all dims in permutation were used");
    }
    var new_array: [len]T = undefined;
    for (0..len) |dim| {
        new_array[dim] = array[perm[dim]];
    }
    return new_array;
}

pub fn arrayInsert(comptime len: u8, array: [len]u64, index: usize, val: u64) [len + 1]u64 {
    var new_array: [len + 1]u64 = undefined;
    for (0..index) |i| {
        new_array[i] = array[i];
    }
    new_array[index] = val;
    for (index..len) |i| {
        new_array[i + 1] = array[i];
    }
    return new_array;
}

pub fn arrayDelete(comptime len: u8, array: [len]u64, index: usize) [len - 1]u64 {
    var new_array: [len - 1]u64 = undefined;
    for (0..index) |i| {
        new_array[i] = array[i];
    }
    for (index + 1..len) |i| {
        new_array[i - 1] = array[i];
    }
    return new_array;
}

pub fn ravelMultiIndex(comptime ndims: u8, strides: [ndims]u64, offset: u64, multi_idx: [ndims]u64) usize {
    var flat_idx = offset;
    for (multi_idx, strides) |idx, stride| {
        flat_idx += idx * stride;
    }
    return flat_idx;
}

pub fn isContiguous(strides: []const symbolic.RuntimeExpr) bool {
    if (symbolic.isSymbolic(strides)) {
        return false;
    }

    var prev: u64 = std.math.maxInt(u64);
    for (strides) |stride| {
        if (stride.Const.value > 0) {
            if (stride.Const.value > prev) {
                return false;
            }
            prev = @intCast(stride);
        }
    }
    return true;
}

// Infer the contiguous stride pattern from the shape
// This is the default stride pattern unless a stride is manually provided
// using asStrided
pub fn contiguousStrides(comptime shape: []const symbolic.Expr) [shape.len]symbolic.Expr {
    const ndims = shape.len;
    if (ndims == 0) {
        return .{};
    }

    var offset: symbolic.Expr = symbolic.Const.of(1);
    var strides: [ndims]symbolic.Expr = undefined;
    for (0..ndims - 1) |d| {
        const stride = symbolic.Op.mul(shape[ndims - d - 1], offset);
        strides[ndims - d - 2] = stride;
        offset = stride;
    }
    strides[ndims - 1] = symbolic.Const.of(1);
    for (0..ndims) |d| {
        if (std.meta.activeTag(shape[d]) == .Const and (shape[d].Const.value == 0 or shape[d].Const.value == 1)) {
            strides[d] = symbolic.Const.of(0);
        }
    }
    return strides;
}

pub fn broadcastShape(shape1: anytype, shape2: anytype) [@max(shape1.len, shape2.len)]u64 {
    if (std.mem.eql(u64, &shape1, &shape2)) {
        return shape1;
    }
    const bc_ndims = @max(shape1.len, shape2.len);
    var bc_shape: [bc_ndims]u64 = undefined;
    for (0..bc_ndims) |i| {
        const dim1 = if (i >= shape1.len) 1 else shape1[shape1.len - i - 1];
        const dim2 = if (i >= shape2.len) 1 else shape2[shape2.len - i - 1]; // orelse dim1;
        if (dim1 != 1 and dim2 != 1 and dim1 != dim2) {
            @compileError(std.fmt.comptimePrint(
                \\
                \\Shapes are not compatible for broadcasting
                \\Shape 1: {[shape1]any}
                \\Shape 2: {[shape2]any}
                \\
                \\Reason: lhs dim {[i]d} and rhs dim{[i]d} must be equal, or one of them must be 1
                \\
                \\Dim 1: {[dim1]d}
                \\Dim 2: {[dim2]d}
            ,
                .{ .shape1 = shape1, .shape2 = shape2, .dim1 = dim1, .dim2 = dim2, .i = i },
            ));
        }
        bc_shape[bc_ndims - i - 1] = if (dim1 == dim2 or dim2 == 1) dim1 else dim2;
    }
    return bc_shape;
}

pub fn numElements(shape: []const symbolic.Expr) symbolic.Expr {
    var prod = symbolic.Const.of(1);
    for (shape) |s| {
        prod = symbolic.Op.mul(prod, s);
    }
    return prod;
}

pub fn extractDType(comptime Type: type) dtypes.DType {
    switch (@typeInfo(Type)) {
        .Array => |info| return extractDType(info.child),
        .Int, .Float, .Bool, .ComptimeInt, .ComptimeFloat => return @field(dtypes.DType, rawTypeName(Type)),
        .Struct => |info| if (info.backing_integer) |_| return @field(dtypes.DType, rawTypeName(Type)),
        else => {},
    }
    @compileError("ArrayType input for Tensor must be a array type (e.g. [M][N][P]DType), received " ++ std.fmt.std.fmt.comptimePrint("{any}", .{Type}));
}

pub fn extractNdims(comptime ArrayType: type) u8 {
    switch (@typeInfo(ArrayType)) {
        .Array => |info| return 1 + extractNdims(info.child),
        .Pointer => |info| return 1 + extractNdims(info.child),
        .Int, .Float, .Bool, .ComptimeInt, .ComptimeFloat => return 0,
        .Struct => |info| if (info.backing_integer) |_| return 0,
        else => {},
    }
    @compileError("ArrayType input for Tensor must be a array type (e.g. [M][N][P]DType), received " ++ std.fmt.comptimePrint("{any}", .{ArrayType}));
}

pub fn extractShape(comptime ArrayType: type) [extractNdims(ArrayType)]symbolic.Expr {
    switch (@typeInfo(ArrayType)) {
        .Array => |info| return .{symbolic.Const.of(info.len)} ++ extractShape(info.child),
        .Pointer => |info| return .{symbolic.Var.of(.hello)} ++ extractShape(info.child),
        .Int, .Float, .Bool, .ComptimeInt, .ComptimeFloat => return .{},
        .Struct => |info| if (info.backing_integer) |_| return .{},
        else => {},
    }
    @compileError("ArrayType input for Tensor must be a array type (e.g. [M][N][P]DType), received " ++ std.fmt.std.fmt.comptimePrint("{any}", .{ArrayType}));
}

pub fn rawTypeName(comptime T: type) []const u8 {
    const name = @typeName(T);
    for (0..name.len) |i| {
        if (name[name.len - i - 1] == '.') {
            return name[name.len - i ..];
        }
    }
    return name;
}

pub fn rawTagName(tagged: anytype) []const u8 {
    const name = @tagName(tagged);
    for (0..name.len) |i| {
        if (name[name.len - i - 1] == '.') {
            return name[name.len - i ..];
        }
    }
    return name;
}

pub fn signedToUnsignedDimNdims(ndims: u8, dim: i16) u8 {
    const value = if (dim < 0) ndims + dim else dim;
    if (value < 0 or value > ndims) {
        @compileError(std.fmt.std.fmt.comptimePrint(
            "Dimension index {d} is out of bounds {d}",
            .{ value, ndims },
        ));
    }
    return @intCast(value);
}

pub fn simplifiedView(input: anytype) struct {
    ndims: u8,
    shape: []const u64,
    strides: []const u64,
    offset: u64,
} {
    // TODO: Simplify contiguous or broadcasted sub intervals of the view
    const t = tensor.asTensor(input);
    if (t.ndims == 0) {
        return .{
            .ndims = 0,
            .shape = &.{},
            .strides = &.{},
            .offset = 0,
        };
    }

    var start_dim: u8 = 0;
    var end_dim = t.ndims;
    var simplified_shape = t.shape.*;
    var simplified_strides = t.strides.*;

    for (1..t.ndims) |dim| {
        if (simplified_strides[t.ndims - dim - 1] == 0) {
            simplified_shape[t.ndims - dim - 1] *= simplified_shape[t.ndims - dim];
            simplified_strides[t.ndims - dim - 1] = 0;
            start_dim += 1;
        } else if (isContiguous(simplified_strides[t.ndims - dim - 1 ..])) {
            simplified_shape[t.ndims - dim - 1] *= simplified_shape[t.ndims - dim];
            simplified_strides[t.ndims - dim - 1] = 1;
            end_dim -= 1;
        } else {
            break;
        }
    }

    return .{
        .ndims = end_dim - start_dim,
        .shape = simplified_shape[start_dim..end_dim],
        .strides = simplified_strides[start_dim..end_dim],
        .offset = t.offset,
    };
}

// test "simplify view" {
//     const s1 = comptime blk: {
//         const t1 = tensor.Tensor([2][2][2]f32).empty();
//         const s1 = simplifiedView(t1);
//         break :blk s1;
//     };
//     try std.testing.expectEqual(s1.shape[0..s1.ndims].*, .{8});
//     try std.testing.expectEqual(s1.strides[0..s1.ndims].*, .{1});

//     const s2 = comptime blk: {
//         const t2 = tensor.Tensor([2]f32).empty().expand(.{ 2, 2, 2 });
//         const s2 = simplifiedView(t2);
//         break :blk s2;
//     };
//     @compileLog(s2.ndims, s2.shape, s2.strides);

//     try std.testing.expectEqual(s2.shape[0..s2.ndims].*, .{ 4, 2 });
//     try std.testing.expectEqual(s2.strides[0..s2.ndims].*, .{ 0, 1 });
// }

pub fn paramsOf(comptime entrypoint: anytype) []const *const AnyTensor {
    const params, const len = comptime blk: {
        var list = ComptimeLinkedList(*const AnyTensor){};
        var curr = tensor.asTensor(entrypoint).toAnyTensor();
        var stack = (ComptimeLinkedList(*const AnyTensor){}).appendLeft(curr);
        while (stack.popLeft()) |tup| {
            @setEvalBranchQuota(500 * stack.len + 1000);
            curr, stack = tup;
            switch (tensor.asTensor(curr).meta.instr) {
                .InitOp => |instr| switch (instr.op) {
                    .param => {
                        list = list.appendLeft(curr);
                    },
                    else => {},
                },
                inline else => |instr| for (instr.in) |in| {
                    stack = stack.appendLeft(in);
                },
            }
        }
        var params: [list.len]*const AnyTensor = .{undefined} ** list.len;
        var i: usize = 0;
        while (list.popLeft()) |tup| {
            curr, list = tup;
            params[i] = curr;
            i += 1;
            if (list.len == 0) {
                break;
            }
        }
        const SortContext = struct {
            values: []*const AnyTensor,
            pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                return ctx.values[a].meta.label.?.len < ctx.values[b].meta.label.?.len or blk: {
                    if (ctx.values[a].meta.label.?.len > ctx.values[b].meta.label.?.len) {
                        break :blk false;
                    }
                    for (ctx.values[a].meta.label.?, ctx.values[b].meta.label.?) |char_a, char_b| {
                        if (char_a == char_b) {
                            continue;
                        } else {
                            break :blk char_a < char_b;
                        }
                    }
                    break :blk false;
                };
            }

            pub fn swap(ctx: @This(), a: usize, b: usize) void {
                return std.mem.swap(*const AnyTensor, &ctx.values[a], &ctx.values[b]);
            }
        };
        std.mem.sortUnstableContext(0, params.len, SortContext{ .values = &params });

        var active: ?*const AnyTensor = null;
        var deduped_params: [params.len]*const AnyTensor = undefined;
        i = 0;
        for (params) |param| {
            if (active) |act| {
                if (!std.mem.eql(u8, act.meta.label.?, param.meta.label.?)) {
                    active = param;
                    deduped_params[i] = param;
                    i += 1;
                } else {
                    continue;
                }
            } else {
                active = param;
                deduped_params[i] = param;
                i += 1;
            }
        }
        break :blk .{ deduped_params, i };
    };

    return params[0..len];
}

pub fn ComptimeLinkedList(comptime T: type) type {
    return struct {
        const Self = @This();
        const Node = struct {
            data: T,
            next: ?*const Node,
        };

        head: Node = .{
            .data = undefined,
            .next = null,
        },
        len: usize = 0,

        pub fn init(data: T) Self {
            return (Self{}).appendLeft(data);
        }

        pub fn appendLeft(self: Self, data: T) Self {
            return .{
                .head = .{
                    .data = data,
                    .next = &self.head,
                },
                .len = self.len + 1,
            };
        }

        pub fn popLeft(self: Self) ?std.meta.Tuple(&.{ T, Self }) {
            if (self.head.next) |next| {
                return .{
                    self.head.data,
                    .{
                        .head = next.*,
                        .len = self.len - 1,
                    },
                };
            } else {
                return null;
            }
        }

        pub fn format(
            self: Self,
            comptime _: []const u8,
            _: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            if (self.head) |head| {
                var node: ?*const Node = &head;
                while (node) |curr| {
                    try std.fmt.format(writer, "{x}->", .{@intFromPtr(curr)});
                    node = curr.next;
                }
                try std.fmt.format(writer, "null", .{});
            }
        }
    };
}

pub fn gcd(a: u64, b: u64) u64 {
    if (b == 0) {
        return a;
    } else {
        return gcd(b, @mod(a, b));
    }
}
