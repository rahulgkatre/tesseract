const types = @import("tensor/types.zig");
const std = @import("std");
const utils = @import("utils.zig");
const dtypes = @import("utils.zig");
const autograd = @import("autograd.zig");

pub usingnamespace @import("tensor/functions.zig");

pub fn Where(Mask: type, True: type, False: type) type {
    std.debug.assert(dtypes.isBool(Mask._dtype));
    std.debug.assert(True._dtype == False._dtype);
    const bc_value_shape = utils.broadcastShape(True._shape, False._shape);
    const bc_result_shape = utils.broadcastShape(Mask._shape, bc_value_shape);
    return types.TensorType(True._dtype, bc_result_shape);
}
/// Conditional elementwise operator
/// out[i] = if (mask[i]) true_value[i] else false_value[i]
/// Supports broadcasting between all 3 tensors, but true value and false value are broadcasted together first and must also have the same dtype
pub fn where(mask: anytype, true_value: anytype, false_value: anytype) Where(types.TensorTypeOf(mask), types.TensorTypeOf(true_value), false_value) {
    const Out = Where(true_value, false_value);
    const mask_expand = mask.expand(Out._shape);
    const true_expand = types.asTensor(true_value).expand(Out._shape);
    const false_expand = types.asTensor(false_value).expand(Out._shape);
    return .{
        .instr = .{
            .TernaryOp = .{
                .in = .{ mask_expand.toAnyTensor(), true_expand.toAnyTensor(), false_expand.toAnyTensor() },
                .op = .where,
            },
        },
        .autograd = &.{
            .grad_fn = autograd.noGrad,
            .constant = false,
        },
    };
}
