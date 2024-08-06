pub fn Where(comptime true_value: anytype, comptime false_value: anytype) type {
    const True = typing.TensorTypeOf(true_value);
    const False = typing.TensorTypeOf(false_value);
    std.debug.assert(True.dtype == False.dtype);
    const bc_value_shape = utils.broadcastShape(True.shape, False.shape);
    const bc_result_shape = utils.broadcastShape(shape, bc_value_shape);
    return typing.TensorType(True.dtype, bc_result_shape);
}
/// Conditional elementwise operator
/// out[i] = if (mask[i]) true_value[i] else false_value[i]
/// Supports broadcasting between all 3 tensors, but true value and false value are broadcasted together first and must also have the same dtype
pub fn where(mask: dtypes.BoolTensor(Self), true_value: anytype, false_value: anytype) where(true_value, false_value) {
    const Out = Where(true_value, false_value);
    const mask_expand = mask.expand(Out.shape);
    const true_expand = typing.asTensor(true_value).expand(Out.shape);
    const false_expand = typing.asTensor(false_value).expand(Out.shape);
    return .{
        .meta = &.{
            .instr = .{
                .TernaryOp = .{
                    .in = .{ mask_expand.toAnyTensor(), true_expand.toAnyTensor(), false_expand.toAnyTensor() },
                    .op = .where,
                },
            },
            .constant = false,
            .label = null,
        },
    };
}
