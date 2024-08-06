const ops = @import("../ops.zig");
const dtypes = @import("../dtypes.zig");
const utils = @import("../utils.zig");
const autograd = @import("../autograd.zig");

/// Metadata for tensors, shared between the shape-typed Tensor and AnyTensor
pub const Metadata = struct {
    instr: ops.Instruction,
    grad_fn: *const anyopaque = autograd.noGrad,
    constant: bool,
    label: ?[]const u8,
    requires_grad: bool = false,
    dim_names: ?[]const ?[]const u8 = null,
};

pub fn PadMode(comptime dtype: dtypes.DType) type {
    return union(ops.DataOp.Args.Pad.Mode) {
        constant: dtypes.ZigType(dtype),
        reflect: void,
        replicate: void,
        circular: void,
    };
}
