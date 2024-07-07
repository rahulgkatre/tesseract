const ops = @import("../ops.zig");
const utils = @import("../utils.zig");
const autograd = @import("../autograd.zig");

/// Metadata for tensors, shared between the shape-typed Tensor and AnyTensor
pub const Metadata = struct {
    instr: ops.Instruction,
    reverse_ad_fn: *const anyopaque = autograd.noBackward,
    forward_ad_fn: ?*const anyopaque = null,
    constant: bool,
    label: ?[]const u8,
    requires_grad: bool = false,
    dim_names: ?[]const ?[]const u8 = null,
};
