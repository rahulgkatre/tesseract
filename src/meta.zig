const ops = @import("ops.zig");
const std = @import("std");
const utils = @import("utils.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;
const tensor = @import("tensor.zig");
const graph = @import("graph.zig");
const autograd = @import("autograd.zig");

/// Metadata for tensors, shared between the shape-typed Tensor and AnyTensor
pub const Metadata = struct {
    instr: ops.Instruction,
    backward_fn: *const anyopaque = autograd.noBackward,
    constant: bool,
    label: ?[]const u8,
    requires_grad: bool = false,
};
