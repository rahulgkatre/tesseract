pub const std = @import("std");
pub const dtypes = @import("src/dtypes.zig");
pub const graph = @import("src/graph.zig");

const tensor = @import("src/tensor.zig");

// Expose only Tensor from tensor.zig
pub usingnamespace tensor;

pub usingnamespace @import("src/functions.zig");
pub const nn = @import("src/nn.zig");
pub const debug = @import("src/debug.zig");

test "tesseract" {
    @setEvalBranchQuota(100000);
    std.testing.refAllDecls(@This());
}
