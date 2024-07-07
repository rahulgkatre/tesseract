pub const std = @import("std");
pub const dtypes = @import("src/dtypes.zig");
pub const graph = @import("src/graph.zig");

pub usingnamespace @import("src/tensor/tensor.zig");
pub usingnamespace @import("src/tensor/functions.zig");

pub const nn = @import("src/nn.zig");
pub const debug = @import("src/debug.zig");
pub const autograd = @import("src/autograd.zig");

const tesseract = @This();
test tesseract {
    _ = tesseract;
    _ = nn;
    _ = @import("src/tensor/tensor.zig");
}
