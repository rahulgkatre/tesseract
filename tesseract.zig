const dtypes = @import("src/dtypes.zig");
const tensor = @import("src/tensor/tensor.zig");
const functions = @import("src/functions.zig");

pub usingnamespace tensor;
pub usingnamespace functions;

pub const nn = @import("src/nn.zig");
pub const debug = @import("src/debug.zig");
pub const autograd = @import("src/autograd.zig");
pub const graph = @import("src/graph.zig");

const tesseract = @This();
test tesseract {
    _ = tensor;
    _ = functions;
    _ = nn;
    _ = debug;
    _ = autograd;
    _ = graph;
}
