pub const std = @import("std");
pub const dtypes = @import("src/dtypes.zig");

const tensor = @import("src/tensor.zig");
const GraphInternal = @import("src/Graph.zig");
const ProgramInternal = @import("src/Program.zig");

// Expose the simple Tensor function rather than the full one
pub const Tensor = tensor.InferredStrides;
pub const constant = tensor.constant;
pub const range = tensor.range;

// Expose only some functions publicly when using Tesseract as a Zig library
pub const Graph = struct {
    pub const viz = GraphInternal.viz;
    pub const trace = GraphInternal.trace;
    pub const Fusion = GraphInternal.Fusion;
};

pub const Program = struct {
    pub const code = ProgramInternal.code;
    pub const fromGraph = ProgramInternal.fromGraph;
};

test "tesseract" {
    _ = ProgramInternal;
}

pub fn init() void {
    GraphInternal.init(std.heap.page_allocator);
    ProgramInternal.init(std.heap.page_allocator);
}

pub fn deinit() void {
    GraphInternal.deinit();
    ProgramInternal.deinit();
}
