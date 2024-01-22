const ZigCodegen = @import("codegen/ZigCodegen.zig");

pub const CodegenTypes = enum {
    Zig,
};

pub const Codegen = union(CodegenTypes) {
    Zig: ZigCodegen,
};

// TODO: I think these expression codegen functions can be moved to codegen.zig
// All programming languages we would target support the same arithmetic expressions
pub fn idxToPos(comptime tensor_type: type) void {
    _ = tensor_type;
}

pub fn broadcastIdxToPos(comptime tensor_type1: type, comptime tensor_type2: type) void {
    _ = tensor_type2;
    _ = tensor_type1;
}

pub fn posToIdx(comptime tensor_type: type) void {
    _ = tensor_type;
}
