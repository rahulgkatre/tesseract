const ZigCodegen = @import("codegen/ZigCodegen.zig");
const ops = @import("ops.zig");

pub const CodegenTypes = enum {
    Zig,
};

pub const Codegen = union(CodegenTypes) {
    Zig: ZigCodegen,
    const Self = @This();
    pub fn write_header(gen: *const Codegen, writer: anytype) void {
        switch (gen.*) {
            inline else => |*cg| cg.write_header(writer),
        }
    }
    pub fn write_footer(gen: *const Codegen, writer: anytype) void {
        switch (gen.*) {
            inline else => |*cg| cg.write_footer(writer),
        }
    }
    pub fn write_alloc(gen: *const Codegen, writer: anytype, id: usize, comptime dtype: type, size: usize) void {
        switch (gen.*) {
            inline else => |*cg| cg.write_storage(writer, id, dtype, size),
        }
    }
};

// All programming languages we would target support the same arithmetic expressions and formatting
pub fn idxToPos(comptime tensor_type: type, comptime idx_name: []const u8) []const u8 {
    _ = idx_name;
    _ = tensor_type;
    return "0";
}

pub fn broadcastIdxToPos(comptime tensor_type1: type, comptime tensor_type2: type, comptime idx_name: []const u8) []const u8 {
    _ = idx_name;
    _ = tensor_type2;
    _ = tensor_type1;
    return "0";
}

pub fn posToIdx(
    comptime tensor_type: type,
    comptime idx_name: []const u8,
) []const u8 {
    _ = idx_name;
    _ = tensor_type;
    return "0";
}
