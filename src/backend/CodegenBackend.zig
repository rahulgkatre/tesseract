// TODO: Write a backend that emits a zig file for all the code
// https://www.youtube.com/watch?v=iWIuaUmMhbI
// The purpose of this is to inline function calls in source code
// which will hopefully make it easier for LLVM to optimize them with loop transforms

const std = @import("std");
const ops = @import("../ops.zig");
const Backend = @import("../backend.zig").Backend;
const CodegenBackend = @This();
const codegen = @import("../codegen.zig");

const CodeGenerator = struct {
    var generator: ?codegen.CodegenTypes = null;
    var file: ?std.fs.File = null;
    fn init(_generator: codegen.CodegenTypes, _file: std.fs.File) void {
        generator = _generator;
        file = _file;
    }
    fn gen() codegen.CodegenTypes {
        return generator.?;
    }
    fn writer() @TypeOf(file.?.writer()) {
        return file.?.writer();
    }
    fn deinit() void {
        file.?.close();
    }
};

// TODO: This should generate code for allocating an array
pub fn Storage(comptime dtype: type) type {
    return struct {
        const Self = @This();
        // pub const vec_len = std.simd.suggestVectorLength(dtype) orelse @sizeOf(dtype);
        // pub const vec_alignment = @alignOf(@Vector(vec_len, dtype));
        // data: []align(vec_alignment) dtype,
        // size: usize,
        pub fn fill(self: *Self, value: dtype) void {
            _ = value;
            _ = self;
            //@memset(self.data, value);
        }
        pub fn load(self: *Self, data: []const dtype) void {
            _ = data;
            _ = self;
            // @memcpy(self.data, data);
        }
    };
}

pub fn storage(_: *const CodegenBackend, comptime dtype: type, comptime size: usize) *Backend.Storage(dtype) {
    _ = size;
    // const store = StorageArena.allocator().create(Backend.Storage(dtype)) catch unreachable;
    // const store_type = Storage(dtype);
    var fake: Backend.Storage(dtype) = .{ .Codegen = .{} };
    return @constCast(&fake);
}

pub fn runtime(_: *const CodegenBackend, _: anytype) void {
    // StorageArena.init(std.heap.ArenaAllocator.init(std.heap.page_allocator));
}

pub fn finished(_: *const CodegenBackend) void {
    // StorageArena.deinit();
}

pub inline fn map(
    _: *const CodegenBackend,
    comptime op: ops.MapOp,
    x_ptr: anytype,
    comptime x_id: usize,
    out_ptr: *@TypeOf(x_ptr.*),
    comptime out_id: usize,
) []const u8 {
    switch (CodeGenerator.gen()) {
        inline else => |cg| cg.map(op, x_ptr, x_id, out_ptr, out_id),
    }
}

pub inline fn cast(
    _: *const CodegenBackend,
    comptime new_dtype: type,
    x_ptr: anytype,
    comptime x_id: usize,
    out_ptr: *@TypeOf(x_ptr.*),
    comptime out_id: usize,
) []const u8 {
    switch (CodeGenerator.gen()) {
        inline else => |cg| cg.cast(new_dtype, x_ptr, x_id, out_ptr, out_id),
    }
}

pub inline fn zip(
    _: *const CodegenBackend,
    comptime op: ops.ZipOp,
    a_ptr: anytype,
    comptime a_id: usize,
    b_ptr: anytype,
    comptime b_id: usize,
    out_ptr: *@TypeOf(a_ptr.*).Broadcast(@TypeOf(b_ptr.*)),
    comptime out_id: usize,
) []const u8 {
    switch (CodeGenerator.gen()) {
        inline else => |cg| cg.zip(op, a_ptr, a_id, b_ptr, b_id, out_ptr, out_id),
    }
}

pub inline fn reduce(
    _: *const CodegenBackend,
    comptime op: ops.MapOp,
    x_ptr: anytype,
    comptime x_id: usize,
    comptime dim: ?u8,
    out_ptr: *@TypeOf(x_ptr.*),
    comptime out_id: usize,
) []const u8 {
    switch (CodeGenerator.gen()) {
        inline else => |cg| cg.reduce(op, x_ptr, x_id, dim, out_ptr, out_id),
    }
}
