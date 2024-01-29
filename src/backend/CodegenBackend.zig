// TODO: Write a backend that emits a zig file for all the code
// https://www.youtube.com/watch?v=iWIuaUmMhbI
// The purpose of this is to inline function calls in source code
// which will hopefully make it easier for LLVM to optimize them with loop transforms

const std = @import("std");
const ops = @import("../ops.zig");
const Backend = @import("../backend.zig").Backend;
const CodegenBackend = @This();
const codegen = @import("../codegen.zig");

const StorageArena = struct {
    var global_arena: std.heap.ArenaAllocator = undefined;
    fn init(arena: std.heap.ArenaAllocator) void {
        global_arena = arena;
    }
    fn deinit() void {
        global_arena.deinit();
        global_arena = undefined;
    }
    fn allocator() std.mem.Allocator {
        return global_arena.allocator();
    }
};

const CodeGenerator = struct {
    var generator: ?codegen.Codegen = null;
    var file: ?std.fs.File = null;
    fn init(_generator: codegen.Codegen, filename: []const u8) void {
        generator = _generator;
        file = std.fs.cwd().createFile(filename, .{}) catch @panic("Unable to create codegen output file");
        generator.?.write_header(writer());
    }
    fn gen() codegen.Codegen {
        return generator.?;
    }
    fn writer() @TypeOf(file.?.writer()) {
        return file.?.writer();
    }
    fn deinit() void {
        generator.?.write_footer(writer());
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
    const store = StorageArena.allocator().create(Backend.Storage(dtype)) catch unreachable;
    const store_type = Storage(dtype);
    _ = store_type;
    store.* = .{
        .Codegen = .{},
    };
    return store;
}

pub fn runtime(_: *const CodegenBackend, args: anytype) void {
    StorageArena.init(std.heap.ArenaAllocator.init(std.heap.page_allocator));
    CodeGenerator.init(codegen.Codegen{ .Zig = .{} }, args.filename);
}

pub fn finished(_: *const CodegenBackend) void {
    CodeGenerator.deinit();
    StorageArena.deinit();
}

pub inline fn map(
    _: *const CodegenBackend,
    comptime op: ops.MapOp,
    x_ptr: anytype,
    out_ptr: *@TypeOf(x_ptr.*),
) void {
    switch (CodeGenerator.gen()) {
        inline else => |cg| cg.map(CodeGenerator.writer(), op, x_ptr, out_ptr) catch unreachable,
    }
}

pub inline fn cast(
    _: *const CodegenBackend,
    comptime new_dtype: type,
    x_ptr: anytype,
    out_ptr: *@TypeOf(x_ptr.*).Cast(new_dtype),
) void {
    switch (CodeGenerator.gen()) {
        inline else => |cg| cg.cast(CodeGenerator.writer(), new_dtype, x_ptr, out_ptr) catch unreachable,
    }
}

pub inline fn zip(
    _: *const CodegenBackend,
    comptime op: ops.ZipOp,
    a_ptr: anytype,
    b_ptr: anytype,
    out_ptr: *@TypeOf(a_ptr.*).Broadcast(@TypeOf(b_ptr.*)),
) void {
    switch (CodeGenerator.gen()) {
        inline else => |cg| cg.zip(CodeGenerator.writer(), op, a_ptr, b_ptr, out_ptr) catch unreachable,
    }
}

pub inline fn reduce(
    _: *const CodegenBackend,
    comptime op: ops.ReduceOp,
    x_ptr: anytype,
    comptime dim: ?u8,
    out_ptr: *@TypeOf(x_ptr.*).Reduce(dim),
) void {
    switch (CodeGenerator.gen()) {
        inline else => |cg| cg.reduce(CodeGenerator.writer(), op, x_ptr, dim, out_ptr) catch unreachable,
    }
}
