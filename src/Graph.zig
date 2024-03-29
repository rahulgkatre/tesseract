const std = @import("std");
const anytensor = @import("anytensor.zig");
const ops = @import("ops.zig");
const utils = @import("utils.zig");
const dtypes = @import("dtypes.zig");
const Record = @import("record.zig").Record;

var gpa: std.heap.GeneralPurposeAllocator(.{}) = undefined;
var arena: std.heap.ArenaAllocator = undefined;
pub var ordinals: std.AutoArrayHashMap(usize, usize) = undefined;
pub var tensors: std.AutoArrayHashMap(usize, anytensor) = undefined;
pub var gradients: std.AutoArrayHashMap(usize, anytensor) = undefined;

pub fn init() void {
    gpa = .{};
    arena = std.heap.ArenaAllocator.init(gpa.allocator());
    ordinals = std.AutoArrayHashMap(usize, usize).init(arena.allocator());
    tensors = std.AutoArrayHashMap(usize, anytensor).init(arena.allocator());
}

pub fn deinit() void {
    arena.deinit();
    _ = gpa.deinit();
}

pub fn jsonStringify(_: @This(), write_stream: anytype) !void {
    const tensors_json: []anytensor.JsonFormat = gpa.allocator().alloc(anytensor.JsonFormat, tensors.count()) catch unreachable;
    defer gpa.allocator().free(tensors_json);
    const records_json: []Record.JsonFormat = gpa.allocator().alloc(Record.JsonFormat, tensors.count()) catch unreachable;
    defer gpa.allocator().free(records_json);
    for (tensors.keys(), records_json, tensors_json) |key, *rec, *ct| {
        const tensor: *const anytensor = @ptrFromInt(key);
        ct.* = tensor.toJsonFormat();
        rec.* = (&tensor.record).toJsonFormat();
    }
    try write_stream.write(.{
        .tensors = tensors_json,
        .operations = records_json,
    });
}

pub fn viz(writer: anytype) !void {
    const written = arena.allocator().alloc(bool, tensors.count()) catch unreachable;
    defer arena.allocator().free(written);
    try writer.print(
        \\digraph G {{
        \\    compound=true;
        \\
    , .{});
    // TODO: Support for multiple entrypoints in the case of a DAG with multiple sinks
    for (tensors.keys()) |key| {
        const tensor: *const anytensor = @ptrFromInt(key);
        if (written[@intCast(tensor.uid())]) {
            continue;
        }
        written[@intCast(tensor.uid())] = true;
        try tensor.viz(writer);
    }

    try writer.print(
        \\}}
        \\
    , .{});
}
