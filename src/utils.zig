const std = @import("std");
const comptimePrint = std.fmt.comptimePrint;
const ops = @import("ops.zig");
const dtypes = @import("dtypes.zig");
const anytensor = @import("anytensor.zig").anytensor;

pub fn arrayPermute(comptime T: type, comptime len: u8, array: [len]u64, perm: [len]u8) [len]T {
    var used: [len]bool = [_]bool{false} ** len;
    for (perm) |p| {
        if (p < len and !used[p]) {
            used[p] = true;
        } else {
            const msg = comptimePrint("Invalid permutation {any}", .{perm});
            if (@inComptime()) {
                @compileError(msg);
            } else {
                @panic(msg);
            }
        }
    }
    for (used) |u| {
        if (!u) {
            const msg = comptimePrint("Invalid permutation: {any}", .{perm});
            if (@inComptime()) {
                @compileError(msg);
            } else {
                @panic(msg);
            }
        }
    }
    var new_array: [len]T = undefined;
    for (0..len) |dim| {
        new_array[dim] = array[perm[dim]];
    }
    return new_array;
}

pub fn arrayInsert(comptime len: u8, array: [len]u64, index: usize, val: u64) [len + 1]u64 {
    var new_array: [len + 1]u64 = undefined;
    for (0..index) |i| {
        new_array[i] = array[i];
    }
    new_array[index] = val;
    for (index..len) |i| {
        new_array[i + 1] = array[i];
    }
    return new_array;
}

pub fn arrayDelete(comptime len: u8, array: [len]u64, index: usize) [len - 1]u64 {
    var new_array: [len - 1]u64 = undefined;
    for (0..index) |i| {
        new_array[i] = array[i];
    }
    for (index + 1..len) |i| {
        new_array[i - 1] = array[i];
    }
    return new_array;
}

pub fn ravelMultiIndex(comptime ndims: u8, strides: [ndims]u64, offset: u64, multi_idx: [ndims]u64) usize {
    var flat_idx = offset;
    for (multi_idx, strides) |idx, stride| {
        flat_idx += idx * stride;
    }
    return flat_idx;
}

// Infer the contiguous stride pattern from the shape
// This is the default stride pattern unless a stride is manually provided
// using asStrided
pub fn contiguousStrides(comptime ndims: u8, shape: [ndims]u64) [ndims]u64 {
    var offset: u64 = 1;
    var strides: [ndims]u64 = undefined;
    for (0..ndims - 1) |d| {
        const stride = shape[ndims - d - 1] * offset;
        strides[ndims - d - 2] = stride;
        offset = stride;
    }
    strides[ndims - 1] = 1;
    for (0..ndims) |d| {
        if (shape[d] == 0 or shape[d] == 1) {
            strides[d] = 0;
        }
    }
    return strides;
}

/// Utility function for visualizing the full graph that is created at compile time, no scheduling is done yet
pub fn dataflowViz(entrypoints: []const *const anytensor, writer: anytype, allocator: std.mem.Allocator) !void {
    var written = std.AutoArrayHashMap(*const anytensor, void).init(allocator);
    defer written.deinit();
    try writer.print(
        \\digraph G {{
        \\    compound=true;
        \\
    , .{});
    var queue = std.ArrayList(*const anytensor).init(allocator);
    defer queue.deinit();

    for (entrypoints) |entry| {
        try queue.append(@ptrCast(entry));
    }

    while (queue.popOrNull()) |tensor| {
        if (written.contains(tensor)) {
            continue;
        }
        try written.putNoClobber(tensor, {});

        switch (tensor.record.*) {
            .DataOp => |rec| {
                try switch (rec.op) {
                    .AsType => writer.print(
                        \\    {[op]s}_{[out]x}[label="{[op]s}({[data]s})"];
                        \\
                    , .{
                        .op = @tagName(rec.op),
                        .out = @intFromPtr(tensor),
                        .data = @tagName(tensor.dtype),
                    }),
                    .AsStrided => writer.print(
                        \\    {[op]s}_{[out]x}[label="{[op]s}{[data]any}"];
                        \\
                    , .{
                        .op = @tagName(rec.op),
                        .out = @intFromPtr(tensor),
                        .data = .{
                            tensor.shape[0..tensor.ndims],
                            tensor.strides[0..tensor.ndims],
                            tensor.offset,
                        },
                    }),
                };
            },
            inline else => |rec| {
                try writer.print(
                    \\    {[op]s}_{[out]x}[label="{[op]s}"];
                    \\
                , .{
                    .op = @tagName(rec.op),
                    .out = @intFromPtr(tensor),
                });
            },
        }

        switch (tensor.record.*) {
            .TernaryOp => |rec| {
                try writer.print(
                    \\    T_{[a]x}->{[op]s}_{[out]x}[label="{[a_dtype]s}{[a_shape]any}"];
                    \\    T_{[b]x}->{[op]s}_{[out]x}[label="{[b_dtype]s}{[b_shape]any}"];
                    \\    T_{[c]x}->{[op]s}_{[out]x}[label="{[c_dtype]s}{[c_shape]any}"];
                    \\    T_{[out]x}[label="T_{[out]x}"shape=box];
                    \\    {[op]s}_{[out]x}->T_{[out]x}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\    
                , .{
                    .op = @tagName(rec.op),
                    .out = @intFromPtr(tensor),
                    .out_dtype = @tagName(tensor.dtype),
                    .out_shape = tensor.shape[0..tensor.ndims],
                    .a = @intFromPtr(rec.a.tensor),
                    .a_dtype = @tagName(rec.a.tensor.dtype),
                    .a_shape = rec.a.tensor.shape[0..rec.a.tensor.ndims],
                    .b = @intFromPtr(rec.b.tensor),
                    .b_dtype = @tagName(rec.b.tensor.dtype),
                    .b_shape = rec.b.tensor.shape[0..rec.b.tensor.ndims],
                    .c = @intFromPtr(rec.c.tensor),
                    .c_dtype = @tagName(rec.c.tensor.dtype),
                    .c_shape = rec.c.tensor.shape[0..rec.c.tensor.ndims],
                });
            },
            .BinaryOp => |rec| {
                try writer.print(
                    \\    T_{[a]x}->{[op]s}_{[out]x}[label="{[a_dtype]s}{[a_shape]any}"];
                    \\    T_{[b]x}->{[op]s}_{[out]x}[label="{[b_dtype]s}{[b_shape]any}"];
                    \\    T_{[out]x}[label="T_{[out]x}"shape=box];
                    \\    {[op]s}_{[out]x}->T_{[out]x}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\
                , .{
                    .op = @tagName(rec.op),
                    .out = @intFromPtr(tensor),
                    .out_dtype = @tagName(tensor.dtype),
                    .out_shape = tensor.shape[0..tensor.ndims],
                    .a = @intFromPtr(rec.a.tensor),
                    .a_dtype = @tagName(rec.a.tensor.dtype),
                    .a_shape = rec.a.tensor.shape[0..rec.a.tensor.ndims],
                    .b = @intFromPtr(rec.b.tensor),
                    .b_dtype = @tagName(rec.b.tensor.dtype),
                    .b_shape = rec.b.tensor.shape[0..rec.b.tensor.ndims],
                });
            },
            .InitOp => |rec| {
                try writer.print(
                    \\    T_{[out]x}[label="T_{[out]x}"shape=box];
                    \\    {[op]s}_{[out]x}->T_{[out]x}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\
                , .{
                    .op = @tagName(rec.op),
                    .out = @intFromPtr(tensor),
                    .out_dtype = @tagName(tensor.dtype),
                    .out_shape = tensor.shape[0..tensor.ndims],
                });
            },
            inline else => |rec| {
                try writer.print(
                    \\    T_{[a]x}->{[op]s}_{[out]x}[label="{[a_dtype]s}{[a_shape]any}"];
                    \\    T_{[out]x}[label="T_{[out]x}"shape=box];
                    \\    {[op]s}_{[out]x}->T_{[out]x}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\
                , .{
                    .op = @tagName(rec.op),
                    .out = @intFromPtr(tensor),
                    .out_dtype = @tagName(tensor.dtype),
                    .out_shape = tensor.shape[0..tensor.ndims],
                    .a = @intFromPtr(rec.a.tensor),
                    .a_dtype = @tagName(rec.a.tensor.dtype),
                    .a_shape = rec.a.tensor.shape[0..rec.a.tensor.ndims],
                });
            },
        }

        switch (tensor.record.*) {
            .TernaryOp => |rec| {
                try queue.append(rec.a.tensor);
                try queue.append(rec.b.tensor);
                try queue.append(rec.c.tensor);
            },
            .BinaryOp => |rec| {
                try queue.append(rec.a.tensor);
                try queue.append(rec.b.tensor);
            },
            .InitOp => {},
            inline else => |rec| {
                try queue.append(rec.a.tensor);
            },
        }
    }

    try writer.print(
        \\}}
        \\
    , .{});
}

// pub fn jsonStringify(_: @This(), write_stream: anytype) !void {
//     const tensors_json: []anytensor.JsonFormat = gpa.allocator().alloc(anytensor.JsonFormat, tensors.count()) catch unreachable;
//     defer gpa.allocator().free(tensors_json);
//     const records_json: []Record.JsonFormat = gpa.allocator().alloc(Record.JsonFormat, tensors.count()) catch unreachable;
//     defer gpa.allocator().free(records_json);
//     for (tensors.keys(), records_json, tensors_json) |key, *rec, *ct| {
//         const tensor: *const anytensor = @ptrFromInt(key);
//         ct.* = tensor.toJsonFormat();
//         rec.* = Record.toJsonFormat(tensor.record, tensor);
//     }
//     try write_stream.write(.{
//         .tensors = tensors_json,
//         .operations = records_json,
//     });
// }

pub fn dataflowJson(entrypoints: []const *const anytensor, writer: anytype, allocator: std.mem.Allocator) !void {
    var tensors_json = std.AutoArrayHashMap(*const anytensor, anytensor.JsonFormat).init(allocator);
    defer tensors_json.deinit();

    var records_json = std.ArrayList(@import("record.zig").Record.JsonFormat).init(allocator);
    defer records_json.deinit();

    var queue = std.ArrayList(*const anytensor).init(allocator);
    defer queue.deinit();

    for (entrypoints) |entry| {
        try queue.append(@ptrCast(entry));
    }

    while (queue.popOrNull()) |tensor| {
        if (tensors_json.contains(tensor)) {
            continue;
        }
        try tensors_json.put(tensor, tensor.toJsonFormat());
        try records_json.append(tensor.record.toJsonFormat(tensor));
        switch (tensor.record.*) {
            .TernaryOp => |rec| {
                try queue.append(rec.a.tensor);
                try queue.append(rec.b.tensor);
                try queue.append(rec.c.tensor);
            },
            .BinaryOp => |rec| {
                try queue.append(rec.a.tensor);
                try queue.append(rec.b.tensor);
            },
            .InitOp => {},
            inline else => |rec| {
                try queue.append(rec.a.tensor);
            },
        }
    }

    try std.json.stringify(.{
        .tensors = tensors_json.values(),
        .operations = records_json.items,
    }, .{}, writer);
    try writer.print("\n", .{});
}
