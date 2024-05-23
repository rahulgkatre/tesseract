const std = @import("std");
const comptimePrint = std.fmt.comptimePrint;
const ops = @import("ops.zig");
const dtypes = @import("dtypes.zig");
const tracker = @import("tracker.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;

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

pub fn numEntries(comptime ndims: u8, shape: [ndims]u64) u128 {
    var prod: u128 = 1;
    for (shape) |s| {
        prod *= s;
    }
    return prod;
}

/// Utility function for visualizing the full graph that is created at compile time, no scheduling is done yet
pub fn dataflowViz(entrypoints: []const *const AnyTensor, writer: anytype, allocator: std.mem.Allocator) !void {
    const Viz = struct {
        fn opGroupViz(curr: ?*const tracker.OpGroupTracker.OpGroup, viz_writer: anytype) !u32 {
            if (curr) |group| {
                const depth = try opGroupViz(group.outer, viz_writer);
                try viz_writer.print("    subgraph cluster_{s}_{x} {{", .{ group.name, group.id });
                return depth + 1;
            } else {
                return 0;
            }
        }

        fn inputViz(in: *const AnyTensor, out: *const AnyTensor, viz_writer: anytype) !void {
            try viz_writer.print(
                \\    T_{[in]x}->{[op]s}_{[out]x}[label="{[dtype]s}{[shape]any}"];
                \\
            , .{
                .op = switch (out.meta.op_tracker) {
                    inline else => |info| @tagName(info.op),
                },
                .out = @intFromPtr(out),
                .in = @intFromPtr(in),
                .dtype = @tagName(in.dtype),
                .shape = in.shape[0..in.ndims],
            });
        }
    };

    var written = std.AutoArrayHashMap(*const AnyTensor, void).init(allocator);
    defer written.deinit();
    try writer.print(
        \\digraph G {{
        \\    compound=true;
        \\
    , .{});
    var queue = std.ArrayList(*const AnyTensor).init(allocator);
    defer queue.deinit();

    for (entrypoints) |entry| {
        try queue.append(@ptrCast(entry));
    }

    while (queue.popOrNull()) |out| {
        if (written.contains(out)) {
            continue;
        }
        try written.putNoClobber(out, {});

        const depth: u32 = try Viz.opGroupViz(out.meta.op_group_tracker.curr, writer);
        try writer.print("\n", .{});

        switch (out.meta.op_tracker) {
            .BufferOp => |info| {
                try switch (info.op) {
                    .Cast => writer.print(
                        \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\ngroup: {[group]s}\ndtype: {[data]s}"];
                        \\
                    , .{
                        .type = @typeName(@TypeOf(info.op)),
                        .op = @tagName(info.op),
                        .out = @intFromPtr(out),
                        .data = @tagName(out.dtype),
                        .group = @as([]const u8, if (out.meta.op_group_tracker.curr) |group| group.name else "null"),
                    }),
                    .View => writer.print(
                        \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\ngroup: {[group]s}\nshape {[shape]any}\nstrides {[strides]any}\noffset {[offset]d}"];
                        \\
                    , .{
                        .type = @typeName(@TypeOf(info.op)),
                        .op = @tagName(info.op),
                        .out = @intFromPtr(out),
                        .shape = out.shape[0..out.ndims],
                        .strides = out.strides[0..out.ndims],
                        .offset = out.offset,
                        .group = @as([]const u8, if (out.meta.op_group_tracker.curr) |group| group.name else "null"),
                    }),
                    else => writer.print(
                        \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\ngroup: {[group]s}\nshape: {[data]any}"];
                        \\
                    , .{
                        .type = @typeName(@TypeOf(info.op)),
                        .op = @tagName(info.op),
                        .out = @intFromPtr(out),
                        .data = out.shape[0..out.ndims],
                        .group = @as([]const u8, if (out.meta.op_group_tracker.curr) |group| group.name else "null"),
                    }),
                };
            },
            .InitOp => |info| try switch (info.op) {
                .Full => writer.print(
                    \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\ngroup: {[group]s}\nvalue: {[value]s}"];
                    \\
                , .{
                    .type = @typeName(@TypeOf(info.op)),
                    .op = @tagName(info.op),
                    .out = @intFromPtr(out),
                    .group = @as([]const u8, if (out.meta.op_group_tracker.curr) |group| group.name else "null"),
                    .value = info.args.Full.value,
                }),
                .Input => writer.print(
                    \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\ngroup: {[group]s}\nlabel: {[label]s}"];
                    \\
                , .{
                    .type = @typeName(@TypeOf(info.op)),
                    .op = @tagName(info.op),
                    .out = @intFromPtr(out),
                    .group = @as([]const u8, if (out.meta.op_group_tracker.curr) |group| group.name else "null"),
                    .label = out.meta.label orelse "null",
                }),
                .Range => writer.print(
                    \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\ngroup: {[group]s}\nstart: {[start]s}, stop: {[stop]s}"];
                    \\
                , .{
                    .type = @typeName(@TypeOf(info.op)),
                    .op = @tagName(info.op),
                    .out = @intFromPtr(out),
                    .group = @as([]const u8, if (out.meta.op_group_tracker.curr) |group| group.name else "null"),
                    .start = info.args.Range.start,
                    .stop = info.args.Range.stop,
                }),
                else => writer.print(
                    \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\ngroup: {[group]s}"];
                    \\
                , .{
                    .type = @typeName(@TypeOf(info.op)),
                    .op = @tagName(info.op),
                    .out = @intFromPtr(out),
                    .group = @as([]const u8, if (out.meta.op_group_tracker.curr) |group| group.name else "null"),
                }),
            },
            inline else => |info| try writer.print(
                \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\ngroup: {[group]s}\nargs: {[args]any}"];
                \\
            , .{
                .type = @typeName(@TypeOf(info.op)),
                .op = @tagName(info.op),
                .out = @intFromPtr(out),
                .args = info.args,
                .group = @as([]const u8, if (out.meta.op_group_tracker.curr) |group| group.name else "null"),
            }),
        }

        switch (out.meta.op_tracker) {
            .InitOp => {},
            inline else => |info| {
                for (info.in) |in| {
                    try Viz.inputViz(in, out, writer);
                }
            },
        }

        for (0..depth) |_| {
            try writer.print("    }}", .{});
        }

        try writer.print("\n", .{});

        switch (out.meta.op_tracker) {
            inline else => |info| {
                try writer.print(
                    \\    T_{[out]x}[label="dtype {[dtype]s}\nshape {[shape]any}\nstrides {[strides]any}\noffset {[offset]d}\nfolded_constant {[fc]}\nlabel: {[label]s}"shape=box];
                    \\    {[op]s}_{[out]x}->T_{[out]x}[label="{[dtype]s}{[shape]any}"];
                    \\
                , .{
                    .op = @tagName(info.op),
                    .out = @intFromPtr(out),
                    .dtype = @tagName(out.dtype),
                    .shape = out.shape[0..out.ndims],
                    .strides = out.strides[0..out.ndims],
                    .offset = out.offset,
                    .fc = out.meta.folded_constant,
                    .label = out.meta.label orelse "null",
                });
            },
        }

        switch (out.meta.op_tracker) {
            .InitOp => {},
            inline else => |info| {
                for (info.in) |in| {
                    try queue.append(in);
                }
            },
        }
    }

    try writer.print(
        \\}}
        \\
    , .{});
}

pub fn dataflowJson(entrypoints: []const *const AnyTensor, writer: anytype, allocator: std.mem.Allocator) !void {
    var tensors_json = std.AutoArrayHashMap(*const AnyTensor, AnyTensor.JsonFormat).init(allocator);
    defer tensors_json.deinit();

    var op_trackers_json = std.ArrayList(@import("tracker.zig").OpTracker.Json).init(allocator);
    defer op_trackers_json.deinit();

    var queue = std.ArrayList(*const AnyTensor).init(allocator);
    defer queue.deinit();

    for (entrypoints) |entry| {
        try queue.append(@ptrCast(entry));
    }

    while (queue.popOrNull()) |out| {
        if (tensors_json.contains(out)) {
            continue;
        }
        try tensors_json.put(out, out.toJsonFormat());
        try op_trackers_json.append(out.meta.op_tracker.toJson(out));
        switch (out.meta.op_tracker) {
            .InitOp => {},
            inline else => |info| {
                for (info.in) |in| {
                    try queue.append(in);
                }
            },
        }
    }

    try std.json.stringify(.{
        .tensors = tensors_json.values(),
        .operations = op_trackers_json.items,
    }, .{}, writer);
    try writer.print("\n", .{});
}
