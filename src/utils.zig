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
        fn blocksViz(curr_block: ?*const tracker.BlockTracker.Block, viz_writer: anytype) !u32 {
            if (curr_block) |block| {
                const depth = try blocksViz(block.outer, viz_writer);
                try viz_writer.print("    subgraph cluster_{s}_{x} {{", .{ block.name, block.id });
                // try viz_writer.print("    {{", .{});

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
                .op = switch (out.op_tracker.*) {
                    inline else => |opt| @tagName(opt.op),
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

    while (queue.popOrNull()) |tensor| {
        if (written.contains(tensor)) {
            continue;
        }
        try written.putNoClobber(tensor, {});

        const depth: u32 = try Viz.blocksViz(tensor.block_tracker.curr_block, writer);
        try writer.print("\n", .{});

        switch (tensor.op_tracker.*) {
            .ArrayOp => |opt| {
                try switch (opt.op) {
                    .Cast => writer.print(
                        \\    {[op]s}_{[out]x}[label="{[op_type]s}.{[op]s}\nblock: {[block]s}\ndtype: {[data]s}"];
                        \\
                    , .{
                        .op_type = @typeName(@TypeOf(opt.op)),
                        .op = @tagName(opt.op),
                        .out = @intFromPtr(tensor),
                        .data = @tagName(tensor.dtype),
                        .block = @as([]const u8, if (tensor.block_tracker.curr_block) |block| block.name else "null"),
                    }),
                    .View => writer.print(
                        \\    {[op]s}_{[out]x}[label="{[op_type]s}.{[op]s}\nblock: {[block]s}\nshape {[shape]any}\nstrides {[strides]any}\noffset {[offset]d}"];
                        \\
                    , .{
                        .op_type = @typeName(@TypeOf(opt.op)),
                        .op = @tagName(opt.op),
                        .out = @intFromPtr(tensor),
                        .shape = tensor.shape[0..tensor.ndims],
                        .strides = tensor.strides[0..tensor.ndims],
                        .offset = tensor.offset,
                        .block = @as([]const u8, if (tensor.block_tracker.curr_block) |block| block.name else "null"),
                    }),
                    else => writer.print(
                        \\    {[op]s}_{[out]x}[label="{[op_type]s}.{[op]s}\nblock: {[block]s}\nshape: {[data]any}"];
                        \\
                    , .{
                        .op_type = @typeName(@TypeOf(opt.op)),
                        .op = @tagName(opt.op),
                        .out = @intFromPtr(tensor),
                        .data = tensor.shape[0..tensor.ndims],
                        .block = @as([]const u8, if (tensor.block_tracker.curr_block) |block| block.name else "null"),
                    }),
                };
            },
            .ReduceOp => |opt| try writer.print(
                \\    {[op]s}_{[out]x}[label="{[op_type]s}.{[op]s}\nblock: {[block]s}\n{[data]any}"];
                \\
            , .{
                .op_type = @typeName(@TypeOf(opt.op)),
                .op = @tagName(opt.op),
                .out = @intFromPtr(tensor),
                .data = opt.dims,
                .block = @as([]const u8, if (tensor.block_tracker.curr_block) |block| block.name else "null"),
            }),
            .InitOp => |opt| try switch (opt.op) {
                .Full => writer.print(
                    \\    {[op]s}_{[out]x}[label="{[op_type]s}.{[op]s}\nblock: {[block]s}\nvalue: {[value]s}"];
                    \\
                , .{
                    .op_type = @typeName(@TypeOf(opt.op)),
                    .op = @tagName(opt.op),
                    .out = @intFromPtr(tensor),
                    .block = @as([]const u8, if (tensor.block_tracker.curr_block) |block| block.name else "null"),
                    .value = opt.args.Full,
                }),
                .Range => writer.print(
                    \\    {[op]s}_{[out]x}[label="{[op_type]s}.{[op]s}\nblock: {[block]s}\nstart: {[start]s}, stop: {[stop]s}"];
                    \\
                , .{
                    .op_type = @typeName(@TypeOf(opt.op)),
                    .op = @tagName(opt.op),
                    .out = @intFromPtr(tensor),
                    .block = @as([]const u8, if (tensor.block_tracker.curr_block) |block| block.name else "null"),
                    .start = opt.args.Range.start,
                    .stop = opt.args.Range.stop,
                }),
                else => writer.print(
                    \\    {[op]s}_{[out]x}[label="{[op_type]s}.{[op]s}\nblock: {[block]s}"];
                    \\
                , .{
                    .op_type = @typeName(@TypeOf(opt.op)),
                    .op = @tagName(opt.op),
                    .out = @intFromPtr(tensor),
                    .block = @as([]const u8, if (tensor.block_tracker.curr_block) |block| block.name else "null"),
                }),
            },
            inline else => |opt| try writer.print(
                \\    {[op]s}_{[out]x}[label="{[op_type]s}.{[op]s}\nblock: {[block]s}"];
                \\
            , .{
                .op_type = @typeName(@TypeOf(opt.op)),
                .op = @tagName(opt.op),
                .out = @intFromPtr(tensor),
                .block = @as([]const u8, if (tensor.block_tracker.curr_block) |block| block.name else "null"),
            }),
        }

        switch (tensor.op_tracker.*) {
            .TernaryOp => |opt| {
                try Viz.inputViz(opt.a, tensor, writer);
                try Viz.inputViz(opt.b, tensor, writer);
                try Viz.inputViz(opt.c, tensor, writer);
            },
            .BinaryOp => |opt| {
                try Viz.inputViz(opt.a, tensor, writer);
                try Viz.inputViz(opt.b, tensor, writer);
            },
            .InitOp => {},
            inline else => |opt| {
                try Viz.inputViz(opt.a, tensor, writer);
            },
        }

        for (0..depth) |_| {
            try writer.print("    }}", .{});
        }

        try writer.print("\n", .{});

        switch (tensor.op_tracker.*) {
            inline else => |opt| {
                try writer.print(
                    \\    T_{[out]x}[label="dtype {[dtype]s}\nshape {[shape]any}\nstrides {[strides]any}\noffset {[offset]d}\n folded_constant {[fc]}"shape=box];
                    \\    {[op]s}_{[out]x}->T_{[out]x}[label="{[dtype]s}{[shape]any}"];
                    \\
                , .{
                    .op = @tagName(opt.op),
                    .out = @intFromPtr(tensor),
                    .dtype = @tagName(tensor.dtype),
                    .shape = tensor.shape[0..tensor.ndims],
                    .strides = tensor.strides[0..tensor.ndims],
                    .offset = tensor.offset,
                    .fc = tensor.folded_constant,
                });
            },
        }

        switch (tensor.op_tracker.*) {
            .TernaryOp => |opt| {
                try queue.append(opt.a);
                try queue.append(opt.b);
                try queue.append(opt.c);
            },
            .BinaryOp => |opt| {
                try queue.append(opt.a);
                try queue.append(opt.b);
            },
            .InitOp => {},
            inline else => |opt| {
                try queue.append(opt.a);
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

    var op_trackers_json = std.ArrayList(@import("tracker.zig").OpTracker.JsonFormat).init(allocator);
    defer op_trackers_json.deinit();

    var queue = std.ArrayList(*const AnyTensor).init(allocator);
    defer queue.deinit();

    for (entrypoints) |entry| {
        try queue.append(@ptrCast(entry));
    }

    while (queue.popOrNull()) |tensor| {
        if (tensors_json.contains(tensor)) {
            continue;
        }
        try tensors_json.put(tensor, tensor.toJsonFormat());
        try op_trackers_json.append(tensor.op_tracker.toJsonFormat(tensor));
        switch (tensor.op_tracker.*) {
            .TernaryOp => |opt| {
                try queue.append(opt.a);
                try queue.append(opt.b);
                try queue.append(opt.c);
            },
            .BinaryOp => |opt| {
                try queue.append(opt.a);
                try queue.append(opt.b);
            },
            .InitOp => {},
            inline else => |opt| {
                try queue.append(opt.a);
            },
        }
    }

    try std.json.stringify(.{
        .tensors = tensors_json.values(),
        .operations = op_trackers_json.items,
    }, .{}, writer);
    try writer.print("\n", .{});
}
