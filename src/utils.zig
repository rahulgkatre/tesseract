const std = @import("std");
const comptimePrint = std.fmt.comptimePrint;
const ops = @import("ops.zig");
const dtypes = @import("dtypes.zig");
const anytensor = @import("anytensor.zig").anytensor;

pub const BlockTracker = extern struct {
    pub const Block = struct {
        name: []const u8,
        outer: ?*const Block,
        id: u32,
    };

    block_end: bool = false,
    curr_block: ?*const Block = null,
    next_block: ?*const Block = null,
    next_block_id: u32 = 0,

    pub fn enterBlock(bt: BlockTracker, block_name: []const u8) BlockTracker {
        return .{
            .curr_block = bt.curr_block,
            .next_block = &Block{
                .name = block_name,
                .outer = bt.next_block,
                .id = bt.next_block_id,
            },
            .next_block_id = bt.next_block_id + 1,
        };
    }

    pub fn joinBlock(bt: BlockTracker, block: ?*const Block) BlockTracker {
        return .{
            .curr_block = bt.curr_block,
            .next_block = block,
            .next_block_id = bt.next_block_id + 1,
        };
    }

    pub fn update(bt: BlockTracker) BlockTracker {
        return .{
            .curr_block = bt.next_block,
            .next_block = bt.next_block,
            .next_block_id = bt.next_block_id + 1,
        };
    }

    pub fn exitBlock(bt: BlockTracker) BlockTracker {
        return .{
            .block_end = true,
            .curr_block = bt.curr_block,
            .next_block = if (bt.next_block) |next_block| next_block.outer else @compileError("No block to end"),
            .next_block_id = bt.next_block_id + 1,
        };
    }
};

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
pub fn dataflowViz(entrypoints: []const *const anytensor, writer: anytype, allocator: std.mem.Allocator) !void {
    const RecursiveBlocksViz = struct {
        fn blocksViz(curr_block: ?*const BlockTracker.Block, viz_writer: anytype) !u32 {
            if (curr_block) |block| {
                const depth = try blocksViz(block.outer, viz_writer);
                try viz_writer.print("    subgraph cluster_{s}_{d} {{", .{ block.name, block.id });
                return depth + 1;
            } else {
                return 0;
            }
        }
    };

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

        const depth: u32 = try RecursiveBlocksViz.blocksViz(tensor.block_tracker.curr_block, writer);
        try writer.print("\n", .{});

        switch (tensor.record.*) {
            .ArrayOp => |rec| {
                try switch (rec.op) {
                    .Cast => writer.print(
                        \\    {[op]s}_{[out]x}[label="{[op_type]s}.{[op]s}\nblock: {[block]s}\ndtype: {[data]s}"];
                        \\
                    , .{
                        .op_type = @typeName(@TypeOf(rec.op)),
                        .op = @tagName(rec.op),
                        .out = @intFromPtr(tensor),
                        .data = @tagName(tensor.dtype),
                        .block = @as([]const u8, if (tensor.block_tracker.curr_block) |block| block.name else "null"),
                    }),
                    .View => writer.print(
                        \\    {[op]s}_{[out]x}[label="{[op_type]s}.{[op]s}\nblock: {[block]s}\nshape {[shape]any}\nstrides {[strides]any}\noffset {[offset]d}"];
                        \\
                    , .{
                        .op_type = @typeName(@TypeOf(rec.op)),
                        .op = @tagName(rec.op),
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
                        .op_type = @typeName(@TypeOf(rec.op)),
                        .op = @tagName(rec.op),
                        .out = @intFromPtr(tensor),
                        .data = tensor.shape[0..tensor.ndims],
                        .block = @as([]const u8, if (tensor.block_tracker.curr_block) |block| block.name else "null"),
                    }),
                };
            },
            .ReduceOp => |rec| try writer.print(
                \\    {[op]s}_{[out]x}[label="{[op_type]s}.{[op]s}\nblock: {[block]s}\n{[data]any}"];
                \\
            , .{
                .op_type = @typeName(@TypeOf(rec.op)),
                .op = @tagName(rec.op),
                .out = @intFromPtr(tensor),
                .data = rec.dims,
                .block = @as([]const u8, if (tensor.block_tracker.curr_block) |block| block.name else "null"),
            }),
            .InitOp => |rec| try switch (rec.op) {
                .Full => writer.print(
                    \\    {[op]s}_{[out]x}[label="{[op_type]s}.{[op]s}\nblock: {[block]s}\nvalue: {[value]s}"];
                    \\
                , .{
                    .op_type = @typeName(@TypeOf(rec.op)),
                    .op = @tagName(rec.op),
                    .out = @intFromPtr(tensor),
                    .block = @as([]const u8, if (tensor.block_tracker.curr_block) |block| block.name else "null"),
                    .value = rec.args.Full,
                }),
                .Range => writer.print(
                    \\    {[op]s}_{[out]x}[label="{[op_type]s}.{[op]s}\nblock: {[block]s}\nstart: {[start]s}, stop: {[stop]s}"];
                    \\
                , .{
                    .op_type = @typeName(@TypeOf(rec.op)),
                    .op = @tagName(rec.op),
                    .out = @intFromPtr(tensor),
                    .block = @as([]const u8, if (tensor.block_tracker.curr_block) |block| block.name else "null"),
                    .start = rec.args.Range.start,
                    .stop = rec.args.Range.stop,
                }),
                else => writer.print(
                    \\    {[op]s}_{[out]x}[label="{[op_type]s}.{[op]s}\nblock: {[block]s}"];
                    \\
                , .{
                    .op_type = @typeName(@TypeOf(rec.op)),
                    .op = @tagName(rec.op),
                    .out = @intFromPtr(tensor),
                    .block = @as([]const u8, if (tensor.block_tracker.curr_block) |block| block.name else "null"),
                }),
            },
            inline else => |rec| try writer.print(
                \\    {[op]s}_{[out]x}[label="{[op_type]s}.{[op]s}\nblock: {[block]s}"];
                \\
            , .{
                .op_type = @typeName(@TypeOf(rec.op)),
                .op = @tagName(rec.op),
                .out = @intFromPtr(tensor),
                .block = @as([]const u8, if (tensor.block_tracker.curr_block) |block| block.name else "null"),
            }),
        }

        switch (tensor.record.*) {
            .TernaryOp => |rec| {
                try writer.print(
                    \\    T_{[a]x}->{[op]s}_{[out]x}[label="{[a_dtype]s}{[a_shape]any}"];
                    \\    T_{[b]x}->{[op]s}_{[out]x}[label="{[b_dtype]s}{[b_shape]any}"];
                    \\    T_{[c]x}->{[op]s}_{[out]x}[label="{[c_dtype]s}{[c_shape]any}"];
                    \\    T_{[out]x}[label="dtype {[out_dtype]s}\nshape {[out_shape]any}\nstrides {[out_strides]any}\noffset {[out_offset]d}"shape=box];
                    \\    {[op]s}_{[out]x}->T_{[out]x}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\    
                , .{
                    .op = @tagName(rec.op),
                    .out = @intFromPtr(tensor),
                    .out_dtype = @tagName(tensor.dtype),
                    .out_shape = tensor.shape[0..tensor.ndims],
                    .out_strides = tensor.strides[0..tensor.ndims],
                    .out_offset = tensor.offset,
                    .a = @intFromPtr(rec.a),
                    .a_dtype = @tagName(rec.a.dtype),
                    .a_shape = rec.a.shape[0..rec.a.ndims],
                    .b = @intFromPtr(rec.b),
                    .b_dtype = @tagName(rec.b.dtype),
                    .b_shape = rec.b.shape[0..rec.b.ndims],
                    .c = @intFromPtr(rec.c),
                    .c_dtype = @tagName(rec.c.dtype),
                    .c_shape = rec.c.shape[0..rec.c.ndims],
                });
            },
            .BinaryOp => |rec| {
                try writer.print(
                    \\    T_{[a]x}->{[op]s}_{[out]x}[label="{[a_dtype]s}{[a_shape]any}"];
                    \\    T_{[b]x}->{[op]s}_{[out]x}[label="{[b_dtype]s}{[b_shape]any}"];
                    \\    T_{[out]x}[label="dtype {[out_dtype]s}\nshape {[out_shape]any}\nstrides {[out_strides]any}\noffset {[out_offset]d}"shape=box];
                    \\    {[op]s}_{[out]x}->T_{[out]x}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\
                , .{
                    .op = @tagName(rec.op),
                    .out = @intFromPtr(tensor),
                    .out_dtype = @tagName(tensor.dtype),
                    .out_shape = tensor.shape[0..tensor.ndims],
                    .out_strides = tensor.strides[0..tensor.ndims],
                    .out_offset = tensor.offset,
                    .a = @intFromPtr(rec.a),
                    .a_dtype = @tagName(rec.a.dtype),
                    .a_shape = rec.a.shape[0..rec.a.ndims],
                    .b = @intFromPtr(rec.b),
                    .b_dtype = @tagName(rec.b.dtype),
                    .b_shape = rec.b.shape[0..rec.b.ndims],
                });
            },
            .InitOp => |rec| {
                try writer.print(
                    \\    T_{[out]x}[label="dtype {[out_dtype]s}\nshape {[out_shape]any}\nstrides {[out_strides]any}\noffset {[out_offset]d}"shape=box];
                    \\    {[op]s}_{[out]x}->T_{[out]x}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\
                , .{
                    .op = @tagName(rec.op),
                    .out = @intFromPtr(tensor),
                    .out_dtype = @tagName(tensor.dtype),
                    .out_shape = tensor.shape[0..tensor.ndims],
                    .out_strides = tensor.strides[0..tensor.ndims],
                    .out_offset = tensor.offset,
                });
            },
            inline else => |rec| {
                try writer.print(
                    \\    T_{[a]x}->{[op]s}_{[out]x}[label="{[a_dtype]s}{[a_shape]any}"];
                    \\    T_{[out]x}[label="dtype {[out_dtype]s}\nshape {[out_shape]any}\nstrides {[out_strides]any}\noffset {[out_offset]d}"shape=box];
                    \\    {[op]s}_{[out]x}->T_{[out]x}[label="{[out_dtype]s}{[out_shape]any}"];
                    \\
                , .{
                    .op = @tagName(rec.op),
                    .out = @intFromPtr(tensor),
                    .out_dtype = @tagName(tensor.dtype),
                    .out_shape = tensor.shape[0..tensor.ndims],
                    .out_strides = tensor.strides[0..tensor.ndims],
                    .out_offset = tensor.offset,
                    .a = @intFromPtr(rec.a),
                    .a_dtype = @tagName(rec.a.dtype),
                    .a_shape = rec.a.shape[0..rec.a.ndims],
                });
            },
        }

        for (0..depth) |_| {
            try writer.print("    }}", .{});
        }

        try writer.print("\n", .{});

        switch (tensor.record.*) {
            .TernaryOp => |rec| {
                try queue.append(rec.a);
                try queue.append(rec.b);
                try queue.append(rec.c);
            },
            .BinaryOp => |rec| {
                try queue.append(rec.a);
                try queue.append(rec.b);
            },
            .InitOp => {},
            inline else => |rec| {
                try queue.append(rec.a);
            },
        }
    }

    try writer.print(
        \\}}
        \\
    , .{});
}

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
                try queue.append(rec.a);
                try queue.append(rec.b);
                try queue.append(rec.c);
            },
            .BinaryOp => |rec| {
                try queue.append(rec.a);
                try queue.append(rec.b);
            },
            .InitOp => {},
            inline else => |rec| {
                try queue.append(rec.a);
            },
        }
    }

    try std.json.stringify(.{
        .tensors = tensors_json.values(),
        .operations = records_json.items,
    }, .{}, writer);
    try writer.print("\n", .{});
}
