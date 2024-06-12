const std = @import("std");
const comptimePrint = std.fmt.comptimePrint;
const ops = @import("ops.zig");
const dtypes = @import("dtypes.zig");
const meta = @import("meta.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;
const utils = @import("utils.zig");

pub const debug_writer = std.io.Writer(std.fs.File, std.fs.File.WriteError, std.fs.File.write){ .context = std.io.getStdOut() };

/// Utility function for visualizing the full graph that is created at compile time, no scheduling is done yet
pub fn dataflowViz(entrypoints: anytype, writer: anytype, allocator: std.mem.Allocator) !void {
    const anytensor_entrypoints: [entrypoints.len]*const AnyTensor = entrypoints;
    const Viz = struct {
        fn inputViz(src: *const AnyTensor, dst: *const AnyTensor, viz_writer: anytype) !void {
            try viz_writer.print(
                \\    T_{[src]x}->{[op]s}_{[dst]x}[label="{[dtype]s}{[shape]any}"];
                \\
            , .{
                .op = switch (dst.meta.instr) {
                    inline else => |info| utils.rawTagName(info.op),
                },
                .dst = @intFromPtr(dst),
                .src = @intFromPtr(src),
                .dtype = utils.rawTagName(src.dtype),
                .shape = src.shape[0..src.ndims],
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

    for (anytensor_entrypoints[0..]) |entry| {
        try queue.append(@ptrCast(entry));
    }

    while (queue.popOrNull()) |dst| {
        if (written.contains(dst)) {
            continue;
        }
        try written.putNoClobber(dst, {});

        switch (dst.meta.instr) {
            .TypeOp => |info| {
                try switch (info.op) {
                    .cast => writer.print(
                        \\    {[op]s}_{[dst]x}[label="{[type]s}.{[op]s}\ndtype: {[data]s}"];
                        \\
                    , .{
                        .type = utils.rawTypeName(@TypeOf(info.op)),
                        .op = utils.rawTagName(info.op),
                        .dst = @intFromPtr(dst),
                        .data = utils.rawTagName(dst.dtype),
                    }),
                    .view => writer.print(
                        \\    {[op]s}_{[dst]x}[label="{[type]s}.{[op]s}\nshape {[shape]any}\nstrides {[strides]any}\noffset {[offset]d}"];
                        \\
                    , .{
                        .type = utils.rawTypeName(@TypeOf(info.op)),
                        .op = utils.rawTagName(info.op),
                        .dst = @intFromPtr(dst),
                        .shape = dst.shape[0..dst.ndims],
                        .strides = dst.strides[0..dst.ndims],
                        .offset = dst.offset,
                    }),
                    else => writer.print(
                        \\    {[op]s}_{[dst]x}[label="{[type]s}.{[op]s}\nshape: {[data]any}"];
                        \\
                    , .{
                        .type = utils.rawTypeName(@TypeOf(info.op)),
                        .op = utils.rawTagName(info.op),
                        .dst = @intFromPtr(dst),
                        .data = dst.shape[0..dst.ndims],
                    }),
                };
            },
            .InitOp => |info| try switch (info.op) {
                .full => writer.print(
                    \\    {[op]s}_{[dst]x}[label="{[type]s}.{[op]s}\nvalue: {[value]s}"];
                    \\
                , .{
                    .type = utils.rawTypeName(@TypeOf(info.op)),
                    .op = utils.rawTagName(info.op),
                    .dst = @intFromPtr(dst),
                    .value = info.args.full,
                }),
                .input => writer.print(
                    \\    {[op]s}_{[dst]x}[label="{[type]s}.{[op]s}\nlabel: {[label]s}"];
                    \\
                , .{
                    .type = utils.rawTypeName(@TypeOf(info.op)),
                    .op = utils.rawTagName(info.op),
                    .dst = @intFromPtr(dst),
                    .label = dst.meta.label orelse "null",
                }),
                .range => writer.print(
                    \\    {[op]s}_{[dst]x}[label="{[type]s}.{[op]s}\nstart: {[start]s}, stop: {[stop]s}"];
                    \\
                , .{
                    .type = utils.rawTypeName(@TypeOf(info.op)),
                    .op = utils.rawTagName(info.op),
                    .dst = @intFromPtr(dst),
                    .start = info.args.range.start,
                    .stop = info.args.range.stop,
                }),
                else => writer.print(
                    \\    {[op]s}_{[dst]x}[label="{[type]s}.{[op]s}"];
                    \\
                , .{
                    .type = utils.rawTypeName(@TypeOf(info.op)),
                    .op = utils.rawTagName(info.op),
                    .dst = @intFromPtr(dst),
                }),
            },
            inline else => |info| try writer.print(
                \\    {[op]s}_{[dst]x}[label="{[type]s}.{[op]s}\nargs: {[args]any}"];
                \\
            , .{
                .type = utils.rawTypeName(@TypeOf(info.op)),
                .op = utils.rawTagName(info.op),
                .dst = @intFromPtr(dst),
                .args = info.args,
            }),
        }

        switch (dst.meta.instr) {
            .InitOp => {},
            inline else => |info| {
                for (info.src) |src| {
                    try Viz.inputViz(src, dst, writer);
                }
            },
        }

        try writer.print("\n", .{});

        switch (dst.meta.instr) {
            inline else => |info| {
                try writer.print(
                    \\    T_{[dst]x}[label="dtype: {[dtype]s}\nshape: {[shape]any}\nstrides: {[strides]any}\noffset: {[offset]d}\nconstant: {[fc]}\nlabel: {[label]s}"shape=box];
                    \\    {[op]s}_{[dst]x}->T_{[dst]x}[label="{[dtype]s}{[shape]any}"];
                    \\
                , .{
                    .op = utils.rawTagName(info.op),
                    .dst = @intFromPtr(dst),
                    .dtype = utils.rawTagName(dst.dtype),
                    .shape = dst.shape[0..dst.ndims],
                    .strides = dst.strides[0..dst.ndims],
                    .offset = dst.offset,
                    .fc = dst.meta.constant,
                    .label = dst.meta.label orelse "null",
                });
            },
        }

        switch (dst.meta.instr) {
            .InitOp => {},
            inline else => |info| {
                for (info.src) |src| {
                    try queue.append(src);
                }
            },
        }
    }

    try writer.print(
        \\}}
        \\
    , .{});
}

pub fn dataflowJson(entrypoints: anytype, writer: anytype, allocator: std.mem.Allocator) !void {
    const anytensor_entrypoints: [entrypoints.len]*const AnyTensor = entrypoints;

    var tensors_json = std.AutoArrayHashMap(*const AnyTensor, AnyTensor.Json).init(allocator);
    defer tensors_json.deinit();

    var instructions_json = std.ArrayList(ops.Instruction.Json).init(allocator);
    defer instructions_json.deinit();

    var queue = std.ArrayList(*const AnyTensor).init(allocator);
    defer queue.deinit();

    for (anytensor_entrypoints) |entry| {
        try queue.append(@ptrCast(entry));
    }

    while (queue.popOrNull()) |dst| {
        if (tensors_json.contains(dst)) {
            continue;
        }
        try tensors_json.put(dst, dst.toJson());
        try instructions_json.append(dst.meta.instr.toJson(dst));
        switch (dst.meta.instr) {
            .InitOp => {},
            inline else => |info| {
                for (info.src) |src| {
                    try queue.append(src);
                }
            },
        }
    }

    try std.json.stringify(.{
        .tensors = tensors_json.values(),
        .operations = instructions_json.items,
    }, .{}, writer);
    try writer.print("\n", .{});
}
