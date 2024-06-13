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
        fn inputViz(in: *const AnyTensor, out: *const AnyTensor, viz_writer: anytype) !void {
            try viz_writer.print(
                \\    T_{[in]x}->{[op]s}_{[out]x}[label="{[dtype]s}{[shape]any}"];
                \\
            , .{
                .op = switch (out.meta.instr) {
                    inline else => |info| utils.rawTagName(info.op),
                },
                .out = @intFromPtr(out),
                .in = @intFromPtr(in),
                .dtype = utils.rawTagName(in.dtype),
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

    for (anytensor_entrypoints[0..]) |entry| {
        try queue.append(@ptrCast(entry));
    }

    while (queue.popOrNull()) |out| {
        if (written.contains(out)) {
            continue;
        }
        try written.putNoClobber(out, {});

        switch (out.meta.instr) {
            .DataOp => |info| {
                try switch (info.op) {
                    .cast => writer.print(
                        \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\ndtype: {[data]s}"];
                        \\
                    , .{
                        .type = utils.rawTypeName(@TypeOf(info.op)),
                        .op = utils.rawTagName(info.op),
                        .out = @intFromPtr(out),
                        .data = utils.rawTagName(out.dtype),
                    }),
                    .view => writer.print(
                        \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\nshape {[shape]any}\nstrides {[strides]any}\noffset {[offset]d}"];
                        \\
                    , .{
                        .type = utils.rawTypeName(@TypeOf(info.op)),
                        .op = utils.rawTagName(info.op),
                        .out = @intFromPtr(out),
                        .shape = out.shape[0..out.ndims],
                        .strides = out.strides[0..out.ndims],
                        .offset = out.offset,
                    }),
                    else => writer.print(
                        \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\nshape: {[data]any}"];
                        \\
                    , .{
                        .type = utils.rawTypeName(@TypeOf(info.op)),
                        .op = utils.rawTagName(info.op),
                        .out = @intFromPtr(out),
                        .data = out.shape[0..out.ndims],
                    }),
                };
            },
            .InitOp => |info| try switch (info.op) {
                .full => writer.print(
                    \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\nvalue: {[value]s}"];
                    \\
                , .{
                    .type = utils.rawTypeName(@TypeOf(info.op)),
                    .op = utils.rawTagName(info.op),
                    .out = @intFromPtr(out),
                    .value = info.args.full,
                }),
                .input => writer.print(
                    \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\nlabel: {[label]s}"];
                    \\
                , .{
                    .type = utils.rawTypeName(@TypeOf(info.op)),
                    .op = utils.rawTagName(info.op),
                    .out = @intFromPtr(out),
                    .label = out.meta.label orelse "null",
                }),
                .range => writer.print(
                    \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\nstart: {[start]s}, stop: {[stop]s}"];
                    \\
                , .{
                    .type = utils.rawTypeName(@TypeOf(info.op)),
                    .op = utils.rawTagName(info.op),
                    .out = @intFromPtr(out),
                    .start = info.args.range.start,
                    .stop = info.args.range.stop,
                }),
                else => writer.print(
                    \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}"];
                    \\
                , .{
                    .type = utils.rawTypeName(@TypeOf(info.op)),
                    .op = utils.rawTagName(info.op),
                    .out = @intFromPtr(out),
                }),
            },
            inline else => |info| try writer.print(
                \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\nargs: {[args]any}"];
                \\
            , .{
                .type = utils.rawTypeName(@TypeOf(info.op)),
                .op = utils.rawTagName(info.op),
                .out = @intFromPtr(out),
                .args = info.args,
            }),
        }

        switch (out.meta.instr) {
            .InitOp => {},
            inline else => |info| {
                for (info.in) |in| {
                    try Viz.inputViz(in, out, writer);
                }
            },
        }

        try writer.print("\n", .{});

        switch (out.meta.instr) {
            inline else => |info| {
                try writer.print(
                    \\    T_{[out]x}[label="dtype: {[dtype]s}\nshape: {[shape]any}\nstrides: {[strides]any}\noffset: {[offset]d}\nconstant: {[fc]}\nlabel: {[label]s}"shape=box];
                    \\    {[op]s}_{[out]x}->T_{[out]x}[label="{[dtype]s}{[shape]any}"];
                    \\
                , .{
                    .op = utils.rawTagName(info.op),
                    .out = @intFromPtr(out),
                    .dtype = utils.rawTagName(out.dtype),
                    .shape = out.shape[0..out.ndims],
                    .strides = out.strides[0..out.ndims],
                    .offset = out.offset,
                    .fc = out.meta.constant,
                    .label = out.meta.label orelse "null",
                });
            },
        }

        switch (out.meta.instr) {
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

    while (queue.popOrNull()) |out| {
        if (tensors_json.contains(out)) {
            continue;
        }
        try tensors_json.put(out, out.toJson());
        try instructions_json.append(out.meta.instr.toJson(out));
        switch (out.meta.instr) {
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
        .operations = instructions_json.items,
    }, .{}, writer);
    try writer.print("\n", .{});
}
