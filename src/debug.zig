const std = @import("std");
const comptimePrint = std.fmt.comptimePrint;
const ops = @import("ops.zig");
const dtypes = @import("dtypes.zig");
const utils = @import("utils.zig");

const AnyTensor = @import("tensor/tensor.zig").AnyTensor;

pub const debug_writer = std.io.Writer(std.fs.File, std.fs.File.WriteError, std.fs.File.write){ .context = std.io.getStdOut() };

/// Utility function for visualizing the full graph that is created at compile time, no scheduling is done yet
pub fn dataflowViz(entrypoints: anytype, writer: anytype, allocator: std.mem.Allocator) !void {
    const Viz = struct {
        fn inputViz(in: *const AnyTensor, out: *const AnyTensor, viz_writer: anytype) !void {
            try viz_writer.print(
                \\    T_{[in]x}->{[op]s}_{[out]x}[label="{[dtype]s}{[shape]any}"];
                \\
            , .{
                .op = switch (out.instr.*) {
                    inline else => |info| utils.rawTagName(info.op),
                },
                .out = @intFromPtr(out),
                .in = @intFromPtr(in),
                .dtype = utils.rawTagName(in.layout.dtype),
                .shape = in.layout.shape,
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

    inline for (0..entrypoints.len) |i| {
        try queue.append(@ptrCast(entrypoints[i]));
    }

    while (queue.popOrNull()) |out| {
        if (written.contains(out)) {
            continue;
        }
        try written.putNoClobber(out, {});

        switch (out.instr.*) {
            .DataOp => |info| {
                try switch (info.op) {
                    .cast => writer.print(
                        \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\ndtype: {[data]s}"];
                        \\
                    , .{
                        .type = utils.rawTypeName(@TypeOf(info.op)),
                        .op = utils.rawTagName(info.op),
                        .out = @intFromPtr(out),
                        .data = utils.rawTagName(out.layout.dtype),
                    }),
                    .view => writer.print(
                        \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\nshape {[shape]any}\nstrides {[strides]any}\noffset {[offset]d}"];
                        \\
                    , .{
                        .type = utils.rawTypeName(@TypeOf(info.op)),
                        .op = utils.rawTagName(info.op),
                        .out = @intFromPtr(out),
                        .shape = out.layout.shape,
                        .strides = out.layout.strides,
                        .offset = out.layout.offset,
                    }),
                    else => writer.print(
                        \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\nshape: {[data]any}"];
                        \\
                    , .{
                        .type = utils.rawTypeName(@TypeOf(info.op)),
                        .op = utils.rawTagName(info.op),
                        .out = @intFromPtr(out),
                        .data = out.layout.shape,
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
                    .label = out.labels.name orelse "null",
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

        switch (out.instr.*) {
            .InitOp => {},
            inline else => |info| {
                for (info.in) |in| {
                    try Viz.inputViz(in, out, writer);
                }
            },
        }

        try writer.print("\n", .{});

        switch (out.instr.*) {
            inline else => |info| {
                try writer.print(
                    \\    T_{[out]x}[label="dtype: {[dtype]s}\nshape: {[shape]any}\nstrides: {[strides]any}\noffset: {[offset]d}\nconstant: {[constant]}\nlabel: {[label]s}"shape=box];
                    \\    {[op]s}_{[out]x}->T_{[out]x}[label="{[dtype]s}{[shape]any}"];
                    \\
                , .{
                    .op = utils.rawTagName(info.op),
                    .out = @intFromPtr(out),
                    .dtype = utils.rawTagName(out.layout.dtype),
                    .shape = out.layout.shape,
                    .strides = out.layout.strides,
                    .offset = out.layout.offset,
                    .constant = out.autograd.constant,
                    .label = out.labels.name orelse "null",
                });
            },
        }

        switch (out.instr.*) {
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
        try instructions_json.append(out.instr.toJson(out));
        switch (out.instr) {
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
