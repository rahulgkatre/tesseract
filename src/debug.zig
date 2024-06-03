const std = @import("std");
const comptimePrint = std.fmt.comptimePrint;
const ops = @import("ops.zig");
const dtypes = @import("dtypes.zig");
const meta = @import("meta.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;
const utils = @import("utils.zig");

pub const debug_writer = std.io.Writer(std.fs.File, std.fs.File.WriteError, std.fs.File.write){ .context = std.io.getStdOut() };

/// Utility function for visualizing the full graph that is created at compile time, no scheduling is done yet
pub fn dataflowViz(entrypoints: anytype, writer: anytype, allocator: std.mem.Allocator, draw_groups: bool) !void {
    const anytensor_entrypoints: [entrypoints.len]*const AnyTensor = entrypoints;
    const Viz = struct {
        fn opGroupViz(curr: ?*const meta.OpGroupTracker.OpGroup, viz_writer: anytype) !u32 {
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

        const depth: u32 = blk: {
            var depth: u32 = 0;
            if (draw_groups) {
                depth = try Viz.opGroupViz(out.meta.op_group_tracker.curr, writer);
                try writer.print("\n", .{});
            }
            break :blk depth;
        };

        switch (out.meta.op_tracker) {
            .TypeOp => |info| {
                try switch (info.op) {
                    .Cast => writer.print(
                        \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\ngroup: {[group]s}\ndtype: {[data]s}"];
                        \\
                    , .{
                        .type = utils.rawTypeName(@TypeOf(info.op)),
                        .op = utils.rawTagName(info.op),
                        .out = @intFromPtr(out),
                        .data = utils.rawTagName(out.dtype),
                        .group = @as([]const u8, if (out.meta.op_group_tracker.curr) |group| group.name else "null"),
                    }),
                    .View => writer.print(
                        \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\ngroup: {[group]s}\nshape {[shape]any}\nstrides {[strides]any}\noffset {[offset]d}"];
                        \\
                    , .{
                        .type = utils.rawTypeName(@TypeOf(info.op)),
                        .op = utils.rawTagName(info.op),
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
                        .type = utils.rawTypeName(@TypeOf(info.op)),
                        .op = utils.rawTagName(info.op),
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
                    .type = utils.rawTypeName(@TypeOf(info.op)),
                    .op = utils.rawTagName(info.op),
                    .out = @intFromPtr(out),
                    .group = @as([]const u8, if (out.meta.op_group_tracker.curr) |group| group.name else "null"),
                    .value = info.args.Full.value,
                }),
                .Input => writer.print(
                    \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\ngroup: {[group]s}\nlabel: {[label]s}"];
                    \\
                , .{
                    .type = utils.rawTypeName(@TypeOf(info.op)),
                    .op = utils.rawTagName(info.op),
                    .out = @intFromPtr(out),
                    .group = @as([]const u8, if (out.meta.op_group_tracker.curr) |group| group.name else "null"),
                    .label = out.meta.label orelse "null",
                }),
                .Range => writer.print(
                    \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\ngroup: {[group]s}\nstart: {[start]s}, stop: {[stop]s}"];
                    \\
                , .{
                    .type = utils.rawTypeName(@TypeOf(info.op)),
                    .op = utils.rawTagName(info.op),
                    .out = @intFromPtr(out),
                    .group = @as([]const u8, if (out.meta.op_group_tracker.curr) |group| group.name else "null"),
                    .start = info.args.Range.start,
                    .stop = info.args.Range.stop,
                }),
                else => writer.print(
                    \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\ngroup: {[group]s}"];
                    \\
                , .{
                    .type = utils.rawTypeName(@TypeOf(info.op)),
                    .op = utils.rawTagName(info.op),
                    .out = @intFromPtr(out),
                    .group = @as([]const u8, if (out.meta.op_group_tracker.curr) |group| group.name else "null"),
                }),
            },
            inline else => |info| try writer.print(
                \\    {[op]s}_{[out]x}[label="{[type]s}.{[op]s}\ngroup: {[group]s}\nargs: {[args]any}"];
                \\
            , .{
                .type = utils.rawTypeName(@TypeOf(info.op)),
                .op = utils.rawTagName(info.op),
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

pub fn dataflowJson(entrypoints: anytype, writer: anytype, allocator: std.mem.Allocator) !void {
    const anytensor_entrypoints: [entrypoints.len]*const AnyTensor = entrypoints;

    var tensors_json = std.AutoArrayHashMap(*const AnyTensor, AnyTensor.Json).init(allocator);
    defer tensors_json.deinit();

    var op_trackers_json = std.ArrayList(@import("meta.zig").OpTracker.Json).init(allocator);
    defer op_trackers_json.deinit();

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
