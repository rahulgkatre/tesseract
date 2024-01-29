const std = @import("std");
const tensor = @import("../tensor.zig");
const ops = @import("../ops.zig");
const Backend = @import("../backend.zig").Backend;
const comptimePrint = std.fmt.comptimePrint;
const codegen = @import("../codegen.zig");
const ZigCodegen = @This();

const header =
    \\const std = @import("std");
    \\fn main({s}) !void {{
    \\
;
pub fn write_header(_: *const ZigCodegen, writer: anytype) void {
    _ = writer.write(comptimePrint(header, .{""})) catch unreachable;
}
pub fn write_footer(_: *const ZigCodegen, writer: anytype) void {
    _ = writer.write("}\n") catch unreachable;
}

const storage_alloc =
    \\var t{d} = try allocator.alloc({s}, {d});
    \\
;
pub fn write_alloc(_: *const ZigCodegen, writer: anytype, id: usize, comptime dtype: type, size: usize) void {
    writer.print(storage_alloc, .{ id, @typeName(dtype), size }) catch unreachable;
}

const data_read = "t{d}[{s}]";
const no_broadcast_loop =
    \\for (0..{s}) |i| {{{{
    \\    t{s}[i] = {s};  
    \\}}}}
    \\
;

pub fn cast(
    _: *const ZigCodegen,
    writer: anytype,
    comptime new_dtype: type,
    x: anytype,
    out: *@TypeOf(x.*).Cast(new_dtype),
) !void {
    const Out: type = @TypeOf(out.*);
    const out_dtype: type = @TypeOf(x.*).dtype;
    const err_msg = comptimePrint("Cannot cast dtype {} to {}", .{ out_dtype, new_dtype });
    const fmt = comptimePrint(no_broadcast_loop, .{
        "{d}",
        "{d}",
        comptimePrint(switch (@typeInfo(new_dtype)) {
            .Float => switch (@typeInfo(out_dtype)) {
                .Int => "@floatFromInt({s})",
                .Float => "@floatCast({s})",
                else => @compileError(err_msg),
            },
            .Int => switch (@typeInfo(out_dtype)) {
                .Float => "@intFromFloat({s})",
                .Bool => "@intFromBool({s})",
                .Int => "@intCast({s})",
                else => @compileError(err_msg),
            },
            else => @compileError(err_msg),
        }, .{data_read}),
    });
    try writer.print(fmt, .{
        Out.size,
        out.id.?,
        x.id.?,
        "i",
    });
}

pub fn map(
    _: *const ZigCodegen,
    writer: anytype,
    comptime op: ops.MapOp,
    x: anytype,
    out: *@TypeOf(x.*),
) !void {
    const Out: type = @TypeOf(out.*);
    const fmt = comptimePrint(no_broadcast_loop, .{
        "{d}",
        "{d}",
        comptimePrint(switch (op) {
            .Neg => if (@typeInfo(Out.dtype) == .Bool) "!({s})" else "-({s})",
            .Log2 => "@log2({s})",
            .Exp2 => "@exp2({s})",
            .Sqrt => "@sqrt({s})",
            .Recip => "1.0 / ({s})",
            else => @compileError("Not implemented"),
        }, .{data_read}),
    });
    // @compileLog(fmt);
    try writer.print(fmt, .{
        Out.size,
        out.id.?,
        x.id.?,
        "i",
    });
}

fn zipOpFmt(comptime op: ops.ZipOp) []const u8 {
    return comptime switch (op) {
        .Add => "({s}) + ({s})",
        .Mul => "({s}) * ({s})",
        .Maximum => "@max({s}, {s})",
        .Lt => "({s}) < ({s})",
        .Eq => "({s}) == ({s})",
        .Xor => "({s}) ^ ({s})",
        else => @compileError("Not implemented"),
    };
}
pub fn zip(
    _: *const ZigCodegen,
    writer: anytype,
    comptime op: ops.ZipOp,
    a: anytype,
    b: anytype,
    out: *@TypeOf(a.*).Broadcast(@TypeOf(b.*)),
) !void {
    const Out: type = @TypeOf(out.*);
    const A: type = @TypeOf(a.*);
    const B: type = @TypeOf(b.*);
    const op_fmt = comptime zipOpFmt(op);
    if (A == B) {
        const fmt = comptimePrint(no_broadcast_loop, .{
            "{d}",
            "{d}",
            comptimePrint(op_fmt, .{
                data_read,
                data_read,
            }),
        });
        try writer.print(fmt, .{ Out.size, out.id.?, a.id.?, "i", b.id.?, "i" });
    } else {
        const broadcast_loop =
            \\for (0..{{d}}) |i| {{{{
            \\    const idx = {{s}};
            \\    t{{d}}[i] = {s};
            \\}}}}
            \\
        ;
        const fmt = comptimePrint(broadcast_loop, .{
            comptimePrint(op_fmt, .{
                data_read,
                data_read,
            }),
        });
        try writer.print(fmt, .{
            Out.size,
            codegen.posToIdx(Out, "i"),
            out.id.?,
            a.id.?,
            codegen.broadcastIdxToPos(A, Out, "idx"),
            b.id.?,
            codegen.broadcastIdxToPos(B, Out, "idx"),
        });
    }
}

pub fn reduce(
    _: *const ZigCodegen,
    writer: anytype,
    comptime op: ops.ReduceOp,
    x: anytype,
    comptime dim: ?u8,
    out: *@TypeOf(x.*).Reduce(dim),
) !void {
    const Out: type = @TypeOf(out.*);
    const X: type = @TypeOf(x.*);
    const op_fmt = comptimePrint(switch (op) {
        .Sum => zipOpFmt(.Add),
        .Max => zipOpFmt(.Maximum),
    }, .{ "acc", data_read });
    if (dim == null) {
        const reduce_all_loop =
            \\{{{{
            \\    var acc = t{{d}}[0];
            \\    for (1..{{d}}) |i| {{{{
            \\        acc = {s};
            \\    }}}}
            \\    t{{d}}[0] = acc;
            \\}}}}
            \\
        ;
        const fmt = comptimePrint(reduce_all_loop, .{
            op_fmt,
        });
        try writer.print(fmt, .{
            x.id.?,
            X.size,
            x.id.?,
            "i",
            out.id.?,
        });
    } else {
        const reduce_dim_loop =
            \\for (0..{{d}}) |i| {{{{
            \\    const idx = {{s}};
            \\    const pos = {{s}};
            \\    var acc = t{{d}}[pos];
            \\    for (1..{{d}}) |j| {{{{
            \\        acc = {s};
            \\    }}}}
            \\    t{{d}}[i] = acc;
            \\}}}}
            \\
        ;
        const fmt = comptimePrint(reduce_dim_loop, .{op_fmt});
        try writer.print(fmt, .{
            Out.size,
            codegen.posToIdx(@TypeOf(out.*), "i"),
            codegen.idxToPos(@TypeOf(x.*), "idx"),
            x.id.?,
            X.shape[dim.?],
            x.id.?,
            comptimePrint("pos + j * {d}", .{X.strides[dim.?]}),
            out.id.?,
        });
    }
}
