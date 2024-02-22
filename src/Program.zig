const ops = @import("ops.zig");
const std = @import("std");
const codegen = @import("codegen.zig");
const Graph = @import("Graph.zig");
const Program = @This();

var arena: std.heap.ArenaAllocator = undefined;
var allocator: std.mem.Allocator = undefined;
var body: Body = undefined;

pub fn init() void {
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    allocator = arena.allocator();
}

pub fn deinit() void {
    arena.deinit();
    arena = undefined;
    allocator = undefined;
}

/// Lower the node (and any nodes fused with it)
/// to a loop nest representation
pub fn loops(v: *Graph.Vertex) *Loop {
    const statement: Statement = switch (v.edge) {
        .InitOp => |edge| .{ .InitOp = .{ .op = edge.op, .out = v } },
        .ZipOp => |edge| .{ .ZipOp = .{ .op = edge.op, .a = edge.a, .b = edge.b, .out = v } },
        .MapOp => |edge| .{ .MapOp = .{ .op = edge.op, .x = edge.x, .out = v } },
        .ReduceOp => |edge| .{ .ReduceOp = .{ .op = edge.op, .x = edge.x, .out = v } },
        .TypeOp => |edge| .{ .TypeOp = .{ .op = edge.op, .x = edge.x, .out = v } },
    };

    const loop: *Loop = build_loop: {
        const root_loop: *Loop = allocator.create(Loop) catch unreachable;
        var curr_loop = root_loop;
        for (0..v.tensor.ndims) |d| {
            curr_loop.* = .{
                .upper_bound = switch (v.edge) {
                    .ReduceOp => |edge| edge.x.tensor.shape[d],
                    else => v.tensor.shape[d],
                },
                .node = v,
                .dim = d,
                .acc = switch (v.edge) {
                    .ReduceOp => |edge| edge.dims[d],
                    else => false,
                },
                .body = .{
                    .contents = std.MultiArrayList(Body.Content){},
                },
                .prev = null,
            };
            if (d != v.tensor.ndims - 1) {
                const next_loop: *Loop = allocator.create(Loop) catch unreachable;
                curr_loop.body.contents.append(allocator, .{ .Loop = next_loop }) catch unreachable;
                curr_loop = next_loop;
            } else {
                curr_loop.body.contents.append(allocator, .{ .Statement = statement }) catch unreachable;
            }

            // if (curr_loop.body.inner_loops == null) {
            //     // If there are no inner loops create a new one
            //     curr_loop.body.inner_loops = std.ArrayList(AffineLoop).init(allocator);
            //     const next_loop: AffineLoop = .{
            //         .upper_bound = v.tensor.shape[d],
            //         .loop_var = std.fmt.allocPrint(allocator, "i{d}_d{d}", .{ v.id, d }) catch unreachable,
            //         .acc = switch (v.edge) {
            //             .ReduceOp => |edge| edge.dims[d],
            //             else => false,
            //         },
            //         .body = .{
            //             .inner_loops = null,
            //             .exprs = null,
            //         },
            //     };
            //     curr_loop.body.inner_loops.?.append(next_loop) catch unreachable;
            //     curr_loop = next_loop;
            // } else {
            //     // Otherwise try to find a loop with the same bounds
            //     std.debug.print("Finding a inner loop to use\n", .{});

            //     var found_next_loop = false;
            //     for (curr_loop.body.inner_loops.?.items) |loop| {
            //         if (loop.upper_bound == v.tensor.shape[d]) {
            //             curr_loop = loop;
            //             found_next_loop = true;
            //         }
            //     }
            //     if (!found_next_loop) {
            //         const next_loop: AffineLoop = .{
            //             .upper_bound = v.tensor.shape[d],
            //             .loop_var = std.fmt.allocPrint(allocator, "i{d}_d{d}", .{ v.id, d }) catch unreachable,
            //             .acc = switch (v.edge) {
            //                 .ReduceOp => |edge| edge.dims[d],
            //                 else => false,
            //             },
            //             .body = .{
            //                 .inner_loops = null,
            //                 .exprs = null,
            //             },
            //         };
            //         curr_loop.body.inner_loops.?.append(next_loop) catch unreachable;
            //         curr_loop = next_loop;
            //     }
            // }
        }
        break :build_loop root_loop;
    };
    switch (v.edge) {
        .InitOp => {},
        .ZipOp => |edge| {
            const b_loop = edge.b.loops();
            const a_loop = edge.a.loops();
            loop.prev = a_loop;
            a_loop.prev = b_loop;
        },
        .TypeOp => |edge| {
            switch (edge.op) {
                .AsType => {
                    const x_loop = edge.x.loops();
                    loop.prev = x_loop;
                },
                else => {
                    return edge.x.loops();
                },
            }
        },
        inline else => |edge| {
            const x_loop = edge.x.loops();
            loop.prev = x_loop;
        },
    }
    return loop;
}

/// Stameent of the form y = f(x)
/// y can either be a value in an array or a variable
/// f(x) is an arithmetic operation on a value in an array or a variable
pub const Statement = union(ops.GraphOps) {
    MapOp: struct {
        op: ops.MapOp,
        x: *Graph.Vertex,
        out: *Graph.Vertex,
    },
    ZipOp: struct {
        op: ops.ZipOp,
        a: *Graph.Vertex,
        b: *Graph.Vertex,
        out: *Graph.Vertex,
    },
    ReduceOp: struct {
        op: ops.ReduceOp,
        x: *Graph.Vertex,
        out: *Graph.Vertex,
    },
    TypeOp: struct {
        op: ops.TypeOp,
        x: *Graph.Vertex,
        out: *Graph.Vertex,
    },
    InitOp: struct {
        op: ops.InitOp,
        out: *Graph.Vertex,
    },
};

/// Abstractions for lowering Graph.Node into a loop which can be codegened
/// loop structs will be stored in a list (program) where order is exact order of code
/// loops are defined as a grammar, every loop has a header and a body
pub const Loop = struct {
    upper_bound: usize,
    node: *Graph.Vertex,
    dim: usize,
    acc: bool = false,
    body: Body,
    prev: ?*Loop,
};

pub const Body = struct {
    pub const Content = union(enum) {
        Loop: *Loop,
        Statement: Statement,
    };
    contents: std.MultiArrayList(Content),
};
