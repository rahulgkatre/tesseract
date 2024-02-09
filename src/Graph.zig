const std = @import("std");
const ops = @import("ops.zig");
const tensor = @import("tensor.zig");
const utils = @import("utils.zig");
const Graph = @This();
const dtypes = @import("dtypes.zig");

var arena: std.heap.ArenaAllocator = undefined;
var allocator: std.mem.Allocator = undefined;
var cache: std.AutoHashMap(usize, usize) = undefined;
var ids: std.AutoHashMap(usize, usize) = undefined;
var nodes: std.AutoHashMap(usize, *Vertex) = undefined;
var entrypoint: ?*Vertex = null;

pub fn init() void {
    arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    allocator = arena.allocator();
    cache = std.AutoHashMap(usize, usize).init(allocator);
    ids = std.AutoHashMap(usize, usize).init(allocator);
    nodes = std.AutoHashMap(usize, *Vertex).init(allocator);
}

pub fn deinit() void {
    arena.deinit();
}

pub const Edge = union(ops.OpTypes) {
    MapOp: struct {
        op: ops.MapOp,
        x: *Vertex,
        fused_x: bool = false,
    },
    ZipOp: struct {
        op: ops.ZipOp,
        a: *Vertex,
        b: *Vertex,
        fused_a: bool = false,
        fused_b: bool = false,
    },
    ReduceOp: struct {
        op: ops.ReduceOp,
        x: *Vertex,
        dims: []const u8,
        fused_x: bool = false,
    },
    TypeOp: struct {
        op: ops.TypeOp,
        x: *Vertex,
        fused_x: bool = false,
    },
    InitOp: struct {
        op: ops.InitOp,
    },
};

var viz_hash_table: std.AutoHashMap(usize, bool) = undefined;
pub const Vertex = struct {
    id: usize,
    edge: Edge,
    fused: bool = false,

    // Tensor metadata which will be used for lowering and optimization
    ndims: u8,
    dtype: dtypes.DType,
    shape: []const usize,
    strides: []const usize,

    fn register(ptr: anytype, edge: Edge, comptime TensorType: type) !void {
        const key = @intFromPtr(ptr);
        if (!ids.contains(key)) {
            const id = ids.count();
            try ids.put(key, id);
            const vertex = try allocator.create(Vertex);
            vertex.* = .{
                .id = id,
                .edge = edge,
                .ndims = TensorType.ndims,
                .dtype = TensorType.dtype,
                .shape = TensorType.shape[0..],
                .strides = TensorType.strides[0..],
            };
            entrypoint = vertex;
            try nodes.put(id, vertex);
        }
    }

    pub fn new(ptr: anytype, edge: Edge, comptime TensorType: type) void {
        register(ptr, edge, TensorType) catch @panic("Out of memory");
    }

    pub fn get(ptr: anytype) *Vertex {
        return nodes.get(ids.get(@intFromPtr(ptr)).?).?;
    }

    fn print(v: *Vertex) void {
        switch (v.edge) {
            inline else => |e| {
                std.debug.print("t{d}[label=\"t{d}:{s}{any}\"shape=box];\n", .{ v.id, v.id, @tagName(v.dtype), v.shape });
                std.debug.print("{s}_{d}->t{d};\n", .{ @tagName(e.op), v.id, v.id });
            },
        }
    }

    pub fn viz(v: *Vertex, fused_out: bool) void {
        if (viz_hash_table.contains(v.id)) {
            return;
        }
        viz_hash_table.put(v.id, true) catch @panic("Out of memory");
        switch (v.edge) {
            .InitOp => |e| {
                std.debug.print("{s}_{d}[label=\"{s}\"];\n", .{ @tagName(e.op), v.id, @tagName(e.op) });
                if (!fused_out) {
                    v.print();
                }
            },
            .MapOp => |e| {
                e.x.viz(e.fused_x);
                std.debug.print("{s}_{d}[label=\"{s}\"];\n", .{ @tagName(e.op), v.id, @tagName(e.op) });
                if (e.fused_x) {
                    switch (e.x.edge) {
                        inline else => |x_edge| std.debug.print("{s}_{d}->{s}_{d};\n", .{ @tagName(x_edge.op), e.x.id, @tagName(e.op), v.id }),
                    }
                } else {
                    e.x.print();
                    std.debug.print("t{d}->{s}_{d};\n", .{ e.x.id, @tagName(e.op), v.id });
                }
                if (!fused_out) {
                    v.print();
                }
            },
            .ZipOp => |e| {
                e.a.viz(e.fused_a);
                e.b.viz(e.fused_b);
                std.debug.print("{s}_{d}[label=\"{s}\"];\n", .{ @tagName(e.op), v.id, @tagName(e.op) });
                if (e.fused_a) {
                    switch (e.a.edge) {
                        inline else => |a_edge| std.debug.print("{s}_{d}->{s}_{d}[label=\"A\"];\n", .{ @tagName(a_edge.op), e.a.id, @tagName(e.op), v.id }),
                    }
                } else {
                    std.debug.print("t{d}->{s}_{d}[label=\"A\"];\n", .{ e.a.id, @tagName(e.op), v.id });
                    e.a.print();
                }
                if (e.fused_b) {
                    switch (e.b.edge) {
                        inline else => |b_edge| std.debug.print("{s}_{d}->{s}_{d}[label=\"B\"];\n", .{ @tagName(b_edge.op), e.b.id, @tagName(e.op), v.id }),
                    }
                } else {
                    std.debug.print("t{d}->{s}_{d}[label=\"B\"];\n", .{ e.b.id, @tagName(e.op), v.id });
                    e.b.print();
                }
                if (!fused_out) {
                    v.print();
                }
            },
            .ReduceOp => |e| {
                e.x.viz(e.fused_x);
                std.debug.print("{s}_{d}[label=\"{s}({any})\"];\n", .{ @tagName(e.op), v.id, @tagName(e.op), e.dims });
                if (e.fused_x) {
                    switch (e.x.edge) {
                        inline else => |x_edge| std.debug.print("{s}_{d}->{s}_{d};\n", .{ @tagName(x_edge.op), e.x.id, @tagName(e.op), v.id }),
                    }
                } else {
                    std.debug.print("t{d}->{s}_{d};\n", .{ e.x.id, @tagName(e.op), v.id });
                    e.x.print();
                }
                if (!fused_out) {
                    v.print();
                }
            },
            .TypeOp => |e| {
                // TypeOps are always treated as fused with both preceding and succeeding ops because they do not correspond to any codegen in the final output
                e.x.viz(true);
                std.debug.print("{s}_{d}[label=\"{s}({s}, {any}, {any})\"];\n", .{ @tagName(e.op), v.id, @tagName(e.op), @tagName(e.x.dtype), e.x.shape, e.x.strides });
                switch (e.x.edge) {
                    inline else => |x_edge| std.debug.print("{s}_{d}->{s}_{d};\n", .{ @tagName(x_edge.op), e.x.id, @tagName(e.op), v.id }),
                }
            },
        }
    }
};

pub fn viz() void {
    viz_hash_table = std.AutoHashMap(usize, bool).init(allocator);
    defer {
        viz_hash_table.deinit();
        viz_hash_table = undefined;
    }
    std.debug.print("strict digraph G {{\n", .{});
    entrypoint.?.viz(false);
    std.debug.print("}}\n", .{});
}

fn fuse(fusion_target: *Vertex, fusing_vertex: *Vertex) !void {
    switch (fusing_vertex.edge) {
        .MapOp => |*e| {
            try std.testing.expectEqual(e.x.id, fusion_target.id);
            e.fused_x = true;
        },
        .ZipOp => |*e| {
            try std.testing.expect(e.a.id == fusion_target.id or e.b.id == fusion_target.id);
            if (e.a.id == fusion_target.id) {
                e.fused_a = true;
            }
            if (e.b.id == fusion_target.id) {
                e.fused_b = true;
            }
        },
        .ReduceOp => |*e| {
            try std.testing.expectEqual(e.x.id, fusion_target.id);
            e.fused_x = true;
        },
        else => {
            @panic("Unable to fuse");
        },
    }
}

fn softmax(x: anytype, comptime dim: u8) @TypeOf(x) {
    const max = x.max(null);
    const x_minus_max = x.sub(max);
    const exp = x_minus_max.exp();
    const sumexp = exp.sum(dim);
    const sm = x_minus_max.div(sumexp);
    return sm;
}

test "manual fuse" {
    const x = comptime tensor.Tensor(.f32, .{ 2, 16 }).full(0);
    const sm = comptime softmax(x, 1);

    Graph.init();
    defer Graph.deinit();
    sm.trace();

    const t9 = Graph.entrypoint.?;
    const t8 = t9.edge.ZipOp.b;
    try fuse(t8, t9);
    const t7 = t8.edge.MapOp.x;
    try fuse(t7, t8);
    const t6 = t7.edge.ReduceOp.x;
    try fuse(t6, t7);
    const t5 = t6.edge.MapOp.x;
    try fuse(t5, t6);

    const t3 = t9.edge.ZipOp.a;
    try fuse(t3, t9);
    try fuse(t3, t5);
    const t2 = t3.edge.ZipOp.b;
    try fuse(t2, t3);
    const t1 = t2.edge.MapOp.x;
    try fuse(t1, t2);

    Graph.viz();
}
