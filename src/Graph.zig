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

    pub fn viz(v: *Vertex, fused_out: bool) void {
        if (viz_hash_table.contains(v.id)) {
            return;
        }
        viz_hash_table.put(v.id, true) catch @panic("Out of memory");
        switch (v.edge) {
            .InitOp => |e| {
                std.debug.print("{s}_{d}[label=\"{s}\"];\n", .{ @tagName(e.op), v.id, @tagName(e.op) });
                std.debug.print("t{d}[label=\"t{d}:{s}{any}\"shape=box];\n", .{ v.id, v.id, @tagName(v.dtype), v.shape });
                std.debug.print("{s}_{d}->t{d};\n", .{ @tagName(e.op), v.id, v.id });
            },
            .MapOp => |e| {
                e.x.viz(e.fused_x);
                std.debug.print("{s}_{d}[label=\"{s}\"];\n", .{ @tagName(e.op), v.id, @tagName(e.op) });
                if (fused_out) {
                    switch (e.x.edge) {
                        inline else => |x_edge| std.debug.print("{s}_{d}->{s}_{d};\n", .{ @tagName(x_edge.op), e.x.id, @tagName(e.op), v.id }),
                    }
                } else {
                    std.debug.print("t{d}[label=\"t{d}:{s}{any}\"shape=box];\n", .{ v.id, v.id, @tagName(v.dtype), v.shape });
                    std.debug.print("t{d}->{s}_{d};\n", .{ e.x.id, @tagName(e.op), v.id });
                    std.debug.print("{s}_{d}->t{d};\n", .{ @tagName(e.op), v.id, v.id });
                }
            },
            .ZipOp => |e| {
                e.a.viz(e.fused_a);
                e.b.viz(e.fused_b);
                std.debug.print("{s}_{d}[label=\"{s}\"];\n", .{ @tagName(e.op), v.id, @tagName(e.op) });
                if (fused_out) {
                    switch (e.a.edge) {
                        inline else => |a_edge| std.debug.print("{s}_{d}->{s}_{d};\n", .{ @tagName(a_edge.op), e.a.id, @tagName(e.op), v.id }),
                    }
                    switch (e.b.edge) {
                        inline else => |b_edge| std.debug.print("{s}_{d}->{s}_{d};\n", .{ @tagName(b_edge.op), e.b.id, @tagName(e.op), v.id }),
                    }
                } else {
                    std.debug.print("t{d}[label=\"t{d}:{s}{any}\"shape=box];\n", .{ v.id, v.id, @tagName(v.dtype), v.shape });
                    std.debug.print("t{d}->{s}_{d};\n", .{ e.a.id, @tagName(e.op), v.id });
                    std.debug.print("t{d}->{s}_{d};\n", .{ e.b.id, @tagName(e.op), v.id });
                    std.debug.print("{s}_{d}->t{d};\n", .{ @tagName(e.op), v.id, v.id });
                }
            },
            .ReduceOp => |e| {
                e.x.viz(e.fused_x);
                std.debug.print("{s}_{d}[label=\"{s}({any})\"];\n", .{ @tagName(e.op), v.id, @tagName(e.op), e.dims });
                if (fused_out) {
                    switch (e.x.edge) {
                        inline else => |x_edge| std.debug.print("{s}_{d}->{s}_{d};\n", .{ @tagName(x_edge.op), e.x.id, @tagName(e.op), v.id }),
                    }
                } else {
                    std.debug.print("t{d}[label=\"t{d}:{s}{any}\"shape=box];\n", .{ v.id, v.id, @tagName(v.dtype), v.shape });
                    std.debug.print("t{d}->{s}_{d};\n", .{ e.x.id, @tagName(e.op), v.id });
                    std.debug.print("{s}_{d}->t{d};\n", .{ @tagName(e.op), v.id, v.id });
                }
            },
            .TypeOp => |e| {
                // TypeOps are always treated as fused with both preceding and succeeding ops because they do not correspond to any codegen in the final output
                e.x.viz(true);
                std.debug.print("{s}_{d}[label=\"{s}({any})\"];\n", .{ @tagName(e.op), v.id, @tagName(e.op), e.dims });
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
    std.debug.print("digraph G {{\n", .{});
    entrypoint.?.viz(false);
    std.debug.print("}}\n", .{});
}

pub fn fuse(v1: *Vertex, v2: *Vertex) void {
    switch (v2.edge) {
        .MapOp => |e| {
            std.testing.expectEqual(e.x.id, v1.id);
            e.fused_x = true;
        },
        .ZipOp => |e| {
            std.testing.expect(e.a.id == v1.id or e.b.id == v1.id);
            if (e.a.id == v1.id) {
                e.fused_a = true;
            }
            if (e.b.id == v1.id) {
                e.fused_b = true;
            }
        },
        .ReduceOp => |e| {
            std.testing.expectEqual(e.x.id, v1.id);
            e.fused_x = true;
        },
        else => {
            @panic("Unable to fuse");
        },
    }
}
