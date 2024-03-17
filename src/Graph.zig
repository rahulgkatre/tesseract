const std = @import("std");
const ops = @import("ops.zig");
const utils = @import("utils.zig");
const Graph = @This();
const dtypes = @import("dtypes.zig");
const Dim = @import("symbolic.zig").Dim;

var gpa: std.heap.GeneralPurposeAllocator(.{}) = undefined;
var arena: std.heap.ArenaAllocator = undefined;
var tensors: std.AutoArrayHashMap(usize, *TensorNode) = undefined;
var reduction_groups: std.AutoHashMap(usize, bool) = undefined;
var dag_sinks: std.AutoArrayHashMap(usize, *TensorNode) = undefined;

pub fn dagSinks() []*TensorNode {
    return dag_sinks.values();
}

pub fn init() void {
    gpa = .{};
    arena = std.heap.ArenaAllocator.init(gpa.allocator());
    tensors = std.AutoArrayHashMap(usize, *TensorNode).init(arena.allocator());
    reduction_groups = std.AutoHashMap(usize, bool).init(arena.allocator());
    dag_sinks = std.AutoArrayHashMap(usize, *TensorNode).init(arena.allocator());
}

pub fn deinit() void {
    arena.deinit();
}

/// Build the computation graph for a tensor.
/// Any new nodes are added to the global computation graph
/// by recursively calling each tensor's `trace_fn` callback.
pub fn trace(tensor: anytype) void {
    switch (@typeInfo(@TypeOf(tensor))) {
        .Pointer => tensor.trace_fn(tensor),
        else => @compileError("Must pass a tensor pointer to Graph.trace()"),
    }
    const key = @intFromPtr(tensor);
    const node = TensorNode.get(key);
    node.global = true;
    dag_sinks.putNoClobber(key, node) catch unreachable;
}

/// Each tensor trace callback uses this function to add its dependencies (input), operation (op), and result (output)
/// to the computation graph
pub fn addOp(comptime op: ops.GraphOp, input: anytype, output: anytype, comptime args: anytype) void {
    _ = OpNode.init(op, input, output, args);
}

pub const OpNode = union(ops.OpTypes) {
    pub const Input = struct {
        tensor: *TensorNode,
        fused: bool = false,

        fn init(ptr: anytype) Input {
            const tensor = TensorNode.get(ptr);
            tensor.consumer_count += 1;
            return .{
                .tensor = TensorNode.get(ptr),
            };
        }

        pub fn jsonStringify(input: Input, write_stream: anytype) !void {
            try write_stream.write(input.tensor.ptr);
        }
    };
    pub const Output = struct {
        tensor: *TensorNode,
        fn init(ptr: anytype) Output {
            return .{
                .tensor = TensorNode.get(ptr),
            };
        }

        pub fn jsonStringify(output: Output, write_stream: anytype) !void {
            try write_stream.write(output.tensor.ptr);
        }
    };
    MapOp: struct {
        op: ops.MapOp,
        x: Input,
        out: Output,
        // label: []const u8,
    },
    ZipOp: struct {
        op: ops.ZipOp,
        a: Input,
        b: Input,
        out: Output,
        // label: []const u8,
    },
    ReduceOp: struct {
        op: ops.ReduceOp,
        x: Input,
        dims: []const bool,
        out: Output,
        // label: []const u8,
    },
    TypeOp: struct {
        op: ops.TypeOp,
        x: Input,
        out: Output,
        // label: []const u8,
    },
    InitOp: struct {
        op: ops.InitOp,
        args: ops.InitOp.Args,
        out: Output,
        // label: []const u8,
    },

    fn init(comptime op: ops.GraphOp, input: anytype, output: anytype, comptime args: anytype) OpNode {
        // const Out = @TypeOf(output.*);
        switch (op) {
            .MapOp, .TypeOp, .ReduceOp => {
                if (!tensors.contains(@intFromPtr(input.x))) input.x.trace_fn(input.x);
                TensorNode.create(input.x);
                TensorNode.create(output);
            },
            .ZipOp => {
                if (!tensors.contains(@intFromPtr(input.a))) input.a.trace_fn(input.a);
                if (!tensors.contains(@intFromPtr(input.b))) input.b.trace_fn(input.b);
                TensorNode.create(input.a);
                TensorNode.create(input.b);
                TensorNode.create(output);
            },
            else => {
                TensorNode.create(output);
            },
        }
        const out = TensorNode.get(output);
        out.op_node = switch (op) {
            .MapOp => |map_op| @unionInit(OpNode, @tagName(op), .{
                .op = map_op,
                .x = Input.init(input.x),
                .out = Output.init(output),
                // .label = std.fmt.comptimePrint("{s}", .{@tagName(map_op)}),
            }),
            .ZipOp => |zip_op| @unionInit(OpNode, @tagName(op), .{
                .op = zip_op,
                .a = Input.init(input.a),
                .b = Input.init(input.b),
                .out = Output.init(output),
                // .label = std.fmt.comptimePrint("{s}", .{@tagName(zip_op)}),
            }),
            .ReduceOp => |reduce_op| blk: {
                reduction_groups.put(out.group.?, true) catch unreachable;
                break :blk @unionInit(OpNode, @tagName(op), .{
                    .op = reduce_op,
                    .x = Input.init(input.x),
                    .out = Output.init(output),
                    .dims = args.dims,
                    // .label = std.fmt.comptimePrint("{s}{any}", .{ @tagName(reduce_op), @as([]const bool, args.dims) }),
                });
            },
            .TypeOp => |type_op| @unionInit(OpNode, @tagName(op), .{
                .op = type_op,
                .x = Input.init(input.x),
                // .label = switch (type_op) {
                //     .AsType => std.fmt.comptimePrint("{s}{any}", .{ @tagName(type_op), Out.dtype }),
                //     .View, .Broadcast => std.fmt.comptimePrint("{s}{any}", .{ @tagName(type_op), Out.shape }),
                //     .AsStrided => std.fmt.comptimePrint("{s}{{{any},{any}}}", .{ @tagName(type_op), Out.shape, Out.strides }),
                // },
                .out = Output.init(output),
            }),
            .InitOp => |init_op| blk: {
                if (init_op == .Input) {
                    out.group = null;
                    out.global = true;
                }
                break :blk @unionInit(OpNode, @tagName(op), .{
                    .op = init_op,
                    .args = args,
                    .out = Output.init(output),
                    // .label = std.fmt.comptimePrint("{s}", .{@tagName(init_op)}),
                });
            },
        };
        return out.op_node;
    }
};

pub const TensorNode = struct {
    ptr: usize,
    dtype: dtypes.DType,
    ndims: u8,
    shape: []const Dim,
    strides: []const Dim,

    // label: []const u8,
    uid: u64,
    group: ?u64 = null,
    consumer_count: u16 = 0,
    global: bool = false,
    op_node: OpNode = undefined,

    const JsonCompatible = struct {
        ptr: usize,
        dtype: dtypes.DType,
        ndims: u8,
        shape: []const Dim,
        strides: []const Dim,
    };

    pub fn jsonStringify(self: *const TensorNode, write_stream: anytype) !void {
        const compatible: JsonCompatible = .{
            .ptr = self.ptr,
            .dtype = self.dtype,
            .ndims = self.ndims,
            .shape = self.shape,
            .strides = self.strides,
        };
        try write_stream.write(compatible);
    }

    pub fn isCached(tensor: *const TensorNode) bool {
        return (tensor.consumer_count > 1 and tensor.group != tensor.uid);
    }

    pub fn memoryView(self: *const TensorNode) u64 {
        switch (self.op_node) {
            .InitOp => return self.uid,
            .ZipOp => |op_node| {
                if (!op_node.a.fused and !op_node.b.fused) {
                    return self.uid;
                }
                const a_mv = op_node.a.tensor.memoryView();
                const b_mv = op_node.b.tensor.memoryView();
                if (op_node.a.fused and !op_node.b.fused and !op_node.a.tensor.isCached()) {
                    return a_mv;
                } else if (op_node.b.fused and !op_node.a.fused and !op_node.b.tensor.isCached()) {
                    return b_mv;
                } else if (!op_node.a.tensor.isCached() and !op_node.b.tensor.isCached()) {
                    return @min(a_mv, b_mv);
                } else {
                    return self.uid;
                }
            },
            inline else => |op_node| return if (op_node.x.fused and !op_node.x.tensor.isCached()) op_node.x.tensor.memoryView() else self.uid,
        }
    }

    pub fn get(ptr: anytype) *TensorNode {
        return switch (@typeInfo(@TypeOf(ptr))) {
            .Pointer => tensors.get(@intFromPtr(ptr)).?,
            .Int => tensors.get(ptr).?,
            else => @panic("Must use a tensor pointer or int from tensor pointer"),
        };
    }

    fn create(tensor: anytype) void {
        // const Tensor = @TypeOf(tensor.*);
        // const fields = @typeInfo
        const ptr = @intFromPtr(tensor);
        if (!tensors.contains(ptr)) {
            const tensor_node = arena.allocator().create(TensorNode) catch unreachable;
            tensor_node.* = .{
                .ptr = ptr,
                .dtype = tensor.dtype,
                .ndims = tensor.ndims,
                .shape = tensor.shape[0..tensor.ndims],
                .strides = tensor.strides[0 .. tensor.ndims + 1],
                .group = tensors.count(),
                .uid = tensors.count(),
                // .label = std.fmt.comptimePrint("{s}{any}", .{ @tagName(tensor.dtype), tensor.shape }),
            };
            tensors.putNoClobber(ptr, tensor_node) catch unreachable;
        }
    }
};

pub fn jsonStringify(_: Graph, write_stream: anytype) !void {
    const operations: []OpNode = gpa.allocator().alloc(OpNode, tensors.count()) catch unreachable;
    defer gpa.allocator().free(operations);
    for (tensors.values(), operations) |t, *operation| {
        operation.* = t.op_node;
    }
    try write_stream.write(.{
        .tensors = tensors.values(),
        .operations = operations,
    });
}
