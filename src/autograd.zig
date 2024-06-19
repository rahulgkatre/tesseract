const std = @import("std");
const tensor = @import("tensor.zig");
const dtypes = @import("dtypes.zig");
const ops = @import("ops.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;
const utils = @import("utils.zig");

const asTensor = tensor.asTensor;
const TensorTypeOf = tensor.TensorTypeOf;
const TensorTuple = tensor.TensorTuple;
const IntTensor = dtypes.IntTensor;
const BoolTensor = dtypes.BoolTensor;
const FloatTensor = dtypes.FloatTensor;
const F = @import("functions.zig");

pub const Trace = utils.ComptimeLinkedList(*const AnyTensor);

pub fn backprop(comptime x: anytype) []const *const AnyTensor {
    const initial_grad = tensor.asTensor(1.0);
    const params = utils.paramsOf(x);
    _, const updated_params = applyGradFn(x, initial_grad, null, params);
    return updated_params;
}

pub fn applyGradFn(comptime x: anytype, comptime grad: anytype, trace: ?Trace, params: []const *const AnyTensor) std.meta.Tuple(&.{ Trace, []const *const AnyTensor }) {
    const gradFn: *const fn (anytype, ?Trace, []const *const AnyTensor) Trace = @ptrCast(x.meta.grad_fn);
    return gradFn(grad, trace, params);
}

pub fn noOpGradFn(_: anytype, trace: ?Trace, _: []const *const AnyTensor) Trace {
    return trace.?;
}

pub fn unaryGradFn(comptime op: ops.UnaryOp, a: anytype, grad: anytype, trace: ?Trace, params: []const *const AnyTensor) std.meta.Tuple(&.{ Trace, []const *const AnyTensor }) {
    const local_grad = switch (op) {
        .exp2 => F.exp2(a).mul(F.LN_2),
        .log2 => F.div(F.INV_LN_2, a),
        .neg => F.neg(a),
        .recip => F.mul(a, a).neg(),
        .sqrt => F.div(0.5, F.sqrt(a)),
        .sin => F.add(a, F.div(asTensor(3.14159, 2))).sin(),
    }.mul(grad).setLabel("grad" ++ (if (a.meta.label) |label| ("_" ++ label) else ""));

    const local_updated_trace = if (trace) |t| t.appendLeft(local_grad.toAnyTensor()) else Trace.init(local_grad.toAnyTensor());
    return applyGradFn(a, local_grad, local_updated_trace, params);
}

pub fn binaryGradFn(comptime op: ops.BinaryOp, a: anytype, b: anytype, grad: anytype, trace: ?Trace, params: []const *const AnyTensor) std.meta.Tuple(&.{ Trace, []const *const AnyTensor }) {
    const local_grad_a, const local_grad_b = switch (op) {
        .add => .{ grad, grad },
        .mul => .{
            F.mul(b, grad).setLabel("grad" ++ (if (a.meta.label) |label| ("_" ++ label) else "")),
            F.mul(a, grad).setLabel("grad" ++ (if (b.meta.label) |label| ("_" ++ label) else "")),
        },
        else => unreachable,
    };

    const a_local_updated_trace = if (trace) |t| t.appendLeft(local_grad_a.toAnyTensor()) else Trace.init(local_grad_a.toAnyTensor());
    _, const a_updated_params = applyGradFn(a, local_grad_a, a_local_updated_trace, params);
    const b_local_updated_trace = if (trace) |t| t.appendLeft(local_grad_b.toAnyTensor()) else Trace.init(local_grad_b.toAnyTensor());
    return applyGradFn(b, local_grad_b, b_local_updated_trace, a_updated_params);
}

test "example" {
    const updated_params = comptime blk: {
        const x1 = tensor.Tensor([1]f32).param("x1");
        const x2 = tensor.Tensor([1]f32).param("x2");
        const a = F.mul(x1, x2).setLabel("a");
        const y1 = a.log2().setLabel("y1");
        const y2 = x2.exp2().setLabel("y2");
        const w = F.mul(y1, y2).setLabel("w");
        const backprop_results = backprop(w);
        break :blk backprop_results[0..backprop_results.len].*;
    };

    const graph = @import("graph.zig");
    var arena1 = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    var g = try graph.Graph.init(&arena1);
    defer g.deinit();

    std.debug.print("\n", .{});
    inline for (updated_params) |update| {
        try g.trace(update, true);
    }

    const debug = @import("debug.zig");
    try debug.dataflowViz(updated_params, debug.debug_writer, arena1.allocator());
}

test binaryGradFn {
    // const xt, const dx = comptime blk: {
    //     const a = tensor.Tensor(f32).input("a");
    //     const v0 = tensor.Tensor(f32).input("v0");
    //     const x0 = tensor.Tensor(f32).input("x0");
    //     const t = tensor.Tensor(f32).param("t");
    //     const x = F.mul(F.mul(0.5, a), F.mul(t, t)).add(F.mul(v0, t)).add(x0).setLabel("x(t)");
    //     break :blk .{ x, applyGradFn(x, x) };
    // };

    // _ = xt;
    // _ = dx;

    // const graph = @import("graph.zig");
    // var arena1 = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    // var g = try graph.Graph.init(&arena1);
    // defer g.deinit();

    // std.debug.print("\n", .{});
    // try g.trace(xt, true);

    // const debug = @import("debug.zig");

    // try debug.dataflowViz(.{&xt}, debug.debug_writer, arena1.allocator());

    // var arena2 = std.heap.ArenaAllocator.init(std.heap.page_allocator);

    // var dg = try graph.Graph.init(&arena2);
    // defer dg.deinit();

    // inline for (dx) |grad| {
    //     try dg.trace(grad, true);
    // }

    // try debug.dataflowViz(.{ dx[0].toAnyTensor(), dx[1].toAnyTensor() }, debug.debug_writer, arena2.allocator());
}
