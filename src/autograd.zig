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

pub fn noBackward(ctx: BackwardContext, _: anytype) BackwardContext {
    return ctx;
}

pub const BackwardFn = *const fn (anytype, BackwardContext) BackwardContext;

pub const BackwardContext = struct {
    grads: []const *const AnyTensor,
    label2idx: *const std.StaticStringMap(usize),

    pub fn init(params: []const *const AnyTensor) BackwardContext {
        var zero_grads: [params.len]*const AnyTensor = undefined;
        var grad_labels_index: [params.len]std.meta.Tuple(&.{ []const u8, usize }) = undefined;
        for (params, 0..) |p, i| {
            const grad_label = "grad_" ++ p.getLabel().?;
            zero_grads[i] = asTensor(0.0).setLabel(grad_label).toAnyTensor();
            grad_labels_index[i] = .{ grad_label, i };
        }

        return .{
            .grads = &zero_grads,
            .label2idx = &std.StaticStringMap(usize).initComptime(grad_labels_index),
        };
    }

    pub fn backwardStep(ctx: BackwardContext, x: anytype, grad: anytype) BackwardContext {
        if (x.meta.constant) {
            return ctx;
        }
        const gradFn: BackwardFn = @ptrCast(x.meta.reverse_ad_fn);
        return @call(.auto, gradFn, .{ ctx, grad });
    }

    pub fn accumulateGrad(ctx: BackwardContext, incoming_grad: anytype, label: []const u8) BackwardContext {
        var new_grads = ctx.grads[0..ctx.grads.len].*;
        const idx: usize = ctx.label2idx.get("grad_" ++ label).?;
        new_grads[idx] = F.add(ctx.grads[idx], incoming_grad).setLabel(ctx.grads[idx].getLabel().?).toAnyTensor();
        return .{
            .grads = &new_grads,
            .label2idx = ctx.label2idx,
        };
    }
};

pub fn backwards(x: anytype) []const *const AnyTensor {
    const initial_grad = tensor.asTensor(1.0);
    const params = utils.paramsOf(x);
    const ctx = BackwardContext.init(params);
    return ctx.backwardStep(x, initial_grad).grads;
}

pub fn unaryBackward(ctx: BackwardContext, incoming_grad: anytype, op: ops.UnaryOp, a: anytype) BackwardContext {
    const grad_a = blk: {
        const local_grad = switch (op) {
            .exp2 => F.exp2(a).mul(F.LN_2),
            .log2 => F.div(F.INV_LN_2, a),
            .neg => F.neg(a),
            .recip => F.mul(a, a).neg(),
            .sqrt => F.div(0.5, F.sqrt(a)),
            .sin => F.add(a, F.div(asTensor(3.14159, 2))).sin(),
        };
        break :blk local_grad.mul(incoming_grad).setLabel("grad" ++ (if (a.meta.label) |label| ("_" ++ label) else ""));
    };
    return ctx.backwardStep(a, grad_a);
}

pub fn binaryBackward(ctx: BackwardContext, incoming_grad: anytype, op: ops.BinaryOp, a: anytype, b: anytype) BackwardContext {
    const grad_a, const grad_b = switch (op) {
        .add => .{ incoming_grad, incoming_grad },
        .mul => .{
            F.mul(b, incoming_grad).setLabel("grad" ++ (if (a.meta.label) |label| ("_" ++ label) else "")),
            F.mul(a, incoming_grad).setLabel("grad" ++ (if (b.meta.label) |label| ("_" ++ label) else "")),
        },
        else => unreachable,
    };

    const ctx_a = ctx.backwardStep(a, grad_a);
    const ctx_b = ctx_a.backwardStep(b, grad_b);
    return ctx_b;
}

test "example" {
    const f, const df = comptime blk: {
        const x = tensor.Tensor(f32).param("x");
        const y = tensor.Tensor(f32).param("y");
        const s = F.add(x.mul(2.0), y.mul(3.0)).setLabel("s");
        const f = F.add(s, s).setLabel("f");
        const df = f.backwards();
        break :blk .{ f, df[0..df.len].* };
    };

    const graph = @import("graph.zig");
    var arena1 = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    var g = try graph.Graph.init(&arena1);
    defer g.deinit();

    std.debug.print("\n", .{});
    try g.trace(f, true);
    inline for (df) |update| {
        try g.trace(update, true);
    }

    const debug = @import("debug.zig");
    try debug.dataflowViz(.{f.toAnyTensor()} ++ df, debug.debug_writer, arena1.allocator());
}

test binaryBackward {
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
