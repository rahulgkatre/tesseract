const std = @import("std");
const tensor = @import("tensor/tensor.zig");
const F = @import("tensor/functions.zig");
const AnyTensor = @import("tensor/anytensor.zig").AnyTensor;

const dtypes = @import("dtypes.zig");
const ops = @import("ops.zig");
const utils = @import("utils.zig");

const tensor_typing = @import("tensor/tensor_typing.zig");
const asTensor = tensor_typing.asTensor;
const TensorTypeOf = tensor_typing.TensorTypeOf;
const TensorTuple = tensor_typing.TensorTuple;
const IntTensor = tensor_typing.IntTensor;
const BoolTensor = tensor_typing.BoolTensor;
const FloatTensor = tensor_typing.FloatTensor;

pub fn backwards(x: anytype) []const *const AnyTensor {
    const initial_grad = tensor_typing.asTensor(1.0);
    const params = utils.paramsOf(x);

    var zero_grads: [params.len]*const AnyTensor = undefined;
    for (params, 0..) |p, i| {
        zero_grads[i] = asTensor(0.0).setLabel("grad_" ++ p.meta.label.?).toAnyTensor();
    }

    return backpropStep(x, initial_grad, &zero_grads);
}

pub fn backpropStep(x: anytype, grad: anytype, param_grads: []const *const AnyTensor) []const *const AnyTensor {
    const gradFn: *const fn (anytype, []const *const AnyTensor) []const *const AnyTensor = @ptrCast(x.meta.grad_fn);
    return gradFn(grad, param_grads);
}

pub fn noGrad(_: anytype, param_grads: []const *const AnyTensor) []const *const AnyTensor {
    return param_grads;
}

pub fn accumulateGrad(label: []const u8, grad: anytype, param_grads: []const *const AnyTensor) []const *const AnyTensor {
    const param_i: usize = blk: {
        for (param_grads, 0..) |p, i| {
            if (p.meta.label) |param_label| {
                if (std.mem.eql(u8, param_label, "grad_" ++ label)) {
                    break :blk i;
                }
            } else {
                @compileLog(p.meta);
                unreachable;
            }
        }
        unreachable;
    };
    var updated_params: [param_grads.len]*const AnyTensor = param_grads[0..param_grads.len].*;
    updated_params[param_i] = F.add(param_grads[param_i].toTensor(), grad).setLabel(param_grads[param_i].meta.label.?).toAnyTensor();
    return &updated_params;
}

pub fn unaryGrad(op: ops.UnaryOp, a: anytype, grad: anytype, param_grads: []const *const AnyTensor) []const *const AnyTensor {
    const grad_a = blk: {
        const local_grad = switch (op) {
            .exp2 => F.exp2(a).mul(F.LN_2),
            .log2 => F.div(F.INV_LN_2, a),
            .neg => F.neg(a),
            .recip => F.mul(a, a).neg(),
            .sqrt => F.div(0.5, F.sqrt(a)),
            .sin => F.add(a, F.div(asTensor(3.14159, 2))).sin(),
        };
        break :blk local_grad.mul(grad).setLabel("grad" ++ (if (a.meta.label) |label| ("_" ++ label) else ""));
    };

    return backpropStep(a, grad_a, param_grads);
}

pub fn binaryGrad(op: ops.BinaryOp, a: anytype, b: anytype, grad: anytype, param_grads: []const *const AnyTensor) []const *const AnyTensor {
    const grad_a, const grad_b = switch (op) {
        .add => .{ grad, grad },
        .mul => .{
            F.mul(b, grad).setLabel("grad" ++ (if (a.meta.label) |label| ("_" ++ label) else "")),
            F.mul(a, grad).setLabel("grad" ++ (if (b.meta.label) |label| ("_" ++ label) else "")),
        },
        else => unreachable,
    };

    return backpropStep(b, grad_b, backpropStep(a, grad_a, param_grads));
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

test binaryGrad {
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
