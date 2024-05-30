const nn = @import("nn.zig");
const std = @import("std");
const dtypes = @import("dtypes.zig");
const tensor = @import("tensor.zig");

// pub fn LayerNorm(comptime start_dim: u8) type {
//     return struct {
//         const Self = @This();
//         pub usingnamespace Module.IFace(Self, struct {
//         pub fn forward(comptime _: Self, comptime x: anytype)  {
//             const mean = x.mean(.{1, 2, 3});
//             const variance = x.variance(.{})
//         }
//     });

//     }
// }

pub fn MultiHeadAttention(dtype: dtypes.DType, d_model: comptime_int, num_heads: comptime_int) type {
    std.debug.assert(@mod(d_model, num_heads) == 0);
    return struct {
        const Self = @This();
        const d_k = @divExact(d_model, num_heads);
        
        w_q: nn.Linear(d_model, d_model, dtype, "w_q"),
        w_k: nn.Linear(d_model, d_model, dtype, "w_k"),
        w_v: nn.Linear(d_model, d_model, dtype, "w_v"),
        w_o: nn.Linear(d_model, d_model, dtype, "w_o"),

        pub fn scaledDotProductAttention(self: Self, q: anytype, k: anytype, v: anytype, mask: anytype) void {
            const tq = tensor.asTensor(q);
            const tk = tensor.asTensor(q);
            const tv = tensor.asTensor(q);

            const attn_scores = tq.matmul(tk.T()).div(@sqrt(dk));
            if (mask != null) {
                @compileError("Not implemented yet");
                
            }

            const attn_probs = attn_scores.softmax(-1);
            return attn_probs.matmul(tv);
        }
        
        pub usingnamespace nn.Module.IFace(Self, struct {


            pub fn forward()




        })
    };



}
