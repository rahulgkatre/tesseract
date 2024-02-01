// fn fn1() Tensor(i32, .{ 2, 1, 4 }) {
//     const tensor1 = Tensor(i32, .{ 2, 1, 4 }).full(TestBackend, 1);
//     const tensor2 = Tensor(i32, .{ 2, 3, 1 }).full(TestBackend, 2);
//     const tensor3 = tensor1.add(tensor2).sum(1);
//     return tensor3;
// }

// fn fn2(input: anytype) Tensor(i32, .{ 2, 1, 4 }) {
//     return comptime blk: {
//         const tensor4 = Tensor(i32, .{ 2, 1, 4 }).full(TestBackend, 4);
//         const tensor5 = Tensor(i32, .{ 2, 3, 1 }).full(TestBackend, 5);
//         const tensor6 = tensor4.mul(tensor5).sum(1).add(input);
//         break :blk tensor6;
//     };
// }

// test "tensors from functions" {
//     const out = comptime blk: {
//         const tensor3 = fn1();
//         const tensor6 = fn2(tensor3);
//         break :blk tensor6;
//     };

//     runEval("tensors from functions", out);
// }
