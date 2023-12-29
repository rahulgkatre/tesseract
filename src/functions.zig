// Something like this I think
pub fn TensorFn(comptime Input: type, comptime Output: type) type {
    return struct {
        const Self = @This();
        forward_fn: *const fn (ptr: *Self, args: Input) Output,
    };
}