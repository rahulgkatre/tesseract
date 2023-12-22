const ops = @import("ops.zig");

const Variable = struct {
    internal: *anyopaque,
    last_fn: ops.Function,
};
