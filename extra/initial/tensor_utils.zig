pub const MAX_NDIMS = 8;
pub const TensorError = error{
    InequalDimensions,
    BroadcastDimensions,
    IndexOutOfBounds,
};

pub fn broadcastIndex(big_shape: []usize, big_index: []usize, small_shape: []usize, out_index: *[]usize) !void {
    const out_ndims = @min(big_shape.len, small_shape.len);
    if (big_shape.len != big_index.len and out_ndims != out_index.len) {
        return TensorError.BroadcastDimensions;
    }

    const ndims1 = big_shape.len;
    const ndims2 = small_shape.len;
    for (0..out_ndims) |dim| {
        out_index[out_ndims - dim - 1] = if (small_shape[ndims2 - dim - 1] == 1) 0 else big_index[ndims1 - dim - 1];
    }
}

pub fn shape2strides(shape: []const usize, out_strides: []usize) !void {
    if (shape.len != out_strides.len) {
        return TensorError.InequalDimensions;
    }
    var stride: usize = undefined;
    var offset: usize = 1;
    for (0..shape.len - 1) |i| {
        stride = shape[shape.len - i - 1] * offset;
        out_strides[shape.len - i - 2] = stride;
        offset = stride;
    }
    out_strides[out_strides.len - 1] = 1;
}

pub fn shape2size(shape: []const usize) usize {
    var size: usize = 1;
    for (shape) |dimSize| {
        size *= dimSize;
    }
    return size;
}
