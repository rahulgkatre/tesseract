const Interface = struct {
    // field: anyopaque,
    const Self = @This();
    evalFn: *const fn (ptr: *Self) Self,
    myfuncFn: *const fn (ptr: *Self, other: i32) Self,
    pub fn eval(self: *Self) Self {
        return self.evalFn(self);
    }
    pub fn myfunc(self: *Self, other: i32) Self {
        return self.myfuncFn(self, other);
    }
};

fn MyType1(comptime T: type) type {
    return struct {
        const Self = @This();
        field: T,
        interface: Interface,
        pub fn init(val: T) Self {
            const impl = struct {
                pub fn eval(ptr: *Interface) Interface {
                    const self = @fieldParentPtr(Self, "interface", ptr);
                    return self.eval();
                }
                pub fn myfunc(ptr: *Interface, other: i32) Interface {
                    _ = other;
                    const self = @fieldParentPtr(Self, "interface", ptr);
                    return self.eval();
                }
            };
            return .{
                .field = val,
                .interface = .{ .evalFn = impl.eval, .myfuncFn = impl.myfunc },
            };
        }
        fn eval(self: *Self) Interface {
            return self.interface;
        }
        fn myfunc(self: *Self, other: i32) Interface {
            _ = other;
            return self.eval();
        }
    };
}

fn MyType2(comptime T: type) type {
    return struct {
        const Self = @This();
        field: T,
        interface: Interface,
        pub fn init(val: T) Self {
            const impl = struct {
                pub fn eval(ptr: *Interface) Interface {
                    const self = @fieldParentPtr(Self, "interface", ptr);
                    return self.eval();
                }
                pub fn myfunc(ptr: *Interface, other: i32) Interface {
                    _ = other;
                    const self = @fieldParentPtr(Self, "interface", ptr);
                    return self.eval();
                }
            };
            return .{
                .field = val,
                .interface = .{ .evalFn = impl.eval, .myfuncFn = impl.myfunc },
            };
        }
        fn eval(self: *Self) Interface {
            return MyType1(T).init(self.field).interface;
        }
        fn myfunc(self: *Self, other: i32) Interface {
            _ = other;
            return self.eval();
        }
    };
}
const std = @import("std");
test "vtab3_embedded_in_struct" {
    var o1 = MyType1(f32).init(3.14);
    var o2 = MyType2(i32).init(10);
    const mystuff = [_]*Interface{
        &o1.interface,
        &o2.interface,
    };
    for (mystuff) |o| {
        _ = o;

        // std.debug.print("{any}\n", .{o});
        // std.debug.print("eval() = {any}\n", .{o.eval()});
    }
}
