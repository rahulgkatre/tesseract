const Interface = struct {
    evalFn: *const fn (ptr: *Interface) Interface,
    pub fn eval(self: *Interface) Interface {
        return self.evalFn(self);
    }
};

const MyType1 = struct {
    interface: Interface,
    pub fn init() MyType1 {
        const impl = struct {
            pub fn eval(ptr: *Interface) Interface {
                const self = @fieldParentPtr(MyType1, "interface", ptr);
                return self.eval();
            }
        };
        return .{
            .interface = .{ .evalFn = impl.eval },
        };
    }
    fn eval(self: *MyType1) Interface {
        std.debug.print("MyType1\n", .{});
        return self.interface;
    }
};

const MyType2 = struct {
    interface: Interface,
    pub fn init() MyType2 {
        const impl = struct {
            pub fn eval(ptr: *Interface) Interface {
                const self = @fieldParentPtr(MyType2, "interface", ptr);
                return self.eval();
            }
        };
        return .{
            .interface = .{ .evalFn = impl.eval },
        };
    }
    fn eval(_: *MyType2) Interface {
        std.debug.print("MyType2\n", .{});
        return MyType1.init().interface;
    }
};
const std = @import("std");
test "vtab3_embedded_in_struct" {
    var o1 = MyType1.init();
    var o2 = MyType2.init();
    var mystuff = [_]*Interface{
        &o1.interface,
        &o2.interface,
    };
    for (mystuff) |o| {
        std.debug.print("{any}\n", .{o});
        _ = o.eval();
    }
}
