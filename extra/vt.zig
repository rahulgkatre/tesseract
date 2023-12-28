const Interface = struct {
    // field: anyopaque,
    evalFn: *const fn (ptr: *Interface) Interface,
    pub fn eval(self: *Interface) Interface {
        return self.evalFn(self);
    }
};

fn MyType1(comptime T: type) type {
    return struct {
        field: T,
        interface: Interface,
        pub fn init(val: T) MyType1(T) {
            const impl = struct {
                pub fn eval(ptr: *Interface) Interface {
                    const self = @fieldParentPtr(MyType1(T), "interface", ptr);
                    return self.eval();
                }
            };
            return .{
                .field = val,
                .interface = .{ .evalFn = impl.eval },
            };
        }
        fn eval(self: *MyType1(T)) Interface {
            std.debug.print("MyType1\n", .{});
            return self.interface;
        }
    };
}

fn MyType2(comptime T: type) type {
    return struct {
        field: T,
        interface: Interface,
        pub fn init(val: T) MyType2(T) {
            const impl = struct {
                pub fn eval(ptr: *Interface) Interface {
                    const self = @fieldParentPtr(MyType2(T), "interface", ptr);
                    return self.eval();
                }
            };
            return .{
                .field = val,
                .interface = .{ .evalFn = impl.eval },
            };
        }
        fn eval(self: *MyType2(T)) Interface {
            std.debug.print("MyType2\n", .{});
            return MyType1(T).init(self.field).interface;
        }
    };
}
const std = @import("std");
test "vtab3_embedded_in_struct" {
    var o1 = MyType1(f32).init(3.14);
    var o2 = MyType2(i32).init(10);
    var mystuff = [_]*Interface{
        &o1.interface,
        &o2.interface,
    };
    for (mystuff) |o| {
        std.debug.print("{any}\n", .{o});
        _ = o.eval();
    }
}
