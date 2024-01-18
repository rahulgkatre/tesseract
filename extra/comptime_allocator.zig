const std = @import("std");

const GlobalArena = struct {
    var global_arena: std.heap.ArenaAllocator = undefined;

    fn init(arena: std.heap.ArenaAllocator) void {
        global_arena = arena;
    }

    fn deinit() void {
        global_arena.deinit();
        global_arena = undefined;
    }

    fn allocator() std.mem.Allocator {
        return global_arena.allocator();
    }
};

fn MyTypeGen(comptime size: usize) type {
    return struct {
        const Self = @This();
        var allocator = GlobalArena.allocator();

        data: ?[size]u8,
        closure: ?*const fn () Self,

        pub fn init() Self {
            if (@inComptime()) {
                return .{
                    .data = null,
                    .closure = struct {
                        pub fn f() Self {
                            const temp = allocator.alloc(u8, size) catch unreachable;
                            return .{ .data = temp[0..size].*, .closure = null };
                        }
                    }.f,
                };
            } else {
                const temp = allocator.alloc(u8, 5) catch unreachable;
                return .{ .data = temp, .closure = null };
            }
        }

        pub fn finish_init(self: *const Self) Self {
            return self.closure.?();
        }
    };
}

test "allocator" {
    const myData = comptime MyTypeGen(5).init();

    GlobalArena.init(std.heap.ArenaAllocator.init(std.heap.page_allocator));
    defer GlobalArena.deinit();
    std.debug.print("{}\n", .{myData});

    // const val = try MyType.init(&[_]u8{ 1, 2, 3 });
    const myNewData = myData.finish_init();

    std.debug.print("{}\n", .{myNewData});
}
