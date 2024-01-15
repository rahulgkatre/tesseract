// NOTE: This build.zig file is only for the demo
// a build file for the library will be added later
const std = @import("std");
pub fn build(b: *std.Build) void {
    const exe = b.addExecutable(.{
        .name = "demo",
        .root_source_file = .{ .path = "demo.zig" },
        .target = b.host,
        .optimize = .ReleaseFast,
    });

    b.installArtifact(exe);
}
