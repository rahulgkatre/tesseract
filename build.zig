const std = @import("std");
pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{});
    const tesseract_file = b.option([]const u8, "tfile", "Tesseract file to compile");
    if (tesseract_file != null) {
        const exe = b.addExecutable(.{
            .name = "tesseract",
            .root_source_file = .{ .path = tesseract_file.? },
            .target = b.host,
            .optimize = optimize,
        });
        b.installArtifact(exe);

        const run_exe = b.addRunArtifact(exe);
        const run_step = b.step("run", "Run the executable");
        run_step.dependOn(&run_exe.step);
    }

    const tests = b.addTest(.{
        .root_source_file = .{ .path = "./tesseract.zig" },
        .target = b.host,
        .optimize = optimize,
    });

    // Does not work: https://github.com/ziglang/zig/issues/17756
    // Added a vscode task to run coverage for now
    // const coverage = b.option(bool, "test-coverage", "Generate test coverage") orelse false;
    // if (coverage) {
    //     // with kcov
    //     tests.setExecCmd(&.{
    //         "kcov",
    //         // "--include-path=.", // any kcov flags can be specified here
    //         // "--clean",
    //         "~/coverage", // output dir for kcov
    //         null, // to get zig to use the --test-cmd-bin flag
    //     });
    // }

    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_tests.step);
}
