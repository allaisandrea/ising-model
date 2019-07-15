proto_library(
    name = "udh_proto",
    srcs = ["udh.proto"]
)

cc_proto_library(
    name = "udh_cc_proto",
    deps = [":udh_proto"]
)

cc_library(
    name = "observables",
    hdrs = [
        "observables.h",
        "tensor.h",
        "ud_spin.h"],
)

cc_binary(
    name = "run-simulation",
    srcs = [
        "udh_run_simulation.cpp",
        "progress.h",
        "udh_measure.h",
        "udh_spin.h",
        "udh_metropolis_algorithm.h",
        "wolff_algorithm.h",
        "udh_io.h",
        "tensor.h",
        "timer.h",
        "udh_arguments.h"],
    deps = [":udh_cc_proto"],
    linkopts = [
        "-lboost_program_options"
    ],
)

cc_binary(
    name = "compute-autocorrelation",
    srcs = [
        "compute_autocorrelation.cpp",
        "udh_file_group.h",
        "udh_parameters_set.h",
        "udh_io.h",
        "udh_arguments.h"],
    deps = [":udh_cc_proto"],
)

cc_binary(
    name = "print-parameters-table",
    srcs = [
        "print_parameters_table.cpp",
        "udh_parameters_set.h",
        "udh_io.h",
        "udh_arguments.h"],
    deps = [":udh_cc_proto"],
)

cc_test (
    name = "test",
    srcs = [
        "test.cpp",
        "progress.h",
        "udh_measure.h",
        "observables.h",
        "distributions.h",
        "next_configuration.h",
        "udh_spin.h",
        "ud_spin.h",
        "throttle.h",
        "udh_metropolis_algorithm.h",
        "wolff_algorithm.h",
        "udh_io.h",
        "tensor.h",
        "timer.h",
        "udh_arguments.h"],
    deps = [
        ":udh_cc_proto",
        "@gtest//:gtest",
        "@gtest//:gtest_main" # Only if hello_test.cc has no main()
    ],
    linkopts = [
        "-lboost_program_options"
    ],
)

cc_test (
    name = "tensor_test",
    srcs = [
        "tensor_test.cpp",
        "tensor.h"],
    deps = [
        "@gtest//:gtest",
        "@gtest//:gtest_main" # Only if hello_test.cc has no main()
    ],
)

cc_test (
    name = "observables_test",
    srcs = [
        "observables_test.cpp"],
    deps = [
        ":observables",
        "@gtest//:gtest",
        "@gtest//:gtest_main" # Only if hello_test.cc has no main()
    ],
)

cc_test (
    name = "next_configuration_test",
    srcs = [
        "next_configuration_test.cpp",
        "next_configuration.h"],
    deps = [
        "@gtest//:gtest",
        "@gtest//:gtest_main" # Only if hello_test.cc has no main()
    ],
)
