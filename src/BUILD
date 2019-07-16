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

cc_library(
    name = "wolff_algorithm",
    hdrs = [
        "wolff_algorithm.h",
        "tensor.h"],
)

cc_library(
    name = "udh_metropolis_algorithm",
    hdrs = [
        "udh_metropolis_algorithm.h",
        "tensor.h",
        "udh_spin.h"],
)

cc_library(
    name = "udh_measure",
    hdrs = [
        "udh_measure.h",
        "tensor.h",
        "udh_spin.h"],
)

cc_library(
    name = "compute_distribution_p_value",
    hdrs = [
        "compute_distribution_p_value.h",
        "distributions.h"],
)

cc_library(
    name = "mock_clock",
    srcs= ["mock_clock.cpp"],
    hdrs = ["mock_clock.h"],
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
    name = "tensor_test",
    srcs = [
        "tensor_test.cpp",
        "tensor.h"],
    deps = [
        "@gtest//:gtest",
        "@gtest//:gtest_main" 
    ],
)

cc_test (
    name = "observables_test",
    srcs = [
        "observables_test.cpp"],
    deps = [
        ":observables",
        "@gtest//:gtest",
        "@gtest//:gtest_main" 
    ],
)

cc_test (
    name = "next_configuration_test",
    srcs = [
        "next_configuration_test.cpp",
        "next_configuration.h"],
    deps = [
        "@gtest//:gtest",
        "@gtest//:gtest_main" 
    ],
)

cc_test (
    name = "wolff_algorithm_test",
    srcs = [
        "wolff_algorithm_test.cpp",
        "ud_spin.h",
        "next_configuration.h"],
    deps = [
        ":compute_distribution_p_value",
        ":wolff_algorithm",
        "@gtest//:gtest",
        "@gtest//:gtest_main" 
    ],
)

cc_test (
    name = "udh_spin_test",
    srcs = [
        "udh_spin_test.cpp",
        "udh_spin.h"],
    deps = [
        "@gtest//:gtest",
        "@gtest//:gtest_main" 
    ],
)

cc_test (
    name = "udh_metropolis_algorithm_test",
    srcs = [
        "udh_metropolis_algorithm_test.cpp",
        "next_configuration.h"],
    deps = [
        ":compute_distribution_p_value",
        ":udh_metropolis_algorithm",
        ":wolff_algorithm",
        "@gtest//:gtest",
        "@gtest//:gtest_main"
    ],
)

cc_test (
    name = "udh_io_test",
    srcs = [
        "udh_io_test.cpp",
        "udh_io.h"],
    deps = [
        ":udh_cc_proto",
        "@gtest//:gtest",
        "@gtest//:gtest_main" 
    ],
)

cc_test (
    name = "udh_measure_test",
    srcs = [
        "udh_measure_test.cpp"],
    deps = [
        ":udh_cc_proto",
        ":udh_measure",
        ":udh_metropolis_algorithm",
        "@gtest//:gtest",
        "@gtest//:gtest_main" 
    ],
)

cc_test (
    name = "timer_test",
    srcs = [
        "timer_test.cpp",
        "timer.h"],
    deps = [
        ":mock_clock",
        "@gtest//:gtest",
        "@gtest//:gtest_main" 
    ],
)

cc_test (
    name = "progress_test",
    srcs = [
        "progress_test.cpp",
        "progress.h"],
    deps = [
        "@gtest//:gtest",
        "@gtest//:gtest_main" 
    ],
)

cc_test (
    name = "throttle_test",
    srcs = [
        "throttle_test.cpp",
        "throttle.h"],
    deps = [
        ":mock_clock",
        "@gtest//:gtest",
        "@gtest//:gtest_main" 
    ],
)

cc_test (
    name = "udh_arguments_test",
    srcs = [
        "udh_arguments_test.cpp",
        "udh_arguments.h"],
    deps = [
        ":udh_cc_proto",
        "@gtest//:gtest",
        "@gtest//:gtest_main" 
    ],
)
