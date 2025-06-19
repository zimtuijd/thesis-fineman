project (
        'main',
        'cpp', 'cuda',
        default_options: [
                'b_ndebug=if-release',
                'cpp_std=c++20',
                'werror=true',
        ],
)

add_project_arguments(
        '-arch=sm_86',
        '--std=c++20',
        '--expt-relaxed-constexpr',
        '--extended-lambda',
        '-lcuda',
        '-lcudart',
        language: 'cuda',
)

main = executable('main', 'main.cu', 'digraph.cc', link_args : '-lcuda')