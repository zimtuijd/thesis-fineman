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
        language: 'cuda'
)

main = executable('main', 'main.cu', 'bfs.cu', 'bfs_kernels.cu', 'digraph.cc')