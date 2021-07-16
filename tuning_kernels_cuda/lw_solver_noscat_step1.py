import argparse
import kernel_tuner as kt
import common


# Parse command line arguments
def parse_command_line():
    parser = argparse.ArgumentParser(description='Tuning script for lw_solver_noscat_step1_kernel()')
    parser.add_argument('--tune', default=False, action='store_true')
    parser.add_argument('--run', default=False, action='store_true')
    parser.add_argument('--best_configuration', default=False, action='store_true')
    parser.add_argument('--block_size_x', type=int, default=32)
    parser.add_argument('--block_size_y', type=int, default=1)
    parser.add_argument('--block_size_z', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    command_line = parse_command_line()

    if command_line.tune:
        pass
    elif command_line.run:
        pass
