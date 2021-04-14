import argparse


def get_parser(parser):
    parser.add_argument('expert_path', type=str, default="./expert_demonstrations/Hopper-v2")
    args = parser.parse_args()
    return args
