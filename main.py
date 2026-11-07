"""Config-driven entry point, kept for compatibility: python main.py configs/config_soft.json [metric] [split]"""
import sys

from parser.parser import parse_config
from run_eval import run


def main(config_path, metric="both", split="full"):
    run(parse_config(config_path), metric, split)


if __name__ == "__main__":
    main(*sys.argv[1:])
