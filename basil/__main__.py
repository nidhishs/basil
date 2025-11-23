#!/usr/bin/env python3
"""Command-line interface for Basil.

This module serves as the main entry point for the basil CLI.
It registers subcommands and routes to their respective implementations.
"""
import argparse
import sys

from basil.commands import decode, encode, train


def main():
    """Main CLI entry point."""
    # fmt: off
    parser = argparse.ArgumentParser(prog="basil", description="Balanced Assignment Semantic ID Library (Basil) - CLI tools")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    # fmt: on

    # Register commands
    train.add_parser(subparsers)
    encode.add_parser(subparsers)
    decode.add_parser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # If no command specified, print help
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    # Execute the command
    args.func(args)


if __name__ == "__main__":
    main()
