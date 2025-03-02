#!/usr/bin/env python3
"""
A simple hello world script for the Words To Font project.
"""
import sys

import torch


def main():
    print("💬 Hello from Words To Font!")
    print(f"🐍 Python version: {sys.version}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    print("Ready to convert words to beautiful fonts!")

if __name__ == "__main__":
    main() 