#!/usr/bin/env python3
"""
Runner script for Words To Font hello world.
Run with: uv run run_hello.py
"""
import importlib.util
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def run_hello():
    # Import the hello_world module
    spec = importlib.util.spec_from_file_location("hello_world", "src/hello_world.py")
    hello_world = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hello_world)
    
    # Run the main function
    hello_world.main()

if __name__ == "__main__":
    run_hello()
    print("hi bensen")