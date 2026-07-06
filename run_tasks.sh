#!/bin/bash

TASKS_DIR="./tasks"

find "$TASKS_DIR" -type f -name "*.py" | sort | while read -r file; do
    echo "Running $file"
    uv run ipython "$file"
done