"""Run the preprocessing and indexing pipeline in the required order.

This script executes each stage of the data preparation workflow sequentially so
that metadata, cleaned descriptions, and final Groq-enriched summaries are built
in a predictable order.
"""

import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

scripts = [
    "ingestion.py", 
    "metadata_chunking.py", 
    "description_cleaner.py", 
    "description_chunker.py"
]

for script in scripts:
    script_path = os.path.join(BASE_DIR, script)

    if not os.path.exists(script_path):
        print(f"Error: Cannot find '{script}' at path: {script_path}")
        sys.exit(1)

    print(f"\n--- Running {script} ---")

    result = subprocess.run([sys.executable, script_path], cwd=BASE_DIR)

    if result.returncode != 0:
        print(f"\nPipeline halted: '{script}' failed with exit code {result.returncode}.")
        sys.exit(result.returncode)

print("\nAll pipeline scripts completed successfully!")