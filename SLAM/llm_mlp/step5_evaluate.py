"""
Step 5: Evaluate predictions using the official eval.py script.

This script wraps the existing eval.py and displays results.
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate predictions using official eval.py'
    )
    parser.add_argument(
        '--pred',
        type=str,
        required=True,
        help='Path to prediction file'
    )
    parser.add_argument(
        '--key',
        type=str,
        required=True,
        help='Path to key file'
    )
    
    args = parser.parse_args()
    
    pred_path = os.path.join(os.path.dirname(__file__), '..', args.pred)
    key_path = os.path.join(os.path.dirname(__file__), '..', args.key)
    
    print("="*80)
    print("STEP 5: EVALUATION")
    print("="*80)
    print(f"Predictions: {pred_path}")
    print(f"Key file: {key_path}")
    print()
    
    # Run eval.py
    eval_script = os.path.join(os.path.dirname(__file__), '..', 'starter_code', 'eval.py')
    
    cmd = [sys.executable, eval_script, '--pred', pred_path, '--key', key_path]
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    print("\n" + "="*80)
    
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
