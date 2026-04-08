#!/usr/bin/env python
"""
Run MyTracker (ECO Tracker) on Jetson Nano or local machine

Usage:
    python run_mytracker.py
    python run_mytracker.py --dataset lasot --threads 2
    python run_mytracker.py --dataset otb --debug 1
"""

import sys
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Run MyTracker (ECO) on various datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mytracker.py                                    # Run on LaSOT (head 20 + tail 20)
  python run_mytracker.py --dataset lasot                   # Run on full LaSOT
  python run_mytracker.py --dataset otb                     # Run on OTB
  python run_mytracker.py --dataset lasot --debug 1         # With visualization
  python run_mytracker.py --threads 4                       # Use 4 threads
        """
    )
    
    parser.add_argument('--dataset', type=str, 
                       choices=['lasot_headtail40', 'lasot', 'lasot_first20', 'otb'],
                       default='lasot_headtail40',
                       help='Dataset to run on (default: lasot_headtail40 = head 20 + tail 20 sequences)')
    parser.add_argument('--debug', type=int, default=0, 
                       choices=[0, 1, 2],
                       help='Debug level (0=off, 1=basic, 2=verbose)')
    parser.add_argument('--threads', type=int, default=0,
                       help='Number of threads (default: 0 = auto)')
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.resolve()
    pytracking_dir = project_root / 'MyECOTracker' / 'pytracking'
    
    # Add to path
    sys.path.insert(0, str(pytracking_dir))
    
    # Import after path setup
    from pytracking.evaluation import Tracker, get_dataset, trackerlist
    from pytracking.experiments.myexperiments import (
        eco_verified_otb936_lasot_headtail40,
        eco_verified_otb936_lasot,
        eco_verified_otb936_lasot_first20,
        eco_verified_otb936_otb
    )
    from pytracking.evaluation.running import run_dataset
    
    # Map dataset to experiment function
    dataset_map = {
        'lasot_headtail40': eco_verified_otb936_lasot_headtail40,
        'lasot': eco_verified_otb936_lasot,
        'lasot_first20': eco_verified_otb936_lasot_first20,
        'otb': eco_verified_otb936_otb,
    }
    
    # Get experiment
    expr_func = dataset_map[args.dataset]
    trackers, dataset = expr_func()
    
    print("\n" + "="*50)
    print("Running MyTracker (ECO Tracker)")
    print("="*50)
    print(f"Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Tracker: ECO (MyTrackerECO, verified_otb936)")
    print(f"  Sequences: {len(dataset)}")
    print(f"  Debug Level: {args.debug}")
    print(f"  Threads: {'auto' if args.threads == 0 else args.threads}")
    print("="*50 + "\n")
    
    # Run tracker
    try:
        run_dataset(dataset, trackers, args.debug, args.threads)
        print("\n" + "="*50)
        print("MyTracker execution completed successfully!")
        print("="*50)
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
