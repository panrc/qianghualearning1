#!/usr/bin/env python3
"""
Data Preprocessing Script for Crypto Trading RL Bot

This script precomputes technical indicators for trading data files,
saving them as parquet files for faster loading during training.

Usage:
    python preprocess_data.py --input data/BTCUSDT_5m_2years.csv
    python preprocess_data.py --input-dir data/
    python preprocess_data.py --force  # Force recompute all cached files
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.trading_env import TradingEnv
from src import config

def preprocess_single_file(input_file, force=False):
    """Preprocess a single CSV file"""
    print(f"\n=== Processing {input_file} ===")
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist")
        return False
    
    try:
        # Create environment with caching enabled
        env = TradingEnv(file_path=input_file, use_cache=True)
        
        # Generate cache path
        cache_path = env._generate_cache_path(input_file)
        
        # Check if cache exists and is newer
        if not force and os.path.exists(cache_path):
            cache_mtime = os.path.getmtime(cache_path)
            input_mtime = os.path.getmtime(input_file)
            
            if cache_mtime > input_mtime:
                print(f"✓ Cache is up to date: {cache_path}")
                return True
            else:
                print(f"Cache is outdated, recomputing...")
        
        # Load data (this will trigger preprocessing and caching)
        start_time = time.time()
        success = env.load_data(input_file)
        end_time = time.time()
        
        if success:
            print(f"✓ Successfully processed in {end_time - start_time:.2f} seconds")
            print(f"✓ Cache saved to: {cache_path}")
            print(f"✓ Data shape: {env.data_df.shape}")
            return True
        else:
            print(f"✗ Failed to process {input_file}")
            return False
            
    except Exception as e:
        print(f"✗ Error processing {input_file}: {e}")
        return False

def preprocess_directory(input_dir, force=False):
    """Preprocess all CSV files in a directory"""
    print(f"\n=== Processing directory {input_dir} ===")
    
    if not os.path.isdir(input_dir):
        print(f"Error: Directory {input_dir} does not exist")
        return False
    
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return False
    
    print(f"Found {len(csv_files)} CSV files")
    
    success_count = 0
    total_files = len(csv_files)
    
    for csv_file in csv_files:
        file_path = os.path.join(input_dir, csv_file)
        if preprocess_single_file(file_path, force):
            success_count += 1
    
    print(f"\n=== Summary ===")
    print(f"Successfully processed: {success_count}/{total_files} files")
    
    return success_count == total_files

def list_cache_status(input_dir):
    """List the cache status of all CSV files in a directory"""
    print(f"\n=== Cache Status for {input_dir} ===")
    
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    env = TradingEnv(use_cache=False)  # Just for cache path generation
    
    for csv_file in sorted(csv_files):
        file_path = os.path.join(input_dir, csv_file)
        cache_path = env._generate_cache_path(file_path)
        
        if os.path.exists(cache_path):
            cache_mtime = os.path.getmtime(cache_path)
            input_mtime = os.path.getmtime(file_path)
            
            if cache_mtime > input_mtime:
                status = "✓ Up to date"
            else:
                status = "⚠ Outdated"
        else:
            status = "✗ No cache"
        
        print(f"{csv_file:<40} {status}")

def clean_cache(input_dir):
    """Clean all cache files for a directory"""
    print(f"\n=== Cleaning cache for {input_dir} ===")
    
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    env = TradingEnv(use_cache=False)
    
    removed_count = 0
    for csv_file in csv_files:
        file_path = os.path.join(input_dir, csv_file)
        cache_path = env._generate_cache_path(file_path)
        
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                print(f"✓ Removed cache: {cache_path}")
                removed_count += 1
            except Exception as e:
                print(f"✗ Failed to remove {cache_path}: {e}")
    
    print(f"Removed {removed_count} cache files")

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess crypto trading data for faster training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single file
    python preprocess_data.py --input data/BTCUSDT_5m_2years.csv
    
    # Process all files in directory
    python preprocess_data.py --input-dir data/
    
    # Force recompute all files
    python preprocess_data.py --input-dir data/ --force
    
    # Check cache status
    python preprocess_data.py --input-dir data/ --status
    
    # Clean cache
    python preprocess_data.py --input-dir data/ --clean
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', '-i', help='Input CSV file to process')
    group.add_argument('--input-dir', '-d', help='Input directory containing CSV files')
    
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force recompute even if cache exists')
    parser.add_argument('--status', '-s', action='store_true',
                        help='Show cache status without processing')
    parser.add_argument('--clean', '-c', action='store_true',
                        help='Clean cache files')
    
    args = parser.parse_args()
    
    print("Crypto Trading Data Preprocessor")
    print("=" * 40)
    print(f"Config: {len(config.FEATURE_COLUMNS)} features")
    print(f"Cache directory: {os.path.dirname(config.FEATURE_CACHE_PATH)}")
    
    if args.input:
        # Process single file
        success = preprocess_single_file(args.input, args.force)
        sys.exit(0 if success else 1)
    
    elif args.input_dir:
        if args.status:
            # Show status only
            list_cache_status(args.input_dir)
        elif args.clean:
            # Clean cache
            clean_cache(args.input_dir)
        else:
            # Process directory
            success = preprocess_directory(args.input_dir, args.force)
            sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 