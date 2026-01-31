"""
Data preprocessing utilities for fraud detection.
Includes data loading and memory optimization.
"""

import os
import pandas as pd
import numpy as np
import gc


def load_and_merge_data(data_dir='data/IEEE-CIS'):
    """
    Load transaction and identity data, then merge them.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        train, test: Merged DataFrames
    """
    print('Loading data...')
    train_transaction = pd.read_csv(os.path.join(data_dir, 'train_transaction.csv'))
    train_identity = pd.read_csv(os.path.join(data_dir, 'train_identity.csv'))
    test_transaction = pd.read_csv(os.path.join(data_dir, 'test_transaction.csv'))
    test_identity = pd.read_csv(os.path.join(data_dir, 'test_identity.csv'))
    
    # Merge
    train = train_transaction.merge(train_identity, on='TransactionID', how='left')
    test = test_transaction.merge(test_identity, on='TransactionID', how='left')
    
    del train_transaction, train_identity, test_transaction, test_identity
    gc.collect()
    
    print(f'Train shape: {train.shape}')
    print(f'Test shape: {test.shape}')
    
    return train, test


def reduce_mem_usage(df, verbose=True):
    """
    Reduce memory usage by downcasting numeric types.
    
    Args:
        df: DataFrame to optimize
        verbose: Print memory reduction info
        
    Returns:
        df: Optimized DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            
            if pd.api.types.is_integer_dtype(df[col]):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage: {start_mem:.2f} MB -> {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df
