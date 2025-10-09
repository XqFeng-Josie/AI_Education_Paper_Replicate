#!/usr/bin/env python3
"""
Fix H5 file with proper preprocessing to match the expected data structure.
"""

import os
import pandas as pd
import sys
from pathlib import Path

# Add the selflearner module to the path
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
SELF_LEARNER_MODULE_PATH = os.path.abspath(os.path.join(BASE_PATH, 'selflearner'))
sys.path.append(SELF_LEARNER_MODULE_PATH)

from selflearner.data_load.hdf5.pytables_hdf5_manager import PytablesHdf5Manager

# Dataset constants
DS_ASSESSMENTS = 'assessments'
DS_COURSES = 'courses'
DS_STUD_ASSESSMENTS = 'studentAssessment'
DS_STUD_INFO = 'studentInfo'
DS_STUD_REG = 'studentRegistration'
DS_STUD_VLE = 'studentVle'
DS_VLE = 'vle'


def fix_assessments(df_ass):
    df_ass_sorted = df_ass.sort_values(['code_module', 'code_presentation', 'assessment_type', 'date'])
    df_ass_sorted['assessment_seq'] = (
        df_ass_sorted
        .groupby(['code_module', 'code_presentation', 'assessment_type'], sort=False)
        .cumcount()
        + 1
    )

    def _make_assessment_name(row):
        typ = row['assessment_type']
        seq = int(row['assessment_seq'])
        if typ == 'Exam':
            return 'Exam' if seq == 1 else f'Exam {seq}'
        return f'{typ} {seq}'

    if 'assessment_name' not in df_ass_sorted.columns:
        df_ass_sorted['assessment_name'] = df_ass_sorted.apply(_make_assessment_name, axis=1)

    # Keep only columns needed downstream and preserve original column names
    df_ass = df_ass_sorted.copy()
    print(df_ass.assessment_type.value_counts())
    return df_ass


def preprocess_dataframe(df, ds):
    """Apply the same preprocessing as in Hdf5Creator.preprocess"""
    if ds == DS_ASSESSMENTS:
        df = fix_assessments(df)
        df = df.set_index(['code_module', 'code_presentation'], drop=False)
        # df = df.sort_values(by='date')
        # Group by index levels and assessment_type column
        # df_agg = df.groupby([df.index.get_level_values(0), df.index.get_level_values(1), 'assessment_type'])
        # Assign each row the order within the group MOD-PRES-TYPE
        # df['assessment_name'] = df['assessment_type'] + " " + (df_agg.cumcount() + 1).map(str)
        # df.loc[df['assessment_name'] == 'Exam 1', 'assessment_name'] = 'Exam'
    elif ds == DS_COURSES:
        df = df.set_index(['code_module', 'code_presentation'])
    elif ds == DS_VLE:
        df = df.set_index(['code_module', 'code_presentation'])
    elif ds == DS_STUD_INFO:
        df = df.set_index(['code_module', 'code_presentation'])
    elif ds == DS_STUD_REG:
        df = df.set_index(['code_module', 'code_presentation'])
    elif ds == DS_STUD_ASSESSMENTS:
        df = df.set_index(['id_assessment'])
    elif ds == DS_STUD_VLE:
        df = df.set_index(['code_module', 'code_presentation'])
    else:
        df = df
    return df

def fix_h5_preprocessing():
    """Recreate H5 file with proper preprocessing"""
    
    # Define paths
    data_path = os.path.join(SELF_LEARNER_MODULE_PATH, 'data_load', 'data')
    hdf5_path = os.path.join(data_path, 'oulad.h5')
    
    print("Fixing H5 file with proper preprocessing...")
    print(f"Data path: {data_path}")
    print(f"Output H5 file: {hdf5_path}")
    
    # Remove existing H5 file
    if os.path.exists(hdf5_path):
        os.remove(hdf5_path)
        print("Removed existing H5 file")
    
    # Initialize HDF5 manager
    store_manager = PytablesHdf5Manager(hdf5_path)
    
    # CSV file mappings
    csv_files = {
        DS_COURSES: 'courses.csv',
        DS_ASSESSMENTS: 'assessments.csv',
        DS_VLE: 'vle.csv',
        DS_STUD_INFO: 'studentInfo.csv',
        DS_STUD_ASSESSMENTS: 'studentAssessment.csv',
        DS_STUD_REG: 'studentRegistration.csv',
        DS_STUD_VLE: 'studentVle.csv'
    }
    
    # Convert each CSV file with preprocessing
    for dataset_name, csv_filename in csv_files.items():
        csv_path = os.path.join(data_path, csv_filename)
        
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_filename} not found at {csv_path}")
            continue
            
        print(f"Converting {csv_filename} to {dataset_name}...")
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Apply preprocessing
            df = preprocess_dataframe(df, dataset_name)
            print(f"  Applied preprocessing, new shape: {df.shape}")
            print(f"  Index: {df.index.names}")
            
            # Store in H5 format
            store_manager.store_dataframe(dataset_name, df)
            print(f"  Successfully stored as {dataset_name}")
            
        except Exception as e:
            print(f"  Error converting {csv_filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nConversion complete! H5 file saved to: {hdf5_path}")
    
    # Verify the conversion
    print("\nVerifying conversion...")
    try:
        for dataset_name in csv_files.keys():
            try:
                df = store_manager.load_dataframe(dataset_name)
                print(f"  ✓ {dataset_name}: {len(df)} rows, {len(df.columns)} columns")
                print(f"    Index: {df.index.names}")
            except Exception as e:
                print(f"  ✗ {dataset_name}: Error loading - {e}")
    except Exception as e:
        print(f"Error during verification: {e}")

if __name__ == "__main__":
    fix_h5_preprocessing()
