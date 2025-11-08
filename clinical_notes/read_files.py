import pandas as pd
import numpy as np
from datetime import datetime

# Define chunk size for memory-efficient processing
chunk_size = 10000

# File paths
t1d_file_path = 'T1D_patients.csv'
t1d_parent_file = '/home/sagemaker-user/T2D/clinical_notes/parent_files/CROSSWALK_PATINENTIDS_person_id.csv'
t2d_file_path = 'T2D_patients.csv'
t2d_parent_file = '/home/sagemaker-user/T2D/clinical_notes/parent_files/T2DPopulation_7242025 (2).xlsx'

print("=" * 80)
print("TASK 1: PATIENT ID INTEGRITY CHECK")
print("=" * 80)

# --- T1D Dataset Check ---
print("\n--- T1D Dataset Patient Comparison ---")

# Read parent file for T1D
t1d_parent = pd.read_csv(t1d_parent_file)
parent_t1d_ids = set(t1d_parent['PATIENTID'].dropna().unique())

# Process T1D data in chunks to collect unique patient IDs
t1d_patient_ids = set()
for chunk in pd.read_csv(t1d_file_path, chunksize=chunk_size):
    t1d_patient_ids.update(chunk['PERSON_ID'].dropna().unique())

# Calculate overlaps and differences
t1d_overlap = parent_t1d_ids & t1d_patient_ids
t1d_missing = parent_t1d_ids - t1d_patient_ids
t1d_extra = t1d_patient_ids - parent_t1d_ids

print(f"Total patients in T1D parent file: {len(parent_t1d_ids)}")
print(f"Total patients in T1D data file: {len(t1d_patient_ids)}")
print(f"Patients in common: {len(t1d_overlap)}")
print(f"Patients missing from data (in parent but not in data): {len(t1d_missing)}")
print(f"Extra patients in data (in data but not in parent): {len(t1d_extra)}")

# --- T2D Dataset Check ---
print("\n--- T2D Dataset Patient Comparison ---")

# Read parent file for T2D (Excel file)
t2d_parent = pd.read_excel(t2d_parent_file)
parent_t2d_ids = set(t2d_parent['PatientID'].dropna().unique())

# Process T2D data in chunks to collect unique patient IDs
t2d_patient_ids = set()
for chunk in pd.read_csv(t2d_file_path, chunksize=chunk_size):
    t2d_patient_ids.update(chunk['PERSON_ID'].dropna().unique())

# Calculate overlaps and differences
t2d_overlap = parent_t2d_ids & t2d_patient_ids
t2d_missing = parent_t2d_ids - t2d_patient_ids
t2d_extra = t2d_patient_ids - parent_t2d_ids

print(f"Total patients in T2D parent file: {len(parent_t2d_ids)}")
print(f"Total patients in T2D data file: {len(t2d_patient_ids)}")
print(f"Patients in common: {len(t2d_overlap)}")
print(f"Patients missing from data (in parent but not in data): {len(t2d_missing)}")
print(f"Extra patients in data (in data but not in parent): {len(t2d_extra)}")

print("\n" + "=" * 80)
print("TASK 2 & 3: SUMMARY STATISTICS")
print("=" * 80)

# --- T1D Dataset Statistics ---
print("\n--- T1D Dataset Summary Statistics ---")

# Initialize counters for T1D
t1d_total_notes = 0
t1d_unique_patients = set()
t1d_authortypes = set()
t1d_noteservices = set()
t1d_min_date = None
t1d_max_date = None

# Process T1D in chunks
for chunk in pd.read_csv(t1d_file_path, chunksize=chunk_size):
    t1d_total_notes += len(chunk)
    t1d_unique_patients.update(chunk['PERSON_ID'].dropna().unique())
    
    if 'AUTHORTYPE' in chunk.columns:
        t1d_authortypes.update(chunk['AUTHORTYPE'].dropna().unique())
    
    if 'NOTESERVICE' in chunk.columns:
        t1d_noteservices.update(chunk['NOTESERVICE'].dropna().unique())
    
    if 'CREATIONINSTANT' in chunk.columns:
        # Convert to datetime and find min/max
        chunk['CREATIONINSTANT'] = pd.to_datetime(chunk['CREATIONINSTANT'], errors='coerce')
        chunk_min = chunk['CREATIONINSTANT'].min()
        chunk_max = chunk['CREATIONINSTANT'].max()
        
        if t1d_min_date is None or (pd.notna(chunk_min) and chunk_min < t1d_min_date):
            t1d_min_date = chunk_min
        if t1d_max_date is None or (pd.notna(chunk_max) and chunk_max > t1d_max_date):
            t1d_max_date = chunk_max

print(f"Total number of notes: {t1d_total_notes:,}")
print(f"Total unique patients: {len(t1d_unique_patients):,}")
print(f"Unique AUTHORTYPE values: {len(t1d_authortypes)}")
if t1d_authortypes:
    print(f"  AUTHORTYPE list: {sorted(list(t1d_authortypes))[:10]}")  # Show first 10
    if len(t1d_authortypes) > 10:
        print(f"  ... and {len(t1d_authortypes) - 10} more")
print(f"Unique NOTESERVICE values: {len(t1d_noteservices)}")
if t1d_noteservices:
    print(f"  NOTESERVICE list: {sorted(list(t1d_noteservices))[:10]}")  # Show first 10
    if len(t1d_noteservices) > 10:
        print(f"  ... and {len(t1d_noteservices) - 10} more")
print(f"Date range (CREATIONINSTANT):")
print(f"  Earliest date: {t1d_min_date}")
print(f"  Latest date: {t1d_max_date}")
if t1d_min_date and t1d_max_date:
    date_span = (t1d_max_date - t1d_min_date).days
    print(f"  Time span: {date_span} days ({date_span/365.25:.1f} years)")

# --- T2D Dataset Statistics ---
print("\n--- T2D Dataset Summary Statistics ---")

# Initialize counters for T2D
t2d_total_notes = 0
t2d_unique_patients = set()
t2d_authortypes = set()
t2d_noteservices = set()
t2d_min_date = None
t2d_max_date = None

# Process T2D in chunks
for chunk in pd.read_csv(t2d_file_path, chunksize=chunk_size):
    t2d_total_notes += len(chunk)
    t2d_unique_patients.update(chunk['PERSON_ID'].dropna().unique())
    
    if 'AUTHORTYPE' in chunk.columns:
        t2d_authortypes.update(chunk['AUTHORTYPE'].dropna().unique())
    
    if 'NOTESERVICE' in chunk.columns:
        t2d_noteservices.update(chunk['NOTESERVICE'].dropna().unique())
    
    if 'CREATIONINSTANT' in chunk.columns:
        # Convert to datetime and find min/max
        chunk['CREATIONINSTANT'] = pd.to_datetime(chunk['CREATIONINSTANT'], errors='coerce')
        chunk_min = chunk['CREATIONINSTANT'].min()
        chunk_max = chunk['CREATIONINSTANT'].max()
        
        if t2d_min_date is None or (pd.notna(chunk_min) and chunk_min < t2d_min_date):
            t2d_min_date = chunk_min
        if t2d_max_date is None or (pd.notna(chunk_max) and chunk_max > t2d_max_date):
            t2d_max_date = chunk_max

print(f"Total number of notes: {t2d_total_notes:,}")
print(f"Total unique patients: {len(t2d_unique_patients):,}")
print(f"Unique AUTHORTYPE values: {len(t2d_authortypes)}")
if t2d_authortypes:
    print(f"  AUTHORTYPE list: {sorted(list(t2d_authortypes))[:10]}")  # Show first 10
    if len(t2d_authortypes) > 10:
        print(f"  ... and {len(t2d_authortypes) - 10} more")
print(f"Unique NOTESERVICE values: {len(t2d_noteservices)}")
if t2d_noteservices:
    print(f"  NOTESERVICE list: {sorted(list(t2d_noteservices))[:10]}")  # Show first 10
    if len(t2d_noteservices) > 10:
        print(f"  ... and {len(t2d_noteservices) - 10} more")
print(f"Date range (CREATIONINSTANT):")
print(f"  Earliest date: {t2d_min_date}")
print(f"  Latest date: {t2d_max_date}")
if t2d_min_date and t2d_max_date:
    date_span = (t2d_max_date - t2d_min_date).days
    print(f"  Time span: {date_span} days ({date_span/365.25:.1f} years)")

print("\n" + "=" * 80)
print("DATA INTEGRITY CHECK COMPLETE")
print("=" * 80)