import pandas as pd
import numpy as np

# File paths
input_file = '/home/sagemaker-user/T2D/data_T2D_Sep2025/T2D_Final_Sep2025.xlsx'
crosswalk_file = '/home/sagemaker-user/T2D/data_crosswalk_all/crosswalk_mrn.csv'
output_file = '/home/sagemaker-user/T2D/data_T2D_Sep2025/T2D_Final_Sep2025.csv'

# Read the Excel file
print("Reading Excel file...")
df = pd.read_excel(input_file, dtype={'person_id': str, 'MRN': str})
print(f"Original data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Ensure we're working with unique patients
# Check for duplicates based on MRN
duplicate_mrns = df['MRN'].duplicated().sum()
if duplicate_mrns > 0:
    print(f"\nWarning: Found {duplicate_mrns} duplicate MRNs in input file")
    print("Keeping first occurrence of each MRN...")
    df = df.drop_duplicates(subset=['MRN'], keep='first')
    print(f"Data shape after removing duplicates: {df.shape}")

# Count missing person_ids before
missing_before = df['person_id'].isna().sum()
print(f"\nMissing person_ids before: {missing_before}")

# Read the crosswalk table
print("\nReading crosswalk table...")
crosswalk = pd.read_csv(crosswalk_file, dtype={'PAT_MRN_ID': str, 'PEDSNET_ID': str})
print(f"Crosswalk shape before deduplication: {crosswalk.shape}")

# Remove duplicates - keep first occurrence of each MRN
crosswalk_unique = crosswalk.drop_duplicates(subset=['PAT_MRN_ID'], keep='first')
print(f"Crosswalk shape after deduplication: {crosswalk_unique.shape}")
duplicates_removed = len(crosswalk) - len(crosswalk_unique)
print(f"Removed {duplicates_removed} duplicate MRN entries")

# Create a mapping dictionary from MRN to person_id
# PAT_MRN_ID -> PEDSNET_ID
mrn_to_person_id = dict(zip(crosswalk_unique['PAT_MRN_ID'], crosswalk_unique['PEDSNET_ID']))
print(f"Crosswalk contains {len(mrn_to_person_id)} unique MRN-to-person_id mappings")

# Fill missing person_ids
# Only update rows where person_id is missing or empty
mask = df['person_id'].isna() | (df['person_id'] == '') | (df['person_id'] == 'nan')
df.loc[mask, 'person_id'] = df.loc[mask, 'MRN'].map(mrn_to_person_id)

# Count missing person_ids after
missing_after = (df['person_id'].isna() | (df['person_id'] == '') | (df['person_id'] == 'nan')).sum()
filled_count = missing_before - missing_after

print(f"\nMissing person_ids after: {missing_after}")
print(f"Successfully filled: {filled_count} person_ids")

# Save to CSV
print(f"\nSaving to CSV: {output_file}")
df.to_csv(output_file, index=False)
print("Done!")

# Summary statistics
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Total records: {len(df)}")
print(f"Person IDs filled: {filled_count}")
print(f"Still missing: {missing_after}")
if missing_after > 0:
    print(f"\nNote: {missing_after} records still have missing person_ids.")
    print("These MRNs may not exist in the crosswalk table.")