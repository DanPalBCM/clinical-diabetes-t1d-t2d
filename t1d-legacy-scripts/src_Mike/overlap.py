import pandas as pd
import boto3
import os
from io import StringIO

# Define ICD codes for each category
SEVERE_HYPOGLYCEMIA_CODES = [
    'E08.641',  # Diabetes mellitus due to underlying condition with hypoglycemia with coma
    'E10.641',  # Type 1 diabetes mellitus with hypoglycemia with coma
    'E13.641',  # Other specified diabetes mellitus with hypoglycemia with coma
    'E16.A2',   # Hypoglycemia level 2
    'E16.A3'    # Hypoglycemia level 3
]

UNSPECIFIED_HYPOGLYCEMIA_CODES = [
    'E08.649',  # Diabetes mellitus due to underlying condition with hypoglycemia without coma
    'E10.649',  # Type 1 diabetes mellitus with hypoglycemia without coma
    'E13.649',  # Other specified diabetes mellitus with hypoglycemia without coma
    'E16.0',    # Drug-induced hypoglycemia without coma
    'E16.1',    # Other hypoglycemia
    'E16.2',    # Hypoglycemia, unspecified
    'T38.3X1A', 'T38.3X1D', 'T38.3X1S',  # Poisoning accidental
    'T38.3X2A', 'T38.3X2D', 'T38.3X2S',  # Poisoning intentional self-harm
    'T38.3X4A', 'T38.3X4D', 'T38.3X4S',  # Poisoning undetermined
    'T38.3X5A', 'T38.3X5D', 'T38.3X5S'   # Adverse effect
]

SEIZURE_CODES = [
    'G40.89',  # Other seizures
    'G40.5',   # Epileptic seizures related to external causes
    'G40.6'    # Grand mal seizures, unspecified
]

def check_icd_code(code, icd_list):
    """Check if a code matches any pattern in the ICD list"""
    if pd.isna(code):
        return False
    
    # Extract ICD code from "description | code" format
    code_str = str(code).strip()
    if '|' in code_str:
        # Extract the part after the pipe symbol
        code = code_str.split('|')[-1].strip()
    else:
        code = code_str
    
    code = code.upper()
    
    for pattern in icd_list:
        pattern = pattern.upper()
        if pattern.endswith('.'):
            if code.startswith(pattern[:-1]):
                return True
        else:
            if code == pattern or code.startswith(pattern + '.'):
                return True
    return False

def load_s3_csv(bucket, key):
    """Load CSV file from S3"""
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
    return df

def analyze_hypoglycemia_groups(condition_df, person_df, dataset_name="Dataset"):
    """
    Analyze hypoglycemia groups in a dataset
    
    Parameters:
    - condition_df: DataFrame with condition occurrences (must have 'PERSON_ID' and ICD code column)
    - person_df: DataFrame with person data (must have 'PERSON_ID')
    - dataset_name: Name of the dataset for printing results
    """
    
    # Identify the ICD code column name
    possible_icd_columns = ['condition_source_value', 'icd_code', 'diagnosis_code', 'code', 
                           'condition_concept_name', 'condition_source_concept_name']
    icd_column = None
    
    print(f"\nSearching for ICD code column in {dataset_name}...")
    print(f"Available columns: {condition_df.columns.tolist()[:10]}...")  # Show first 10 columns
    
    for col in possible_icd_columns:
        if col in condition_df.columns:
            icd_column = col
            print(f"Using column '{icd_column}' as ICD code column")
            break
    
    if icd_column is None:
        # Try to identify the column by looking for one that contains ICD-like codes
        for col in condition_df.columns:
            if condition_df[col].dtype == 'object':
                sample = condition_df[col].dropna().head(10).astype(str)
                # Check if any values contain ICD patterns or pipe symbols
                if any('|' in val or val.upper().startswith(('E', 'G', 'T')) for val in sample):
                    icd_column = col
                    print(f"Auto-detected column '{icd_column}' as ICD code column")
                    break
    
    if icd_column is None:
        raise ValueError(f"Could not identify ICD code column in {dataset_name}")
    
    # Show sample of ICD codes for verification
    print(f"\nSample ICD codes from {icd_column}:")
    sample_codes = condition_df[icd_column].dropna().head(5)
    for code in sample_codes:
        print(f"  {str(code)[:100]}")  # Truncate long descriptions
    
    # Find patients in each group using the check_icd_code function
    print(f"\nAnalyzing conditions for {len(condition_df)} records...")
    
    # Create boolean masks for each condition group
    severe_mask = condition_df[icd_column].apply(lambda x: check_icd_code(x, SEVERE_HYPOGLYCEMIA_CODES))
    unspecified_mask = condition_df[icd_column].apply(lambda x: check_icd_code(x, UNSPECIFIED_HYPOGLYCEMIA_CODES))
    seizure_mask = condition_df[icd_column].apply(lambda x: check_icd_code(x, SEIZURE_CODES))
    
    # Get unique patient sets
    severe_patients = set(condition_df[severe_mask]['PERSON_ID'].unique())
    unspecified_patients = set(condition_df[unspecified_mask]['PERSON_ID'].unique())
    seizure_patients = set(condition_df[seizure_mask]['PERSON_ID'].unique())
    
    # Print detailed results
    print(f"\n{'='*60}")
    print(f"Analysis Results for {dataset_name}")
    print(f"{'='*60}")
    print(f"Total unique patients in person table: {len(person_df['PERSON_ID'].unique())}")
    print(f"Total unique patients with conditions: {len(condition_df['PERSON_ID'].unique())}")
    
    print(f"\nGroup 1 - Severe Hypoglycemia: {len(severe_patients)} unique patients")
    # NOTE: do not print raw PERSON_ID values here (real patient
    # identifiers) -- counts only, for a public repo.
    
    print(f"\nGroup 2 - Unspecified Hypoglycemia: {len(unspecified_patients)} unique patients")
    # NOTE: do not print raw PERSON_ID values here (real patient
    # identifiers) -- counts only, for a public repo.
    
    print(f"\nGroup 3 - Seizures: {len(seizure_patients)} unique patients")
    # NOTE: do not print raw PERSON_ID values here (real patient
    # identifiers) -- counts only, for a public repo.
    
    # Check overlaps between groups
    print(f"\nOverlaps between groups:")
    print(f"  Severe & Unspecified: {len(severe_patients & unspecified_patients)} patients")
    print(f"  Severe & Seizures: {len(severe_patients & seizure_patients)} patients")
    print(f"  Unspecified & Seizures: {len(unspecified_patients & seizure_patients)} patients")
    print(f"  All three groups: {len(severe_patients & unspecified_patients & seizure_patients)} patients")
    
    # Show example conditions found for each group (for validation)
    if len(severe_patients) > 0:
        print(f"\nExample severe hypoglycemia conditions found:")
        severe_examples = condition_df[severe_mask][icd_column].value_counts().head(3)
        for code, count in severe_examples.items():
            print(f"  {str(code)[:80]}: {count} occurrences")
    
    return severe_patients, unspecified_patients, seizure_patients

def main():
    """Main execution function"""
    
    print("Starting T1D Hypoglycemia Analysis")
    print("="*60)
    
    # 1. Load the first dataset (260 patients with T1D Hypoglycemia)
    print("\nLoading first dataset (T1D Hypoglycemia patients)...")
    try:
        hypo_person_df = pd.read_csv("/home/sagemaker-user/T2D/src_Mike/data/person.csv")
        print(f"Loaded {len(hypo_person_df)} records from hypoglycemia dataset")
        print(f"Unique patients: {len(hypo_person_df['PERSON_ID'].unique())}")
    except Exception as e:
        print(f"Error loading first dataset: {e}")
        return
    
    # 2. Load T1D cohort data from S3
    print("\nLoading T1D cohort data from S3...")
    bucket = "dsw-sagemaker-dev-s3"
    base_path = "T1D_Tosur/data/T1D_OMOP_variables/"
    
    try:
        # Load person data
        print("Loading person.csv from S3...")
        t1d_person_df = load_s3_csv(bucket, base_path + "person.csv")
        print(f"Loaded {len(t1d_person_df)} person records from T1D cohort")
        print(f"Unique patients: {len(t1d_person_df['PERSON_ID'].unique())}")
        
        # Load condition occurrence data
        print("\nLoading condition_occurrence.csv from S3...")
        t1d_condition_df = load_s3_csv(bucket, base_path + "condition_occurrence.csv")
        print(f"Loaded {len(t1d_condition_df)} condition occurrence records")
        print(f"Unique patients with conditions: {len(t1d_condition_df['PERSON_ID'].unique())}")
        
    except Exception as e:
        print(f"Error loading S3 data: {e}")
        print(f"Please ensure you have AWS credentials configured and access to the bucket.")
        return
    
    # 3. Analyze T1D cohort for hypoglycemia groups
    print("\n" + "="*60)
    print("ANALYZING T1D COHORT (S3 DATA)")
    print("="*60)
    t1d_severe, t1d_unspecified, t1d_seizure = analyze_hypoglycemia_groups(
        t1d_condition_df, 
        t1d_person_df, 
        "T1D Cohort (S3)"
    )
    
    # 4. Check if there's condition data for the first dataset
    print("\n" + "="*60)
    print("ANALYZING HYPOGLYCEMIA DATASET (LOCAL DATA)")
    print("="*60)
    
    # Try multiple possible paths for condition occurrence file
    possible_paths = [
        "/home/sagemaker-user/T2D/src_Mike/data/condition_occurrence.csv",
        "/home/sagemaker-user/T2D/src_Mike/data/condition.csv",
        "/home/sagemaker-user/T2D/src_Mike/data/conditions.csv"
    ]
    
    hypo_condition_df = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found condition occurrence data at: {path}")
            hypo_condition_df = pd.read_csv(path)
            break
    
    if hypo_condition_df is not None:
        hypo_severe, hypo_unspecified, hypo_seizure = analyze_hypoglycemia_groups(
            hypo_condition_df,
            hypo_person_df,
            "T1D Hypoglycemia Dataset (Local)"
        )
    else:
        print(f"Note: No condition occurrence file found in expected locations")
        print("Searched paths:", possible_paths)
        print("Assuming all patients in this dataset have hypoglycemia based on dataset description")
        hypo_severe = set(hypo_person_df['PERSON_ID'].unique())
        hypo_unspecified = set()
        hypo_seizure = set()
    
    # 5. Compare the datasets - overlap analysis
    print("\n" + "="*60)
    print("OVERLAP ANALYSIS: T1D Cohort vs T1D Hypoglycemia Dataset")
    print("="*60)
    
    # Get all patient IDs from both datasets
    hypo_patients = set(hypo_person_df['PERSON_ID'].unique())
    t1d_all_patients = set(t1d_person_df['PERSON_ID'].unique())
    
    # Overall overlap
    overlap_all = hypo_patients & t1d_all_patients
    print(f"\nOverall Dataset Comparison:")
    print(f"  Total patients in hypoglycemia dataset: {len(hypo_patients)}")
    print(f"  Total patients in T1D cohort: {len(t1d_all_patients)}")
    print(f"  Patients in BOTH datasets: {len(overlap_all)}")
    print(f"  Hypoglycemia patients NOT in T1D cohort: {len(hypo_patients - t1d_all_patients)}")
    print(f"  T1D cohort patients NOT in hypoglycemia dataset: {len(t1d_all_patients - hypo_patients)}")
    
    # Severe hypoglycemia overlap analysis
    print(f"\n" + "-"*60)
    print(f"SEVERE HYPOGLYCEMIA OVERLAP ANALYSIS")
    print(f"-"*60)
    print(f"T1D cohort patients with severe hypoglycemia: {len(t1d_severe)}")
    
    if len(t1d_severe) > 0:
        overlap_severe = t1d_severe & hypo_patients
        not_in_hypo = t1d_severe - hypo_patients
        
        print(f"\nOf the {len(t1d_severe)} T1D patients with severe hypoglycemia:")
        print(f"  ✓ PRESENT in hypoglycemia dataset: {len(overlap_severe)} patients")
        print(f"  ✗ NOT PRESENT in hypoglycemia dataset: {len(not_in_hypo)} patients")
        
        # Calculate percentages
        percent_present = (len(overlap_severe) / len(t1d_severe)) * 100
        percent_absent = (len(not_in_hypo) / len(t1d_severe)) * 100
        print(f"\nPercentages:")
        print(f"  {percent_present:.1f}% of T1D severe hypoglycemia patients ARE in hypoglycemia dataset")
        print(f"  {percent_absent:.1f}% of T1D severe hypoglycemia patients ARE NOT in hypoglycemia dataset")
        
        # Show sample of missing patients if there are any
        # NOTE: do not print raw PERSON_ID values here (real patient
        # identifiers) -- counts only, for a public repo.
    
    # Export results to CSV for further analysis
    print("\n" + "="*60)
    print("EXPORTING RESULTS")
    print("="*60)
    
    # Create summary DataFrame
    summary_data = {
        'Dataset': ['T1D Cohort (S3)', 'T1D Hypoglycemia (Local)'],
        'Total Patients': [len(t1d_all_patients), len(hypo_patients)],
        'Severe Hypoglycemia': [len(t1d_severe), len(hypo_severe) if 'hypo_severe' in locals() else 'N/A'],
        'Unspecified Hypoglycemia': [len(t1d_unspecified), len(hypo_unspecified) if 'hypo_unspecified' in locals() else 'N/A'],
        'Seizures': [len(t1d_seizure), len(hypo_seizure) if 'hypo_seizure' in locals() else 'N/A']
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('hypoglycemia_analysis_summary.csv', index=False)
    print("✓ Summary saved to: hypoglycemia_analysis_summary.csv")
    
    # Export patient lists for severe hypoglycemia.
    # WARNING: these .to_csv() calls write real PERSON_ID values to local
    # files. Never commit their output to this repository -- these files
    # must stay local/gitignored when run against real data.
    if len(t1d_severe) > 0:
        # All severe hypoglycemia patients from T1D cohort
        pd.DataFrame({'PERSON_ID': sorted(list(t1d_severe))}).to_csv(
            't1d_severe_hypoglycemia_patients.csv', index=False
        )
        print("✓ T1D severe hypoglycemia patients saved to: t1d_severe_hypoglycemia_patients.csv")
        
        # Patients missing from hypoglycemia dataset
        if len(t1d_severe - hypo_patients) > 0:
            pd.DataFrame({'PERSON_ID': sorted(list(t1d_severe - hypo_patients))}).to_csv(
                't1d_severe_not_in_hypo_dataset.csv', index=False
            )
            print("✓ Missing patients saved to: t1d_severe_not_in_hypo_dataset.csv")
    
    # Export overlap details
    overlap_details = {
        'PERSON_ID': sorted(list(overlap_all)),
        'in_t1d_severe': [pid in t1d_severe for pid in sorted(list(overlap_all))],
        'in_t1d_unspecified': [pid in t1d_unspecified for pid in sorted(list(overlap_all))],
        'in_t1d_seizure': [pid in t1d_seizure for pid in sorted(list(overlap_all))]
    }
    if len(overlap_all) > 0:
        overlap_df = pd.DataFrame(overlap_details)
        overlap_df.to_csv('dataset_overlap_details.csv', index=False)
        print("✓ Overlap details saved to: dataset_overlap_details.csv")
    
    print("\n" + "="*60)
    print("Analysis complete! Check the CSV files for detailed results.")
    print("="*60)

if __name__ == "__main__":
    main()