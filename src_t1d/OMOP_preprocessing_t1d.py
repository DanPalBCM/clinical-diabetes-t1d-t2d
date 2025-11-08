import pandas as pd
import numpy as np
import gc
from datetime import datetime
import boto3
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Define T1D-specific medication mappings
T1D_DRUG_CLASSES = {
    'Insulins': [
        'insulin aspart', 'insulin degludec', 'insulin detemir', 'insulin glargine',
        'insulin glulisine', 'insulin human', 'insulin regular', 'insulin nph',
        'insulin isophane', 'insulin lispro', 'insulin lispro protamine',
        'inhaled human insulin', 'technosphere insulin', 'insulin human isophane',
        'insulin isophane/regular'
    ],
    'Amylin_analogue': ['pramlintide','symlin']
}

# Define condition mappings with ICD codes (same as T2D)
CONDITIONS = {
    'DKA': {
        'ICD9': [
            '250.11',  # Diabetes mellitus with ketoacidosis, Type 1
            '250.13',  # Diabetes mellitus with ketoacidosis, Type 2
            '250.10',  # Diabetes mellitus with ketoacidosis, Type 1, uncontrolled
            '250.12',  # Diabetes mellitus with ketoacidosis, Type 2, uncontrolled
        ],
        'ICD10': [
            'E10.10',  # Type 1 diabetes mellitus with ketoacidosis, without coma
            'E10.11',  # Type 1 diabetes mellitus with ketoacidosis with coma
            'E11.10',  # Type 2 diabetes mellitus with ketoacidosis, without coma
            'E11.11',  # Type 2 diabetes mellitus with ketoacidosis with coma
            #'E13.1',   # Other specified diabetes mellitus with ketoacidosis, without coma
            #'E08.10',  # Diabetes due to underlying condition with ketoacidosis, without coma
            #'E08.11',  # Diabetes due to underlying condition with ketoacidosis with coma
            #'E09.10',  # Type 1 diabetes mellitus, unspecified
            #'E09.11',  # Type 2 diabetes mellitus, unspecified
        ]
    },
    'Ketosis': {
        'ICD9': ['276.2', '790.6'],
        'ICD10': ['E87.2']
    },
    'Dyslipidemia': {
        'ICD9': ['272.'],
        'ICD10': ['E78.0', 'E78.1', 'E78.2', 'E78.3', 'E78.4', 'E78.5', 'E78.6']
    },
    'Hypertension': {
        'ICD9': ['401.', '402.', '403.', '404.', '405.'],
        'ICD10': ['I10.', 'I11.', 'I12.', 'I13.', 'I15.', 'H35.03', 'I67.4']
    },
    'Diabetic_Retinopathy': {
        'ICD9': ['362.01', '362.02', '362.03', '362.04', '362.05', '362.06'],
        'ICD10': ['E08.35', 'E09.35', 'E10.35', 'E11.35', 'E13.35', 'E08.31', 'E08.37',
                  'E09.31', 'E09.37', 'E10.31', 'E10.37', 'E11.31', 'E11.37', 'E13.31',
                  'E13.37', 'E08.32', 'E09.32', 'E10.32', 'E11.32', 'E13.32', 'E08.33',
                  'E09.33', 'E10.33', 'E11.33', 'E13.33', 'E08.34', 'E09.34', 'E10.34',
                  'E11.34', 'E13.34']
    },
    'Microalbuminuria': {
        'ICD9': ['791.06'],
        'ICD10': ['R80.9']
    },
    'Neuropathy': {
        'ICD9': ['250.61', '250.63', '250.60', '250.62', '357.2'],
        'ICD10': ['E10.40', 'E10.41', 'E10.42', 'E10.43', 'E10.44', 'E10.49',
                  'E11.40', 'E11.41', 'E11.42', 'E11.45']
    },
    'Hypoglycemia': {
        'ICD9': ['250.3', '250.8', '251.0', '251.1', '251.2', '270.3', '775.0', '775.6', '962.39'],
        'ICD10': ['E08.641', 'E08.649', 'E09.641', 'E09.649', 'E10.641', 'E10.649',
                  'E11.641', 'E11.649', 'E13.641', 'E13.649', 'E15', 'E16.0', 'E16.1',
                  'E16.2', 'T38.3X1A', 'T38.3X1D', 'T38.3X1S', 'T38.3X2A', 'T38.3X2D',
                  'T38.3X2S', 'T38.3X3A', 'T38.3X3D', 'T38.3X3S', 'T38.3X4A', 'T38.3X4D',
                  'T38.3X4S', 'T38.3X5A', 'T38.3X5D', 'T38.3X5S']
    }
}

# T1D ICD codes for diagnosis date
T1D_CODES = {
    'ICD9': ['250.01', '250.03', '250.11', '250.13', '250.21', '250.23', '250.31', '250.33',
             '250.41', '250.43', '250.51', '250.53', '250.61', '250.63', '250.71', '250.73',
             '250.81', '250.83', '250.91', '250.93'],
    'ICD10': ['E10.', 'E10.0', 'E10.1', 'E10.2', 'E10.3', 'E10.4', 'E10.5', 'E10.6', 
              'E10.7', 'E10.8', 'E10.9']
}

def read_s3_csv(s3_client, bucket, key):
    """Read CSV from S3"""
    print(f"  Reading s3://{bucket}/{key}")
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj['Body'])

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
        if pattern.endswith('.'):
            if code.startswith(pattern[:-1]):
                return True
        else:
            if code == pattern or code.startswith(pattern + '.'):
                return True
    return False

def process_demographics(person_df):
    """Process person table to extract demographics"""
    print("Processing demographics...")
    
    # Select relevant columns
    demo_df = person_df[['PERSON_ID', 'GENDER_CONCEPT_ID', 'BIRTH_DATETIME', 
                         'RACE_CONCEPT_ID', 'ETHNICITY_CONCEPT_ID']].copy()
    
    # Map gender (8507 = Male, 8532 = Female in OMOP)
    demo_df['sex'] = demo_df['GENDER_CONCEPT_ID'].map({8507: 'M', 8532: 'F'})
    
    # Convert BIRTH_DATETIME to birth_date
    demo_df['birth_date'] = pd.to_datetime(demo_df['BIRTH_DATETIME'], errors='coerce')
    
    # Map race to the 4 categories: White, Black, Asian, Other
    race_map = {
        8527: 'White',
        8516: 'Black',
        8515: 'Asian',
        8657: 'Other',  # Native American -> Other
        8557: 'Other',  # Pacific Islander -> Other
        0: 'Other'      # Unknown -> Other
    }
    
    # Map ethnicity to 2 categories: Hispanic or Latino, Other
    ethnicity_map = {
        38003563: 'Hispanic or Latino',
        38003564: 'Other',  # Not Hispanic -> Other
        0: 'Other'          # Unknown -> Other
    }
    
    demo_df['race'] = demo_df['RACE_CONCEPT_ID'].map(race_map).fillna('Other')
    demo_df['ethnicity'] = demo_df['ETHNICITY_CONCEPT_ID'].map(ethnicity_map).fillna('Other')
    
    # Print summary
    print(f"  Birth dates found: {demo_df['birth_date'].notna().sum()}/{len(demo_df)}")
    print(f"  Sex distribution: M={len(demo_df[demo_df['sex']=='M'])}, F={len(demo_df[demo_df['sex']=='F'])}")
    
    return demo_df[['PERSON_ID', 'sex', 'birth_date', 'race', 'ethnicity']]

def get_t1d_diagnosis_date(condition_df):
    """Extract earliest T1D diagnosis date for each patient"""
    print("Finding T1D diagnosis dates...")
    
    # Filter for T1D codes
    t1d_conditions = condition_df[
        condition_df['CONDITION_SOURCE_VALUE'].apply(
            lambda x: check_icd_code(x, T1D_CODES['ICD9'] + T1D_CODES['ICD10'])
        )
    ].copy()
    
    print(f"  Found {len(t1d_conditions)} T1D diagnosis records")
    
    # Convert date columns
    t1d_conditions['condition_date'] = pd.to_datetime(t1d_conditions['CONDITION_START_DATE'])
    
    # Get earliest T1D date per patient
    t1d_dates = t1d_conditions.groupby('PERSON_ID')['condition_date'].min().reset_index()
    t1d_dates.columns = ['PERSON_ID', 't1d_diagnosis_date']
    
    print(f"  Found T1D diagnosis dates for {len(t1d_dates)} patients")
    
    return t1d_dates

def process_medications(drug_df, t1d_dates):
    """Process drug exposure table to extract medication information"""
    print("Processing medications...")
    
    # Convert dates
    drug_df['drug_date'] = pd.to_datetime(drug_df['DRUG_EXPOSURE_START_DATE'])
    
    # Merge with T1D dates
    drug_df = drug_df.merge(t1d_dates, on='PERSON_ID', how='left')
    
    # Calculate years from diagnosis
    drug_df['years_from_diagnosis'] = (
        (drug_df['drug_date'] - drug_df['t1d_diagnosis_date']).dt.days / 365.25
    )
    
    # Initialize result dataframe
    patient_drugs = pd.DataFrame({'PERSON_ID': drug_df['PERSON_ID'].unique()})
    
    # Process each drug class
    for drug_class, drug_list in T1D_DRUG_CLASSES.items():
        print(f"  Processing {drug_class}...")
        
        # Create pattern for matching
        pattern = '|'.join(drug_list)
        
        # Find matching drugs
        class_drugs = drug_df[
            drug_df['DRUG_SOURCE_VALUE'].str.lower().str.contains(pattern, na=False, regex=True)
        ].copy()
        
        if len(class_drugs) > 0:
            print(f"    Found {len(class_drugs)} records for {drug_class}")
            
            # Get earliest drug date and years from diagnosis per patient
            drug_summary = class_drugs.groupby('PERSON_ID').agg({
                'drug_date': 'min',
                'years_from_diagnosis': 'min'
            }).reset_index()
            
            drug_summary[drug_class] = 1
            drug_summary.rename(columns={
                'drug_date': f'{drug_class}_date',
                'years_from_diagnosis': f'{drug_class}_years_from_diagnosis'
            }, inplace=True)
            
            # Merge with patient_drugs
            patient_drugs = patient_drugs.merge(drug_summary, on='PERSON_ID', how='left')
        else:
            print(f"    No records found for {drug_class}")
            patient_drugs[drug_class] = 0
            patient_drugs[f'{drug_class}_date'] = pd.NaT
            patient_drugs[f'{drug_class}_years_from_diagnosis'] = np.nan
    
    # Fill NaN values
    for drug_class in T1D_DRUG_CLASSES.keys():
        patient_drugs[drug_class] = patient_drugs[drug_class].fillna(0).astype(int)
    
    print(f"  Medication summary: {len(patient_drugs)} patients processed")
    
    return patient_drugs

def process_conditions(condition_df, t1d_dates):
    """Process condition occurrence table to extract relevant conditions"""
    print("Processing conditions...")
    
    # Convert dates
    condition_df['condition_date'] = pd.to_datetime(condition_df['CONDITION_START_DATE'])
    
    # Merge with T1D dates
    condition_df = condition_df.merge(t1d_dates, on='PERSON_ID', how='left')
    
    # Calculate years from diagnosis
    condition_df['years_from_diagnosis'] = (
        (condition_df['condition_date'] - condition_df['t1d_diagnosis_date']).dt.days / 365.25
    )
    
    # Initialize result dataframe
    patient_conditions = pd.DataFrame({'PERSON_ID': condition_df['PERSON_ID'].unique()})
    
    # Process each condition
    for condition_name, icd_codes in CONDITIONS.items():
        print(f"  Processing {condition_name}...")
        
        # Find matching conditions
        condition_matches = condition_df[
            condition_df['CONDITION_SOURCE_VALUE'].apply(
                lambda x: check_icd_code(x, icd_codes['ICD9'] + icd_codes['ICD10'])
            )
        ].copy()
        
        if len(condition_matches) > 0:
            print(f"    Found {len(condition_matches)} records for {condition_name}")
            
            # Get earliest condition date and years from diagnosis per patient
            condition_summary = condition_matches.groupby('PERSON_ID').agg({
                'condition_date': 'min',
                'years_from_diagnosis': 'min'
            }).reset_index()
            
            condition_summary[condition_name] = 1
            condition_summary.rename(columns={
                'condition_date': f'{condition_name}_date',
                'years_from_diagnosis': f'{condition_name}_years_from_diagnosis'
            }, inplace=True)
            
            # Merge with patient_conditions
            patient_conditions = patient_conditions.merge(condition_summary, on='PERSON_ID', how='left')
        else:
            print(f"    No records found for {condition_name}")
            patient_conditions[condition_name] = 0
            patient_conditions[f'{condition_name}_date'] = pd.NaT
            patient_conditions[f'{condition_name}_years_from_diagnosis'] = np.nan
    
    # Fill NaN values
    for condition_name in CONDITIONS.keys():
        patient_conditions[condition_name] = patient_conditions[condition_name].fillna(0).astype(int)
    
    print(f"  Conditions summary: {len(patient_conditions)} patients processed")
    
    return patient_conditions

def process_cohort(cohort_name, source_prefix, output_suffix, s3_client, 
                  bucket='dsw-sagemaker-dev-s3'):
    """Process a single T1D cohort"""
    print(f"\n{'=' * 60}")
    print(f"PROCESSING: {cohort_name}")
    print(f"{'=' * 60}")
    
    print(f"Source: s3://{bucket}/{source_prefix}")
    
    # Read the tables
    try:
        person_df = read_s3_csv(s3_client, bucket, f'{source_prefix}person.csv')
        condition_df = read_s3_csv(s3_client, bucket, f'{source_prefix}condition_occurrence.csv')
        drug_df = read_s3_csv(s3_client, bucket, f'{source_prefix}drug_exposure.csv')
        
        print(f"\nLoaded {len(person_df)} patients")
        print(f"Loaded {len(condition_df)} condition records")
        print(f"Loaded {len(drug_df)} drug records")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Process demographics
    demographics = process_demographics(person_df)
    del person_df
    gc.collect()
    
    # Get T1D diagnosis dates
    t1d_dates = get_t1d_diagnosis_date(condition_df)
    
    # Calculate age at diagnosis
    demographics = demographics.merge(t1d_dates, on='PERSON_ID', how='left')
    demographics['age_at_diagnosis'] = (
        (demographics['t1d_diagnosis_date'] - demographics['birth_date']).dt.days / 365.25
    )
    
    # Process medications
    medications = process_medications(drug_df, t1d_dates)
    del drug_df
    gc.collect()
    
    # Process conditions
    conditions = process_conditions(condition_df, t1d_dates)
    del condition_df
    gc.collect()
    
    # Merge all data
    print("\nMerging all data...")
    final_df = demographics.merge(medications, on='PERSON_ID', how='left')
    final_df = final_df.merge(conditions, on='PERSON_ID', how='left')
    
    # Reorder columns
    base_cols = ['PERSON_ID', 'sex', 'age_at_diagnosis', 'race', 'ethnicity', 
                 'birth_date', 't1d_diagnosis_date']
    
    drug_cols = []
    for drug_class in T1D_DRUG_CLASSES.keys():
        drug_cols.extend([drug_class, f'{drug_class}_date', f'{drug_class}_years_from_diagnosis'])
    
    condition_cols = []
    for condition in CONDITIONS.keys():
        condition_cols.extend([condition, f'{condition}_date', f'{condition}_years_from_diagnosis'])
    
    final_cols = base_cols + drug_cols + condition_cols
    final_df = final_df[final_cols]
    
    # Save to S3
    output_key = f'{source_prefix}T1D_patient_features_{output_suffix}.csv'
    csv_buffer = StringIO()
    final_df.to_csv(csv_buffer, index=False)
    
    print(f"\nSaving results to s3://{bucket}/{output_key}")
    s3_client.put_object(Bucket=bucket, Key=output_key, Body=csv_buffer.getvalue())
    
    # Print summary statistics
    print(f"\n{'=' * 40}")
    print(f"SUMMARY FOR {cohort_name}")
    print(f"{'=' * 40}")
    print(f"Total patients: {len(final_df)}")
    print(f"Patients with T1D diagnosis date: {final_df['t1d_diagnosis_date'].notna().sum()}")
    
    if final_df['age_at_diagnosis'].notna().sum() > 0:
        print(f"Average age at diagnosis: {final_df['age_at_diagnosis'].mean():.1f} years")
        print(f"Median age at diagnosis: {final_df['age_at_diagnosis'].median():.1f} years")
    
    print("\n--- Demographics ---")
    print("Sex distribution:")
    print(final_df['sex'].value_counts())
    print("\nRace distribution:")
    print(final_df['race'].value_counts())
    print("\nEthnicity distribution:")
    print(final_df['ethnicity'].value_counts())
    
    print("\n--- Medication Usage ---")
    for drug_class in T1D_DRUG_CLASSES.keys():
        count = final_df[drug_class].sum()
        pct = (count / len(final_df)) * 100 if len(final_df) > 0 else 0
        print(f"{drug_class}: {count} ({pct:.1f}%)")
    
    print("\n--- Condition Prevalence ---")
    for condition in CONDITIONS.keys():
        count = final_df[condition].sum()
        pct = (count / len(final_df)) * 100 if len(final_df) > 0 else 0
        print(f"{condition}: {count} ({pct:.1f}%)")
    
    return final_df

def generate_comparison_report(t1d_df, t1d_hypo_df, s3_client, bucket='dsw-sagemaker-dev-s3'):
    """Generate a comparison report between the two T1D cohorts"""
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON REPORT")
    print("=" * 60)
    
    report = []
    report.append("T1D COHORTS COMPARISON REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Basic statistics
    report.append("COHORT SIZES:")
    report.append(f"  T1D Filtered Cohort: {len(t1d_df)} patients")
    report.append(f"  T1D Hypoglycemia Cohort: {len(t1d_hypo_df)} patients")
    report.append("")
    
    # Age at diagnosis comparison
    report.append("AGE AT DIAGNOSIS:")
    if t1d_df['age_at_diagnosis'].notna().sum() > 0:
        report.append(f"  T1D Filtered - Mean: {t1d_df['age_at_diagnosis'].mean():.1f}, Median: {t1d_df['age_at_diagnosis'].median():.1f}")
    if t1d_hypo_df['age_at_diagnosis'].notna().sum() > 0:
        report.append(f"  T1D Hypoglycemia - Mean: {t1d_hypo_df['age_at_diagnosis'].mean():.1f}, Median: {t1d_hypo_df['age_at_diagnosis'].median():.1f}")
    report.append("")
    
    # Demographics comparison
    report.append("SEX DISTRIBUTION:")
    for cohort_name, df in [("T1D Filtered", t1d_df), ("T1D Hypoglycemia", t1d_hypo_df)]:
        sex_dist = df['sex'].value_counts()
        total = len(df)
        report.append(f"  {cohort_name}:")
        for sex, count in sex_dist.items():
            pct = (count / total) * 100 if total > 0 else 0
            report.append(f"    {sex}: {count} ({pct:.1f}%)")
    report.append("")
    
    # Medication comparison
    report.append("MEDICATION USAGE COMPARISON:")
    for drug_class in T1D_DRUG_CLASSES.keys():
        report.append(f"  {drug_class}:")
        for cohort_name, df in [("T1D Filtered", t1d_df), ("T1D Hypoglycemia", t1d_hypo_df)]:
            count = df[drug_class].sum()
            pct = (count / len(df)) * 100 if len(df) > 0 else 0
            report.append(f"    {cohort_name}: {count} ({pct:.1f}%)")
    report.append("")
    
    # Condition comparison
    report.append("CONDITION PREVALENCE COMPARISON:")
    for condition in CONDITIONS.keys():
        report.append(f"  {condition}:")
        for cohort_name, df in [("T1D Filtered", t1d_df), ("T1D Hypoglycemia", t1d_hypo_df)]:
            count = df[condition].sum()
            pct = (count / len(df)) * 100 if len(df) > 0 else 0
            report.append(f"    {cohort_name}: {count} ({pct:.1f}%)")
    
    report_text = "\n".join(report)
    
    # Print to console
    print(report_text)
    
    # Save to S3
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_key = f'T1D_Tosur/data/preprocessing_reports/comparison_report_{timestamp}.txt'
    
    s3_client.put_object(
        Bucket=bucket,
        Key=report_key,
        Body=report_text.encode('utf-8')
    )
    
    print(f"\n📄 Comparison report saved to s3://{bucket}/{report_key}")

def main():
    """Main function to orchestrate T1D data preprocessing"""
    print("\n" + "🏥 " * 20)
    print("T1D PATIENT DATA PREPROCESSING PIPELINE")
    print("🏥 " * 20)
    
    start_time = datetime.now()
    
    # Initialize S3 client
    print("\n🔗 Initializing S3 connection...")
    s3 = boto3.client('s3')
    
    # Process T1D Filtered Cohort
    t1d_filtered_df = process_cohort(
        cohort_name="T1D Filtered Cohort",
        source_prefix='T1D_Tosur/data/T1D_OMOP_variables/',
        output_suffix='filtered',
        s3_client=s3
    )
    
    # Process T1D Hypoglycemia Cohort
    t1d_hypo_df = process_cohort(
        cohort_name="T1D Hypoglycemia Cohort",
        source_prefix='T1D_Tosur/data/T1D_Hypoglycemia_OMOP_variables/',
        output_suffix='hypoglycemia',
        s3_client=s3
    )
    
    # Generate comparison report if both cohorts processed successfully
    if t1d_filtered_df is not None and t1d_hypo_df is not None:
        generate_comparison_report(t1d_filtered_df, t1d_hypo_df, s3)
    
    # Calculate execution time
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n⏱️  Total execution time: {duration}")
    print("\n✨ Preprocessing pipeline completed successfully! ✨")
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL OUTPUT FILES:")
    print("=" * 60)
    print("📁 T1D Filtered Features:")
    print("   s3://dsw-sagemaker-dev-s3/T1D_Tosur/data/T1D_OMOP_variables/T1D_patient_features_filtered.csv")
    print("\n📁 T1D Hypoglycemia Features:")
    print("   s3://dsw-sagemaker-dev-s3/T1D_Tosur/data/T1D_Hypoglycemia_OMOP_variables/T1D_patient_features_hypoglycemia.csv")
    print("\n📁 Comparison Report:")
    print("   s3://dsw-sagemaker-dev-s3/T1D_Tosur/data/preprocessing_reports/")

if __name__ == "__main__":
    main()