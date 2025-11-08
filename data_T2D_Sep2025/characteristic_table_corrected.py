import pandas as pd
import boto3
import os
import gc
from io import StringIO
import numpy as np
import re
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
S3_BUCKET = 'dsw-sagemaker-dev-s3'
S3_PREFIX = 'OMOP_data_extractions/T2D_Tosur_sep2025/'
BASELINE_FILE = '/home/sagemaker-user/T2D/data_T2D_Sep2025/T2D_Final_Sep2025.csv'
OUTPUT_DIR = '/home/sagemaker-user/T2D/data_T2D_Sep2025/enhanced_OMOP_data/'

# Drug classes configuration
DRUG_CLASSES = {
    'Insulins': [
        'insulin aspart', 'insulin degludec', 'insulin detemir', 'insulin glargine',
        'insulin glulisine', 'insulin human', 'insulin regular', 'insulin nph',
        'insulin isophane', 'insulin lispro', 'insulin lispro protamine',
        'inhaled human insulin', 'technosphere insulin'
    ],
    'Biguanide': ['metformin'],
    'GLP1_agonists': [
        'albiglutide', 'dulaglutide', 'exenatide', 'liraglutide', 'lixisenatide',
        'semaglutide', 'tirzepatide'
    ],
    'DPP4_inhibitors': [
        'alogliptin', 'anagliptin', 'evogliptin', 'gemigliptin', 'linagliptin',
        'saxagliptin', 'sitagliptin', 'teneligliptin', 'vildagliptin'
    ],
    'SGLT2_inhibitors': [
        'canagliflozin', 'dapagliflozin', 'empagliflozin', 'ertugliflozin',
        'ipragliflozin', 'luseogliflozin', 'remogliflozin', 'sotagliflozin', 'tofogliflozin'
    ],
    'Sulfonylureas': [
        'acetohexamide', 'chlorpropamide', 'glimepiride', 'glipizide',
        'glyburide', 'glibenclamide', 'tolazamide', 'tolbutamide'
    ],
    'Meglitinides': ['nateglinide', 'repaglinide'],
    'Thiazolidinediones': ['lobeglitazone', 'pioglitazone', 'rosiglitazone'],
    'Alpha_glucosidase_inhibitors': ['acarbose', 'miglitol', 'voglibose'],
    'Amylin_analogue': ['pramlintide']
}

# Condition codes configuration
CONDITION_CODES = {
    'DKA': {
        'ICD9': ['250.11', '250.13', '250.10', '250.12'],
        'ICD10': ['E10.10', 'E10.11', 'E11.10', 'E11.11']
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
        'ICD9': ['791.0'],
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


def create_race_ethnicity_composite(df):
    """Create composite race/ethnicity groups - FIXED VERSION"""
    print("\n" + "="*60)
    print("CREATING COMPOSITE RACE/ETHNICITY GROUPS")
    print("="*60)
    
    def categorize_race_ethnicity(row):
        # Handle missing values properly
        ethnicity_val = row.get('ethnicity', '')
        race_val = row.get('race', '')
        
        # Convert to string and lowercase, handling NaN
        ethnicity = str(ethnicity_val).lower() if pd.notna(ethnicity_val) else ''
        race = str(race_val).lower() if pd.notna(race_val) else ''
        
        # Hispanic (of any race) - but exclude "Not Hispanic" or "Non-Hispanic"
        # Check for positive Hispanic indicators
        if ethnicity:
            # Positive matches for Hispanic
            if any(term in ethnicity for term in ['hispanic or latino', 'mexican', 'puerto rican', 'cuban', 'central american', 'south american']):
                return 'Hispanic (any race)'
            # Explicit negative matches
            if any(term in ethnicity for term in ['not hispanic', 'non-hispanic', 'non hispanic']):
                # Continue to check race
                pass
            # Simple 'hispanic' or 'latino' that isn't preceded by 'not' or 'non'
            elif 'hispanic' in ethnicity or 'latino' in ethnicity:
                return 'Hispanic (any race)'
        
        # Non-Hispanic Black
        if 'black' in race or 'african' in race:
            return 'Non-Hispanic Black'
        
        # Non-Hispanic White
        if 'white' in race or 'caucasian' in race:
            return 'Non-Hispanic White'
        
        # Other (Asian, Pacific Islander, multiracial, etc)
        if any(term in race for term in ['asian', 'pacific', 'islander', 'multiracial', 'mixed', 'other']):
            return 'Other'
        
        # Unable to obtain
        return 'Unable to obtain'
    
    df['race_ethnicity_composite'] = df.apply(categorize_race_ethnicity, axis=1)
    
    # Print distribution
    composite_counts = df['race_ethnicity_composite'].value_counts()
    print("\nComposite Race/Ethnicity Distribution:")
    for group, count in composite_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {group}: {count:,} ({pct:.1f}%)")
    
    return df


def read_s3_csv(bucket, key, chunksize=None):
    """Read CSV from S3, optionally in chunks"""
    s3 = boto3.client('s3')
    
    if chunksize:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj['Body'], chunksize=chunksize)
    else:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj['Body'])


def load_baseline_data():
    """Load the baseline T2D CSV file"""
    print("\n" + "="*60)
    print("LOADING BASELINE DATA")
    print("="*60)
    
    baseline_df = pd.read_csv(BASELINE_FILE)
    print(f"Loaded {len(baseline_df):,} patients from baseline file")
    print(f"Columns: {', '.join(baseline_df.columns.tolist())}")
    
    # Convert DiagnosisDate to datetime
    baseline_df['DiagnosisDate'] = pd.to_datetime(baseline_df['DiagnosisDate'], errors='coerce')
    
    return baseline_df


def extract_demographics(baseline_df):
    """Extract demographics from OMOP person.csv and merge with baseline"""
    print("\n" + "="*60)
    print("EXTRACTING DEMOGRAPHICS")
    print("="*60)
    
    try:
        # Read demographics file
        demo_df = read_s3_csv(S3_BUCKET, f'{S3_PREFIX}demographics/person.csv')
        print(f"Loaded {len(demo_df):,} patients from OMOP demographics")
        
        # Standardize column names for merging
        demo_df = demo_df.rename(columns={'PERSON_ID': 'person_id'})
        
        # Select relevant demographic columns
        demo_cols = ['person_id']
        if 'GENDER_SOURCE_VALUE' in demo_df.columns:
            demo_cols.append('GENDER_SOURCE_VALUE')
            demo_df = demo_df.rename(columns={'GENDER_SOURCE_VALUE': 'gender'})
        elif 'GENDER_CONCEPT_ID' in demo_df.columns:
            demo_cols.append('GENDER_CONCEPT_ID')
            demo_df = demo_df.rename(columns={'GENDER_CONCEPT_ID': 'gender'})
            
        if 'RACE_SOURCE_VALUE' in demo_df.columns:
            demo_cols.append('RACE_SOURCE_VALUE')
            demo_df = demo_df.rename(columns={'RACE_SOURCE_VALUE': 'race'})
        elif 'RACE_CONCEPT_ID' in demo_df.columns:
            demo_cols.append('RACE_CONCEPT_ID')
            demo_df = demo_df.rename(columns={'RACE_CONCEPT_ID': 'race'})
            
        if 'ETHNICITY_SOURCE_VALUE' in demo_df.columns:
            demo_cols.append('ETHNICITY_SOURCE_VALUE')
            demo_df = demo_df.rename(columns={'ETHNICITY_SOURCE_VALUE': 'ethnicity'})
        elif 'ETHNICITY_CONCEPT_ID' in demo_df.columns:
            demo_cols.append('ETHNICITY_CONCEPT_ID')
            demo_df = demo_df.rename(columns={'ETHNICITY_CONCEPT_ID': 'ethnicity'})
        
        # Update demo_cols to reflect renamed columns
        demo_cols = [col for col in ['person_id', 'gender', 'race', 'ethnicity'] if col in demo_df.columns]
        demo_df = demo_df[demo_cols]
        
        # Merge with baseline
        enhanced_df = baseline_df.merge(demo_df, on='person_id', how='left')
        print(f"Merged demographics for {enhanced_df['gender'].notna().sum():,} patients")
        
        del demo_df
        gc.collect()
        
        return enhanced_df
        
    except Exception as e:
        print(f"Error extracting demographics: {e}")
        return baseline_df


def extract_medications_features(baseline_df, chunk_size=100000):
    """Extract medication features with updated time windows - TRACKS PATIENTS WITH DATA"""
    print("\n" + "="*60)
    print("EXTRACTING MEDICATION FEATURES")
    print("="*60)
    
    try:
        # Find medication file
        s3 = boto3.client('s3')
        medication_file = None
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=f'{S3_PREFIX}medications/')
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].endswith('.csv'):
                    medication_file = obj['Key']
                    break
        
        if not medication_file:
            print("No medication file found, skipping...")
            return baseline_df, {}, {}
        
        # Initialize feature dictionaries for each drug class and time window
        med_features = {}
        for drug_class in DRUG_CLASSES.keys():
            med_features[f'{drug_class}_at_diagnosis'] = set()
            med_features[f'{drug_class}_2year'] = {}  # Store dates to find closest
            med_features[f'{drug_class}_5year'] = {}  # Store dates to find closest
        
        # NEW: Track patients with ANY medication data in each time window
        patients_with_data = {
            'at_diagnosis': set(),
            '2year': set(),
            '5year': set()
        }
        
        # Get patient diagnosis dates
        patient_diagnosis_dates = dict(zip(baseline_df['person_id'], baseline_df['DiagnosisDate']))
        
        print(f"Processing medications in chunks...")
        chunks = read_s3_csv(S3_BUCKET, medication_file, chunksize=chunk_size)
        
        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}...", end='\r')
            
            chunk = chunk.rename(columns={'PERSON_ID': 'person_id'})
            chunk = chunk[chunk['person_id'].isin(baseline_df['person_id'])]
            
            if chunk.empty:
                continue
            
            date_col = None
            for col in ['DRUG_EXPOSURE_START_DATE', 'Drug_Exposure_Start_Date']:
                if col in chunk.columns:
                    date_col = col
                    break
            
            if not date_col or 'DRUG_SOURCE_VALUE' not in chunk.columns:
                continue
            
            chunk[date_col] = pd.to_datetime(chunk[date_col], errors='coerce')
            
            # NEW: Track all patients with medication data in each window
            for _, row in chunk.iterrows():
                person_id = row['person_id']
                drug_date = row[date_col]
                
                if pd.isna(drug_date) or person_id not in patient_diagnosis_dates:
                    continue
                
                diagnosis_date = patient_diagnosis_dates[person_id]
                if pd.isna(diagnosis_date):
                    continue
                
                days_diff = (drug_date - diagnosis_date).days
                
                # Track patients with data in each window
                if -180 <= days_diff <= 180:
                    patients_with_data['at_diagnosis'].add(person_id)
                if 730 <= days_diff <= 1095:
                    patients_with_data['2year'].add(person_id)
                if 1460 <= days_diff <= 2190:
                    patients_with_data['5year'].add(person_id)
            
            # Process each drug class
            for drug_class, drugs in DRUG_CLASSES.items():
                class_mask = pd.Series([False] * len(chunk), index=chunk.index)
                
                for drug in drugs:
                    pattern = r'(?i)\b' + re.escape(drug) + r'\b'
                    drug_mask = chunk['DRUG_SOURCE_VALUE'].str.contains(pattern, regex=True, na=False, case=False)
                    class_mask = class_mask | drug_mask
                
                if class_mask.any():
                    matched_rows = chunk[class_mask].copy()
                    
                    for _, row in matched_rows.iterrows():
                        person_id = row['person_id']
                        drug_date = row[date_col]
                        
                        if pd.isna(drug_date) or person_id not in patient_diagnosis_dates:
                            continue
                        
                        diagnosis_date = patient_diagnosis_dates[person_id]
                        if pd.isna(diagnosis_date):
                            continue
                        
                        days_diff = (drug_date - diagnosis_date).days
                        
                        # Within 6 months of diagnosis (±180 days)
                        if -180 <= days_diff <= 180:
                            med_features[f'{drug_class}_at_diagnosis'].add(person_id)
                        
                        # 2 years: between 2-3 years (730-1095 days), closest to 730
                        if 730 <= days_diff <= 1095:
                            key_2yr = f'{drug_class}_2year'
                            if person_id not in med_features[key_2yr]:
                                med_features[key_2yr][person_id] = days_diff
                            else:
                                # Keep the date closest to 730 days
                                if abs(days_diff - 730) < abs(med_features[key_2yr][person_id] - 730):
                                    med_features[key_2yr][person_id] = days_diff
                        
                        # 5 years: between 4-6 years (1460-2190 days), closest to 1825
                        if 1460 <= days_diff <= 2190:
                            key_5yr = f'{drug_class}_5year'
                            if person_id not in med_features[key_5yr]:
                                med_features[key_5yr][person_id] = days_diff
                            else:
                                # Keep the date closest to 1825 days
                                if abs(days_diff - 1825) < abs(med_features[key_5yr][person_id] - 1825):
                                    med_features[key_5yr][person_id] = days_diff
        
        print(f"\n\nCreating medication feature columns...")
        print(f"Patients with medication data:")
        print(f"  At diagnosis: {len(patients_with_data['at_diagnosis']):,}")
        print(f"  2 years: {len(patients_with_data['2year']):,}")
        print(f"  5 years: {len(patients_with_data['5year']):,}")
        
        # Add medication features to baseline_df
        for drug_class in DRUG_CLASSES.keys():
            # At diagnosis
            baseline_df[f'{drug_class}_at_diagnosis'] = baseline_df['person_id'].apply(
                lambda x: 1 if x in med_features[f'{drug_class}_at_diagnosis'] else 0
            )
            print(f"  {drug_class}_at_diagnosis: {len(med_features[f'{drug_class}_at_diagnosis']):,} patients")
            
            # 2 years
            baseline_df[f'{drug_class}_2year'] = baseline_df['person_id'].apply(
                lambda x: 1 if x in med_features[f'{drug_class}_2year'] else 0
            )
            print(f"  {drug_class}_2year: {len(med_features[f'{drug_class}_2year']):,} patients")
            
            # 5 years
            baseline_df[f'{drug_class}_5year'] = baseline_df['person_id'].apply(
                lambda x: 1 if x in med_features[f'{drug_class}_5year'] else 0
            )
            print(f"  {drug_class}_5year: {len(med_features[f'{drug_class}_5year']):,} patients")
        
        # Convert sets to counts for return
        med_data_counts = {
            'at_diagnosis': len(patients_with_data['at_diagnosis']),
            '2year': len(patients_with_data['2year']),
            '5year': len(patients_with_data['5year'])
        }
        
        gc.collect()
        return baseline_df, med_data_counts, patients_with_data
        
    except Exception as e:
        print(f"Error extracting medication features: {e}")
        import traceback
        traceback.print_exc()
        return baseline_df, {}, {}


def extract_icd_features(baseline_df, chunk_size=100000):
    """Extract ICD code/comorbidity features with time windows - TRACKS PATIENTS WITH DATA"""
    print("\n" + "="*60)
    print("EXTRACTING ICD CODE FEATURES")
    print("="*60)
    
    try:
        # Initialize feature dictionaries for each condition and time window
        icd_features = {}
        for condition in CONDITION_CODES.keys():
            icd_features[f'{condition}_at_diagnosis'] = set()
            icd_features[f'{condition}_2year'] = {}
            icd_features[f'{condition}_5year'] = {}
        
        # NEW: Track patients with ANY ICD data in each time window
        patients_with_data = {
            'at_diagnosis': set(),
            '2year': set(),
            '5year': set()
        }
        
        # Get patient diagnosis dates
        patient_diagnosis_dates = dict(zip(baseline_df['person_id'], baseline_df['DiagnosisDate']))
        
        print(f"Processing ICD codes in chunks...")
        chunks = read_s3_csv(S3_BUCKET, f'{S3_PREFIX}icd_codes/condition_occurrence.csv', 
                           chunksize=chunk_size)
        
        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}...", end='\r')
            
            # Standardize column names
            chunk = chunk.rename(columns={'PERSON_ID': 'person_id'})
            
            # Filter for patients in baseline
            chunk = chunk[chunk['person_id'].isin(baseline_df['person_id'])]
            
            if chunk.empty:
                continue
            
            # Find ICD code column
            icd_col = None
            for col in ['CONDITION_SOURCE_VALUE', 'Condition_Source_Value']:
                if col in chunk.columns:
                    icd_col = col
                    break
            
            # Find date column
            date_col = None
            for col in ['CONDITION_START_DATE', 'Condition_Start_Date']:
                if col in chunk.columns:
                    date_col = col
                    break
            
            if not icd_col or not date_col:
                continue
            
            # Extract ICD codes from pipe-separated format
            chunk['extracted_icd'] = chunk[icd_col].apply(
                lambda x: x.split('|')[1].strip() if pd.notna(x) and '|' in x else x if pd.notna(x) else None
            )
            
            chunk[date_col] = pd.to_datetime(chunk[date_col], errors='coerce')
            
            # NEW: Track all patients with ICD data in each window
            for _, row in chunk.iterrows():
                person_id = row['person_id']
                condition_date = row[date_col]
                
                if pd.isna(condition_date) or person_id not in patient_diagnosis_dates:
                    continue
                
                diagnosis_date = patient_diagnosis_dates[person_id]
                if pd.isna(diagnosis_date):
                    continue
                
                days_diff = (condition_date - diagnosis_date).days
                
                # Track patients with data in each window
                if -180 <= days_diff <= 180:
                    patients_with_data['at_diagnosis'].add(person_id)
                if 730 <= days_diff <= 1095:
                    patients_with_data['2year'].add(person_id)
                if 1460 <= days_diff <= 2190:
                    patients_with_data['5year'].add(person_id)
            
            # Process each condition
            for condition, codes in CONDITION_CODES.items():
                all_codes = codes.get('ICD9', []) + codes.get('ICD10', [])
                
                for code in all_codes:
                    # Match exact or prefix
                    mask = (chunk['extracted_icd'] == code) | (chunk['extracted_icd'].str.startswith(code, na=False))
                    matched_rows = chunk[mask].copy()
                    
                    if matched_rows.empty:
                        continue
                    
                    # Calculate time difference from diagnosis
                    for _, row in matched_rows.iterrows():
                        person_id = row['person_id']
                        condition_date = row[date_col]
                        
                        if pd.isna(condition_date) or person_id not in patient_diagnosis_dates:
                            continue
                        
                        diagnosis_date = patient_diagnosis_dates[person_id]
                        if pd.isna(diagnosis_date):
                            continue
                        
                        days_diff = (condition_date - diagnosis_date).days
                        
                        # Within 6 months of diagnosis (±180 days)
                        if -180 <= days_diff <= 180:
                            icd_features[f'{condition}_at_diagnosis'].add(person_id)

                        # 2 years: between 2-3 years (730-1095 days), closest to 730
                        if 730 <= days_diff <= 1095:
                            key_2yr = f'{condition}_2year'
                            if person_id not in icd_features[key_2yr]:
                                icd_features[key_2yr][person_id] = days_diff
                            else:
                                if abs(days_diff - 730) < abs(icd_features[key_2yr][person_id] - 730):
                                    icd_features[key_2yr][person_id] = days_diff

                        # 5 years: between 4-6 years (1460-2190 days), closest to 1825
                        if 1460 <= days_diff <= 2190:
                            key_5yr = f'{condition}_5year'
                            if person_id not in icd_features[key_5yr]:
                                icd_features[key_5yr][person_id] = days_diff
                            else:
                                if abs(days_diff - 1825) < abs(icd_features[key_5yr][person_id] - 1825):
                                    icd_features[key_5yr][person_id] = days_diff
        
        print(f"\n\nCreating ICD feature columns...")
        print(f"Patients with ICD data:")
        print(f"  At diagnosis: {len(patients_with_data['at_diagnosis']):,}")
        print(f"  2 years: {len(patients_with_data['2year']):,}")
        print(f"  5 years: {len(patients_with_data['5year']):,}")
        
        # Add ICD features to baseline_df
        for condition in CONDITION_CODES.keys():
            # At diagnosis
            baseline_df[f'{condition}_at_diagnosis'] = baseline_df['person_id'].apply(
                lambda x: 1 if x in icd_features[f'{condition}_at_diagnosis'] else 0
            )
            print(f"  {condition}_at_diagnosis: {len(icd_features[f'{condition}_at_diagnosis']):,} patients")
            
            # 2 years
            baseline_df[f'{condition}_2year'] = baseline_df['person_id'].apply(
                lambda x: 1 if x in icd_features[f'{condition}_2year'] else 0
            )
            print(f"  {condition}_2year: {len(icd_features[f'{condition}_2year']):,} patients")
            
            # 5 years
            baseline_df[f'{condition}_5year'] = baseline_df['person_id'].apply(
                lambda x: 1 if x in icd_features[f'{condition}_5year'] else 0
            )
            print(f"  {condition}_5year: {len(icd_features[f'{condition}_5year']):,} patients")
        
        # Convert sets to counts for return
        icd_data_counts = {
            'at_diagnosis': len(patients_with_data['at_diagnosis']),
            '2year': len(patients_with_data['2year']),
            '5year': len(patients_with_data['5year'])
        }
        
        gc.collect()
        return baseline_df, icd_data_counts, patients_with_data
        
    except Exception as e:
        print(f"Error extracting ICD features: {e}")
        import traceback
        traceback.print_exc()
        return baseline_df, {}, {}


def save_enhanced_data(enhanced_df, med_data_counts, icd_data_counts, med_patients_with_data, icd_patients_with_data):
    """Save the enhanced dataframe to CSV along with metadata"""
    print("\n" + "="*60)
    print("SAVING ENHANCED DATA")
    print("="*60)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_file = os.path.join(OUTPUT_DIR, 'T2D_Enhanced_OMOP.csv')
    enhanced_df.to_csv(output_file, index=False)
    
    print(f"Saved enhanced data to: {output_file}")
    print(f"Total patients: {len(enhanced_df):,}")
    print(f"Total features: {len(enhanced_df.columns):,}")
    
    # NEW: Save metadata about patients with data in each window
    # Combine medication and ICD data tracking
    combined_patients_with_data = {
        'at_diagnosis': med_patients_with_data['at_diagnosis'] | icd_patients_with_data['at_diagnosis'],
        '2year': med_patients_with_data['2year'] | icd_patients_with_data['2year'],
        '5year': med_patients_with_data['5year'] | icd_patients_with_data['5year']
    }
    
    metadata = {
        'total_cohort': len(enhanced_df),
        'patients_with_data_at_diagnosis': len(combined_patients_with_data['at_diagnosis']),
        'patients_with_data_2year': len(combined_patients_with_data['2year']),
        'patients_with_data_5year': len(combined_patients_with_data['5year'])
    }
    
    metadata_file = os.path.join(OUTPUT_DIR, 'cohort_metadata.csv')
    pd.DataFrame([metadata]).to_csv(metadata_file, index=False)
    print(f"\nSaved cohort metadata to: {metadata_file}")
    
    return combined_patients_with_data


def clean_lab_values(df):
    """Clean and convert lab measurement columns to numeric"""
    print("\n" + "="*60)
    print("CLEANING LAB VALUES")
    print("="*60)
    
    df_clean = df.copy()
    
    # Clean HbA1c values
    if 'HGBAtDiagnosisResultTXT' in df_clean.columns:
        print("Cleaning HbA1c values...")
        df_clean['HbA1c_numeric'] = df_clean['HGBAtDiagnosisResultTXT'].astype(str)
        
        # Handle >14.0 cases - replace with 14.0
        df_clean['HbA1c_numeric'] = df_clean['HbA1c_numeric'].str.replace('>', '', regex=False)
        df_clean['HbA1c_numeric'] = df_clean['HbA1c_numeric'].str.replace('<', '', regex=False)
        df_clean['HbA1c_numeric'] = pd.to_numeric(df_clean['HbA1c_numeric'], errors='coerce')
        
        valid_hba1c = df_clean['HbA1c_numeric'].notna().sum()
        print(f"  Valid HbA1c values: {valid_hba1c:,}")
    
    # Clean Glucose values
    if 'GlucoseAtDiagnosisResultTXT' in df_clean.columns:
        print("Cleaning Glucose values...")
        df_clean['Glucose_numeric'] = df_clean['GlucoseAtDiagnosisResultTXT'].astype(str)
        
        # Remove text like "CANCELLED: ORDER DUPLICATION"
        df_clean['Glucose_numeric'] = df_clean['Glucose_numeric'].apply(
            lambda x: x if str(x).replace('.', '').replace('-', '').isdigit() else np.nan
        )
        df_clean['Glucose_numeric'] = pd.to_numeric(df_clean['Glucose_numeric'], errors='coerce')
        
        valid_glucose = df_clean['Glucose_numeric'].notna().sum()
        print(f"  Valid Glucose values: {valid_glucose:,}")
    
    # Clean C-peptide values
    if 'CpeptideAtDiagnosisResultTXT' in df_clean.columns:
        print("Cleaning C-peptide values...")
        df_clean['Cpeptide_numeric'] = pd.to_numeric(
            df_clean['CpeptideAtDiagnosisResultTXT'], errors='coerce'
        )
        
        valid_cpeptide = df_clean['Cpeptide_numeric'].notna().sum()
        print(f"  Valid C-peptide values: {valid_cpeptide:,}")
    
    return df_clean


def process_antibody_flags(df):
    """Process antibody flags into categorical variables"""
    print("\nProcessing antibody flags...")
    
    antibody_cols = ['GADPostivePatientFlg', 'icaPostivePatientFlg', 
                     'InsulinPostivePatientFlg', 'ZincPostivePatientFlg']
    
    for col in antibody_cols:
        if col in df.columns:
            # Create clean version: Positive, Negative, Unknown
            new_col = col.replace('Flg', '_status')
            df[new_col] = df[col].apply(
                lambda x: 'Positive' if x == 'Y' else ('Negative' if x == 'N' else 'Unknown')
            )
            print(f"  {col}: {(df[new_col] == 'Positive').sum()} Positive, "
                  f"{(df[new_col] == 'Negative').sum()} Negative, "
                  f"{(df[new_col] == 'Unknown').sum()} Unknown")
    
    return df


def create_medication_visualization(df, patients_with_data):
    """Create medication visualization with NORMALIZED percentages"""
    print("\nCreating medication visualization with normalized percentages...")
    
    med_cols = [col for col in df.columns if any(drug in col for drug in 
                ['Insulins', 'Biguanide', 'GLP1', 'DPP4', 'SGLT2', 'Sulfonylureas', 
                 'Meglitinides', 'Thiazolidinediones', 'Alpha_glucosidase', 'Amylin'])]
    
    if not med_cols:
        return
    
    # Group by medication class and time frame
    med_data = {}
    for col in med_cols:
        for drug_class in ['Insulins', 'Biguanide', 'GLP1_agonists', 'DPP4_inhibitors', 
                          'SGLT2_inhibitors', 'Sulfonylureas', 'Meglitinides', 
                          'Thiazolidinediones', 'Alpha_glucosidase_inhibitors', 'Amylin_analogue']:
            if drug_class in col:
                if drug_class not in med_data:
                    med_data[drug_class] = {'at_diagnosis': 0, '2year': 0, '5year': 0}
                
                if '_at_diagnosis' in col:
                    med_data[drug_class]['at_diagnosis'] = df[col].sum()
                elif '_2year' in col:
                    med_data[drug_class]['2year'] = df[col].sum()
                elif '_5year' in col:
                    med_data[drug_class]['5year'] = df[col].sum()
    
    drug_names = list(med_data.keys())
    x = np.arange(len(drug_names))
    width = 0.25
    
    at_diag = [med_data[drug]['at_diagnosis'] for drug in drug_names]
    two_year = [med_data[drug]['2year'] for drug in drug_names]
    five_year = [med_data[drug]['5year'] for drug in drug_names]
    
    # NEW: Use normalized denominators
    n_at_diag = len(patients_with_data['at_diagnosis'])
    n_2year = len(patients_with_data['2year'])
    n_5year = len(patients_with_data['5year'])
    
    # Single plot with counts AND NORMALIZED percentages
    fig, ax = plt.subplots(figsize=(16, 8))
    bars1 = ax.bar(x - width, at_diag, width, label=f'At Diagnosis (N={n_at_diag:,})', 
                   color='#3498db', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x, two_year, width, label=f'2 Years (N={n_2year:,})', 
                   color='#e67e22', edgecolor='black', alpha=0.8)
    bars3 = ax.bar(x + width, five_year, width, label=f'5 Years (N={n_5year:,})', 
                   color='#9b59b6', edgecolor='black', alpha=0.8)
    
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.set_title('Medication Class Usage by Time Frame (Normalized)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace('_', ' ') for name in drug_names], 
                       rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add count and NORMALIZED percentage labels
    for i, (count, bar) in enumerate(zip(at_diag, bars1)):
        pct = (count/n_at_diag)*100 if n_at_diag > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2, count, 
               f'{int(count):,}\n({pct:.1f}%)', 
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    for i, (count, bar) in enumerate(zip(two_year, bars2)):
        pct = (count/n_2year)*100 if n_2year > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2, count, 
               f'{int(count):,}\n({pct:.1f}%)', 
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    for i, (count, bar) in enumerate(zip(five_year, bars3)):
        pct = (count/n_5year)*100 if n_5year > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2, count, 
               f'{int(count):,}\n({pct:.1f}%)', 
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Medication_Classes_Normalized.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    print("  Medication visualization saved!")


def create_comorbidity_visualization(df, patients_with_data):
    """Create comorbidity visualization with NORMALIZED percentages"""
    print("\nCreating comorbidity visualization with normalized percentages...")
    
    condition_cols = [col for col in df.columns if any(cond in col for cond in 
                     ['DKA', 'Ketosis', 'Dyslipidemia', 'Hypertension', 'Diabetic_Retinopathy',
                      'Microalbuminuria', 'Neuropathy', 'Hypoglycemia'])]
    
    if not condition_cols:
        return
    
    # Group by condition and time frame
    condition_data = {}
    for col in condition_cols:
        for cond in ['DKA', 'Ketosis', 'Dyslipidemia', 'Hypertension', 'Diabetic_Retinopathy',
                     'Microalbuminuria', 'Neuropathy', 'Hypoglycemia']:
            if cond in col:
                if cond not in condition_data:
                    condition_data[cond] = {'at_diagnosis': 0, '2year': 0, '5year': 0}
                
                if '_at_diagnosis' in col:
                    condition_data[cond]['at_diagnosis'] = df[col].sum()
                elif '_2year' in col:
                    condition_data[cond]['2year'] = df[col].sum()
                elif '_5year' in col:
                    condition_data[cond]['5year'] = df[col].sum()
    
    cond_names = list(condition_data.keys())
    x = np.arange(len(cond_names))
    width = 0.25
    
    at_diag = [condition_data[cond]['at_diagnosis'] for cond in cond_names]
    two_year = [condition_data[cond]['2year'] for cond in cond_names]
    five_year = [condition_data[cond]['5year'] for cond in cond_names]
    
    # NEW: Use normalized denominators
    n_at_diag = len(patients_with_data['at_diagnosis'])
    n_2year = len(patients_with_data['2year'])
    n_5year = len(patients_with_data['5year'])
    
    # Single plot with counts AND NORMALIZED percentages
    fig, ax = plt.subplots(figsize=(16, 8))
    bars1 = ax.bar(x - width, at_diag, width, label=f'At Diagnosis (N={n_at_diag:,})', 
                   color='#1abc9c', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x, two_year, width, label=f'2 Years (N={n_2year:,})', 
                   color='#f39c12', edgecolor='black', alpha=0.8)
    bars3 = ax.bar(x + width, five_year, width, label=f'5 Years (N={n_5year:,})', 
                   color='#c0392b', edgecolor='black', alpha=0.8)
    
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.set_title('Comorbidity Occurrence by Time Frame (Normalized)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace('_', ' ') for name in cond_names], 
                       rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add count and NORMALIZED percentage labels
    for i, (count, bar) in enumerate(zip(at_diag, bars1)):
        pct = (count/n_at_diag)*100 if n_at_diag > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2, count, 
               f'{int(count):,}\n({pct:.1f}%)', 
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    for i, (count, bar) in enumerate(zip(two_year, bars2)):
        pct = (count/n_2year)*100 if n_2year > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2, count, 
               f'{int(count):,}\n({pct:.1f}%)', 
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    for i, (count, bar) in enumerate(zip(five_year, bars3)):
        pct = (count/n_5year)*100 if n_5year > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2, count, 
               f'{int(count):,}\n({pct:.1f}%)', 
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Comorbidities_Normalized.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    print("  Comorbidity visualization saved!")


def create_dka_temporal_plot(df):
    """Create DKA at diagnosis percentage by year plot"""
    print("\nCreating DKA at diagnosis temporal plot...")
    
    if 'DiagnosisDate' not in df.columns or 'DKA_at_diagnosis' not in df.columns:
        print("  Missing required columns for DKA temporal plot")
        return
    
    # Extract diagnosis year
    df['DiagnosisYear'] = pd.to_datetime(df['DiagnosisDate']).dt.year
    
    # Calculate DKA rate by year
    yearly_stats = df.groupby('DiagnosisYear').agg({
        'person_id': 'count',  # Total diagnoses
        'DKA_at_diagnosis': 'sum'  # DKA cases
    }).reset_index()
    
    yearly_stats.columns = ['Year', 'Total_Diagnoses', 'DKA_Cases']
    yearly_stats['DKA_Percentage'] = (yearly_stats['DKA_Cases'] / yearly_stats['Total_Diagnoses']) * 100
    
    # Filter out years with very few diagnoses (< 10) for more stable estimates
    yearly_stats_filtered = yearly_stats[yearly_stats['Total_Diagnoses'] >= 10].copy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Percentage of DKA at diagnosis over time
    ax1.plot(yearly_stats_filtered['Year'], yearly_stats_filtered['DKA_Percentage'], 
             marker='o', linewidth=2, markersize=8, color='darkred')
    ax1.fill_between(yearly_stats_filtered['Year'], yearly_stats_filtered['DKA_Percentage'], 
                     alpha=0.3, color='salmon')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('DKA at Diagnosis (%)', fontsize=12)
    ax1.set_title('Percentage of T2D Patients with DKA at Diagnosis by Year', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add data labels for key points
    for idx in range(0, len(yearly_stats_filtered), max(1, len(yearly_stats_filtered)//10)):
        row = yearly_stats_filtered.iloc[idx]
        ax1.annotate(f'{row["DKA_Percentage"]:.1f}%', 
                    xy=(row['Year'], row['DKA_Percentage']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=8, alpha=0.7)
    
    # Plot 2: Absolute numbers (DKA cases and total diagnoses)
    ax2_twin = ax2.twinx()
    
    bar1 = ax2.bar(yearly_stats_filtered['Year'], yearly_stats_filtered['Total_Diagnoses'], 
                   alpha=0.5, color='lightblue', label='Total Diagnoses', edgecolor='black')
    line1 = ax2_twin.plot(yearly_stats_filtered['Year'], yearly_stats_filtered['DKA_Cases'], 
                         marker='s', linewidth=2, markersize=6, color='darkred', 
                         label='DKA Cases')
    
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Total New T2D Diagnoses', fontsize=12, color='blue')
    ax2_twin.set_ylabel('DKA Cases at Diagnosis', fontsize=12, color='darkred')
    ax2.set_title('T2D Diagnoses and DKA Cases by Year', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2_twin.tick_params(axis='y', labelcolor='darkred')
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'DKA_at_Diagnosis_Temporal.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print summary statistics
    print("\nDKA at Diagnosis Summary by Year:")
    print(yearly_stats_filtered.to_string(index=False))
    
    # Save to CSV
    yearly_stats_filtered.to_csv(os.path.join(OUTPUT_DIR, 'DKA_at_Diagnosis_by_Year.csv'), index=False)
    print("  DKA temporal visualization and data saved!")


def create_race_ethnicity_visualization(df):
    """Create race/ethnicity visualization with counts AND percentages on bars"""
    print("\nCreating race/ethnicity composite visualization...")
    
    if 'race_ethnicity_composite' not in df.columns:
        return
    
    # Filter out "Unable to obtain" for the graph
    df_with_data = df[df['race_ethnicity_composite'] != 'Unable to obtain'].copy()
    missing_count = (df['race_ethnicity_composite'] == 'Unable to obtain').sum()
    
    race_eth_counts = df_with_data['race_ethnicity_composite'].value_counts()
    
    # Define order for display
    desired_order = ['Hispanic (any race)', 'Non-Hispanic Black', 'Non-Hispanic White', 'Other']
    race_eth_counts = race_eth_counts.reindex([cat for cat in desired_order if cat in race_eth_counts.index])
    
    total_with_data = len(df_with_data)
    
    # Single plot with counts AND percentages
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(race_eth_counts)), race_eth_counts.values, 
                  color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'], 
                  edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(race_eth_counts)))
    ax.set_xticklabels(race_eth_counts.index, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.set_title(f'Race/Ethnicity Composite Groups (N={total_with_data:,}, Missing={missing_count:,})', 
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add count and percentage labels
    for i, (bar, val) in enumerate(zip(bars, race_eth_counts.values)):
        pct = (val/total_with_data)*100
        ax.text(i, val + max(race_eth_counts.values)*0.02, 
               f'{int(val):,}\n({pct:.1f}%)', 
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'RaceEthnicity_Composite.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    print("  Race/ethnicity composite visualization saved!")


def create_summary_statistics(df):
    """Create comprehensive summary statistics table"""
    print("\n" + "="*60)
    print("CREATING SUMMARY STATISTICS TABLE")
    print("="*60)
    
    summary_stats = []
    
    # Demographics
    print("\nProcessing demographics...")
    if 'AgeatDiagnosis' in df.columns:
        age_data = df['AgeatDiagnosis'].dropna()
        summary_stats.append({
            'Variable': 'Age at Diagnosis (years)',
            'N': len(age_data),
            'Mean (SD)': f"{age_data.mean():.1f} ({age_data.std():.1f})",
            'Median (IQR)': f"{age_data.median():.1f} ({age_data.quantile(0.25):.1f}-{age_data.quantile(0.75):.1f})",
            'Min-Max': f"{age_data.min():.1f}-{age_data.max():.1f}"
        })
    
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        for gender, count in gender_counts.items():
            pct = (count / len(df)) * 100
            summary_stats.append({
                'Variable': f'Gender: {gender}',
                'N': count,
                'Mean (SD)': f"{pct:.1f}%",
                'Median (IQR)': '-',
                'Min-Max': '-'
            })
    
    if 'race_ethnicity_composite' in df.columns:
        race_eth_counts = df['race_ethnicity_composite'].value_counts()
        for group, count in race_eth_counts.items():
            pct = (count / len(df)) * 100
            summary_stats.append({
                'Variable': f'Race/Ethnicity: {group}',
                'N': count,
                'Mean (SD)': f"{pct:.1f}%",
                'Median (IQR)': '-',
                'Min-Max': '-'
            })
    
    # Lab measurements
    print("Processing lab measurements...")
    if 'HbA1c_numeric' in df.columns:
        hba1c_data = df['HbA1c_numeric'].dropna()
        summary_stats.append({
            'Variable': 'HbA1c at Diagnosis (%)',
            'N': len(hba1c_data),
            'Mean (SD)': f"{hba1c_data.mean():.1f} ({hba1c_data.std():.1f})",
            'Median (IQR)': f"{hba1c_data.median():.1f} ({hba1c_data.quantile(0.25):.1f}-{hba1c_data.quantile(0.75):.1f})",
            'Min-Max': f"{hba1c_data.min():.1f}-{hba1c_data.max():.1f}"
        })
    
    if 'Glucose_numeric' in df.columns:
        glucose_data = df['Glucose_numeric'].dropna()
        summary_stats.append({
            'Variable': 'Glucose at Diagnosis (mg/dL)',
            'N': len(glucose_data),
            'Mean (SD)': f"{glucose_data.mean():.0f} ({glucose_data.std():.0f})",
            'Median (IQR)': f"{glucose_data.median():.0f} ({glucose_data.quantile(0.25):.0f}-{glucose_data.quantile(0.75):.0f})",
            'Min-Max': f"{glucose_data.min():.0f}-{glucose_data.max():.0f}"
        })
    
    if 'Cpeptide_numeric' in df.columns:
        cpeptide_data = df['Cpeptide_numeric'].dropna()
        summary_stats.append({
            'Variable': 'C-peptide at Diagnosis (ng/mL)',
            'N': len(cpeptide_data),
            'Mean (SD)': f"{cpeptide_data.mean():.2f} ({cpeptide_data.std():.2f})",
            'Median (IQR)': f"{cpeptide_data.median():.2f} ({cpeptide_data.quantile(0.25):.2f}-{cpeptide_data.quantile(0.75):.2f})",
            'Min-Max': f"{cpeptide_data.min():.2f}-{cpeptide_data.max():.2f}"
        })
    
    # Antibodies
    print("Processing antibody status...")
    antibody_status_cols = [col for col in df.columns if col.endswith('_status')]
    for col in antibody_status_cols:
        status_counts = df[col].value_counts()
        var_name = col.replace('PostivePatient_status', '').replace('_status', '')
        for status, count in status_counts.items():
            pct = (count / len(df)) * 100
            summary_stats.append({
                'Variable': f'{var_name} Antibody: {status}',
                'N': count,
                'Mean (SD)': f"{pct:.1f}%",
                'Median (IQR)': '-',
                'Min-Max': '-'
            })
    
    # Medication classes - overall
    print("Processing medication classes...")
    med_cols = [col for col in df.columns if any(drug in col for drug in 
                ['Insulins', 'Biguanide', 'GLP1', 'DPP4', 'SGLT2', 'Sulfonylureas', 
                 'Meglitinides', 'Thiazolidinediones', 'Alpha_glucosidase', 'Amylin'])]
    
    # Group by medication class
    med_classes = {}
    for col in med_cols:
        for drug_class in ['Insulins', 'Biguanide', 'GLP1_agonists', 'DPP4_inhibitors', 
                          'SGLT2_inhibitors', 'Sulfonylureas', 'Meglitinides', 
                          'Thiazolidinediones', 'Alpha_glucosidase_inhibitors', 'Amylin_analogue']:
            if drug_class in col:
                if drug_class not in med_classes:
                    med_classes[drug_class] = []
                med_classes[drug_class].append(col)
    
    for drug_class, cols in med_classes.items():
        # Any time window
        any_use = df[cols].max(axis=1)
        count = any_use.sum()
        pct = (count / len(df)) * 100
        summary_stats.append({
            'Variable': f'{drug_class} (any time)',
            'N': int(count),
            'Mean (SD)': f"{pct:.1f}%",
            'Median (IQR)': '-',
            'Min-Max': '-'
        })
    
    # Comorbidities - overall
    print("Processing comorbidities...")
    condition_cols = [col for col in df.columns if any(cond in col for cond in 
                     ['DKA', 'Ketosis', 'Dyslipidemia', 'Hypertension', 'Diabetic_Retinopathy',
                      'Microalbuminuria', 'Neuropathy', 'Hypoglycemia'])]
    
    # Group by condition
    conditions = {}
    for col in condition_cols:
        for cond in ['DKA', 'Ketosis', 'Dyslipidemia', 'Hypertension', 'Diabetic_Retinopathy',
                     'Microalbuminuria', 'Neuropathy', 'Hypoglycemia']:
            if cond in col:
                if cond not in conditions:
                    conditions[cond] = []
                conditions[cond].append(col)
    
    for condition, cols in conditions.items():
        # Any time window
        any_occurrence = df[cols].max(axis=1)
        count = any_occurrence.sum()
        pct = (count / len(df)) * 100
        summary_stats.append({
            'Variable': f'{condition} (any time)',
            'N': int(count),
            'Mean (SD)': f"{pct:.1f}%",
            'Median (IQR)': '-',
            'Min-Max': '-'
        })
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_stats)
    
    # Save to CSV
    summary_file = os.path.join(OUTPUT_DIR, 'Summary_Statistics_Table.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary statistics saved to: {summary_file}")
    
    return summary_df


def create_visualizations(df, patients_with_data):
    """Create comprehensive visualizations - OPTIMIZED"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # Call visualization functions with normalized data
    create_medication_visualization(df, patients_with_data)
    create_medication_percentage_visualization(df, patients_with_data)  # NEW!
    create_comorbidity_visualization(df, patients_with_data)
    create_comorbidity_percentage_visualization(df, patients_with_data)  # NEW!
    create_race_ethnicity_visualization(df)
    create_dka_temporal_plot(df)
    
    # 1. Demographics plots
    print("\nCreating demographics plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Demographics Distribution', fontsize=16, fontweight='bold')
    
    # Age distribution
    if 'AgeatDiagnosis' in df.columns:
        age_data = df['AgeatDiagnosis'].dropna()
        n, bins, patches = axes[0, 0].hist(age_data, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].axvline(age_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {age_data.mean():.1f}')
        axes[0, 0].axvline(age_data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {age_data.median():.1f}')
        axes[0, 0].set_xlabel('Age at Diagnosis (years)', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].set_title(f'Age Distribution (N={len(age_data):,})', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        
        stats_text = f'Mean: {age_data.mean():.1f}\nMedian: {age_data.median():.1f}\nSD: {age_data.std():.1f}'
        axes[0, 0].text(0.98, 0.97, stats_text, transform=axes[0, 0].transAxes,
                       fontsize=10, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Gender distribution
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        bars = axes[0, 1].bar(range(len(gender_counts)), gender_counts.values, color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xticks(range(len(gender_counts)))
        axes[0, 1].set_xticklabels(gender_counts.index, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Count', fontsize=12)
        axes[0, 1].set_title('Gender Distribution', fontsize=12, fontweight='bold')
        for i, v in enumerate(gender_counts.values):
            pct = (v/len(df))*100
            axes[0, 1].text(i, v + max(gender_counts.values)*0.01, f'{v:,}\n({pct:.1f}%)', 
                          ha='center', va='bottom', fontsize=10)
    
    # Race distribution (top 10)
    if 'race' in df.columns:
        race_counts = df['race'].value_counts().head(10)
        bars = axes[1, 0].barh(range(len(race_counts)), race_counts.values, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1, 0].set_yticks(range(len(race_counts)))
        axes[1, 0].set_yticklabels(race_counts.index, fontsize=10)
        axes[1, 0].set_xlabel('Count', fontsize=12)
        axes[1, 0].set_title('Race Distribution (Top 10)', fontsize=12, fontweight='bold')
        axes[1, 0].invert_yaxis()
        
        for i, v in enumerate(race_counts.values):
            pct = (v/len(df))*100
            axes[1, 0].text(v + max(race_counts.values)*0.01, i, f'{v:,} ({pct:.1f}%)', 
                          ha='left', va='center', fontsize=9)
    
    # Ethnicity distribution
    if 'ethnicity' in df.columns:
        eth_counts = df['ethnicity'].value_counts()
        bars = axes[1, 1].bar(range(len(eth_counts)), eth_counts.values, color='plum', edgecolor='black', alpha=0.7)
        axes[1, 1].set_xticks(range(len(eth_counts)))
        axes[1, 1].set_xticklabels(eth_counts.index, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Count', fontsize=12)
        axes[1, 1].set_title('Ethnicity Distribution', fontsize=12, fontweight='bold')
        for i, v in enumerate(eth_counts.values):
            pct = (v/len(df))*100
            axes[1, 1].text(i, v + max(eth_counts.values)*0.01, f'{v:,}\n({pct:.1f}%)', 
                          ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Demographics_Distribution.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Lab measurements plots
    print("Creating lab measurements plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Lab Measurements at Diagnosis', fontsize=16, fontweight='bold')
    
    # HbA1c
    if 'HbA1c_numeric' in df.columns:
        hba1c_data = df['HbA1c_numeric'].dropna()
        axes[0].hist(hba1c_data, bins=30, edgecolor='black', alpha=0.7, color='indianred')
        axes[0].axvline(hba1c_data.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {hba1c_data.mean():.1f}')
        axes[0].axvline(hba1c_data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {hba1c_data.median():.1f}')
        axes[0].set_xlabel('HbA1c (%)', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title(f'HbA1c Distribution (N={len(hba1c_data):,})', fontsize=12, fontweight='bold')
        axes[0].legend()
    
    # Glucose
    if 'Glucose_numeric' in df.columns:
        glucose_data = df['Glucose_numeric'].dropna()
        axes[1].hist(glucose_data, bins=30, edgecolor='black', alpha=0.7, color='gold')
        axes[1].axvline(glucose_data.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {glucose_data.mean():.0f}')
        axes[1].axvline(glucose_data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {glucose_data.median():.0f}')
        axes[1].set_xlabel('Glucose (mg/dL)', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title(f'Glucose Distribution (N={len(glucose_data):,})', fontsize=12, fontweight='bold')
        axes[1].legend()
    
    # C-peptide
    if 'Cpeptide_numeric' in df.columns:
        cpeptide_data = df['Cpeptide_numeric'].dropna()
        axes[2].hist(cpeptide_data, bins=30, edgecolor='black', alpha=0.7, color='mediumseagreen')
        axes[2].axvline(cpeptide_data.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {cpeptide_data.mean():.2f}')
        axes[2].axvline(cpeptide_data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {cpeptide_data.median():.2f}')
        axes[2].set_xlabel('C-peptide (ng/mL)', fontsize=12)
        axes[2].set_ylabel('Count', fontsize=12)
        axes[2].set_title(f'C-peptide Distribution (N={len(cpeptide_data):,})', fontsize=12, fontweight='bold')
        axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Lab_Measurements_Distribution.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. Antibody status plots
    print("Creating antibody status plots...")
    antibody_status_cols = [col for col in df.columns if col.endswith('_status')]
    if antibody_status_cols:
        n_antibodies = len(antibody_status_cols)
        fig, axes = plt.subplots(1, n_antibodies, figsize=(5*n_antibodies, 5))
        if n_antibodies == 1:
            axes = [axes]
        fig.suptitle('Antibody Status Distribution', fontsize=16, fontweight='bold')
        
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # Green, Red, Gray
        
        for i, col in enumerate(antibody_status_cols):
            status_counts = df[col].value_counts()
            antibody_name = col.replace('PostivePatient_status', '').replace('_status', '')
            
            wedges, texts, autotexts = axes[i].pie(status_counts.values, 
                                                    labels=status_counts.index,
                                                    autopct='%1.1f%%',
                                                    startangle=90,
                                                    colors=colors[:len(status_counts)],
                                                    textprops={'fontsize': 11})
            axes[i].set_title(f'{antibody_name} Antibody\n(N={len(df):,})', 
                            fontsize=12, fontweight='bold')
            
            for j, (text, count) in enumerate(zip(texts, status_counts.values)):
                text.set_text(f'{text.get_text()}\n(n={count:,})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'Antibody_Status_Distribution.png'), bbox_inches='tight', dpi=300)
        plt.close()
    
    # 4. Diagnosis date temporal distribution
    print("Creating diagnosis date distribution plot...")
    if 'DiagnosisDate' in df.columns:
        df['DiagnosisYear'] = pd.to_datetime(df['DiagnosisDate']).dt.year
        year_counts = df['DiagnosisYear'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(year_counts.index, year_counts.values, marker='o', linewidth=2, markersize=8, color='darkblue')
        ax.fill_between(year_counts.index, year_counts.values, alpha=0.3, color='lightblue')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Diagnoses', fontsize=12)
        ax.set_title('T2D Diagnosis Temporal Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'Diagnosis_Temporal_Distribution.png'), bbox_inches='tight', dpi=300)
        plt.close()
    
    print("\nAll visualizations saved successfully!")

def create_medication_percentage_visualization(df, patients_with_data):
    """Create medication visualization with PERCENTAGES on y-axis"""
    print("\nCreating medication percentage visualization...")
    
    med_cols = [col for col in df.columns if any(drug in col for drug in 
                ['Insulins', 'Biguanide', 'GLP1', 'DPP4', 'SGLT2', 'Sulfonylureas', 
                 'Meglitinides', 'Thiazolidinediones', 'Alpha_glucosidase', 'Amylin'])]
    
    if not med_cols:
        return
    
    # Group by medication class and time frame
    med_data = {}
    for col in med_cols:
        for drug_class in ['Insulins', 'Biguanide', 'GLP1_agonists', 'DPP4_inhibitors', 
                          'SGLT2_inhibitors', 'Sulfonylureas', 'Meglitinides', 
                          'Thiazolidinediones', 'Alpha_glucosidase_inhibitors', 'Amylin_analogue']:
            if drug_class in col:
                if drug_class not in med_data:
                    med_data[drug_class] = {'at_diagnosis': 0, '2year': 0, '5year': 0}
                
                if '_at_diagnosis' in col:
                    med_data[drug_class]['at_diagnosis'] = df[col].sum()
                elif '_2year' in col:
                    med_data[drug_class]['2year'] = df[col].sum()
                elif '_5year' in col:
                    med_data[drug_class]['5year'] = df[col].sum()
    
    drug_names = list(med_data.keys())
    x = np.arange(len(drug_names))
    width = 0.25
    
    # Get normalized denominators
    n_at_diag = len(patients_with_data['at_diagnosis'])
    n_2year = len(patients_with_data['2year'])
    n_5year = len(patients_with_data['5year'])
    
    # Calculate percentages
    at_diag_pct = [(med_data[drug]['at_diagnosis']/n_at_diag)*100 if n_at_diag > 0 else 0 for drug in drug_names]
    two_year_pct = [(med_data[drug]['2year']/n_2year)*100 if n_2year > 0 else 0 for drug in drug_names]
    five_year_pct = [(med_data[drug]['5year']/n_5year)*100 if n_5year > 0 else 0 for drug in drug_names]
    
    # Create plot with PERCENTAGES on y-axis
    fig, ax = plt.subplots(figsize=(16, 8))
    bars1 = ax.bar(x - width, at_diag_pct, width, label=f'At Diagnosis (N={n_at_diag:,})', 
                   color='#3498db', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x, two_year_pct, width, label=f'2 Years (N={n_2year:,})', 
                   color='#e67e22', edgecolor='black', alpha=0.8)
    bars3 = ax.bar(x + width, five_year_pct, width, label=f'5 Years (N={n_5year:,})', 
                   color='#9b59b6', edgecolor='black', alpha=0.8)
    
    ax.set_ylabel('Percentage of Patients (%)', fontsize=12)
    ax.set_title('Medication Class Usage by Time Frame (Percentage)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace('_', ' ') for name in drug_names], 
                       rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for i, (pct, bar) in enumerate(zip(at_diag_pct, bars1)):
        if pct > 0:
            ax.text(bar.get_x() + bar.get_width()/2, pct, 
                   f'{pct:.1f}%', 
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    for i, (pct, bar) in enumerate(zip(two_year_pct, bars2)):
        if pct > 0:
            ax.text(bar.get_x() + bar.get_width()/2, pct, 
                   f'{pct:.1f}%', 
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    for i, (pct, bar) in enumerate(zip(five_year_pct, bars3)):
        if pct > 0:
            ax.text(bar.get_x() + bar.get_width()/2, pct, 
                   f'{pct:.1f}%', 
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Medication_Classes_Percentage.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    print("  Medication percentage visualization saved!")


def create_comorbidity_percentage_visualization(df, patients_with_data):
    """Create comorbidity visualization with PERCENTAGES on y-axis"""
    print("\nCreating comorbidity percentage visualization...")
    
    condition_cols = [col for col in df.columns if any(cond in col for cond in 
                     ['DKA', 'Ketosis', 'Dyslipidemia', 'Hypertension', 'Diabetic_Retinopathy',
                      'Microalbuminuria', 'Neuropathy', 'Hypoglycemia'])]
    
    if not condition_cols:
        return
    
    # Group by condition and time frame
    condition_data = {}
    for col in condition_cols:
        for cond in ['DKA', 'Ketosis', 'Dyslipidemia', 'Hypertension', 'Diabetic_Retinopathy',
                     'Microalbuminuria', 'Neuropathy', 'Hypoglycemia']:
            if cond in col:
                if cond not in condition_data:
                    condition_data[cond] = {'at_diagnosis': 0, '2year': 0, '5year': 0}
                
                if '_at_diagnosis' in col:
                    condition_data[cond]['at_diagnosis'] = df[col].sum()
                elif '_2year' in col:
                    condition_data[cond]['2year'] = df[col].sum()
                elif '_5year' in col:
                    condition_data[cond]['5year'] = df[col].sum()
    
    cond_names = list(condition_data.keys())
    x = np.arange(len(cond_names))
    width = 0.25
    
    # Get normalized denominators
    n_at_diag = len(patients_with_data['at_diagnosis'])
    n_2year = len(patients_with_data['2year'])
    n_5year = len(patients_with_data['5year'])
    
    # Calculate percentages
    at_diag_pct = [(condition_data[cond]['at_diagnosis']/n_at_diag)*100 if n_at_diag > 0 else 0 for cond in cond_names]
    two_year_pct = [(condition_data[cond]['2year']/n_2year)*100 if n_2year > 0 else 0 for cond in cond_names]
    five_year_pct = [(condition_data[cond]['5year']/n_5year)*100 if n_5year > 0 else 0 for cond in cond_names]
    
    # Create plot with PERCENTAGES on y-axis
    fig, ax = plt.subplots(figsize=(16, 8))
    bars1 = ax.bar(x - width, at_diag_pct, width, label=f'At Diagnosis (N={n_at_diag:,})', 
                   color='#1abc9c', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x, two_year_pct, width, label=f'2 Years (N={n_2year:,})', 
                   color='#f39c12', edgecolor='black', alpha=0.8)
    bars3 = ax.bar(x + width, five_year_pct, width, label=f'5 Years (N={n_5year:,})', 
                   color='#c0392b', edgecolor='black', alpha=0.8)
    
    ax.set_ylabel('Percentage of Patients (%)', fontsize=12)
    ax.set_title('Comorbidity Occurrence by Time Frame (Percentage)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace('_', ' ') for name in cond_names], 
                       rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for i, (pct, bar) in enumerate(zip(at_diag_pct, bars1)):
        if pct > 0:
            ax.text(bar.get_x() + bar.get_width()/2, pct, 
                   f'{pct:.1f}%', 
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    for i, (pct, bar) in enumerate(zip(two_year_pct, bars2)):
        if pct > 0:
            ax.text(bar.get_x() + bar.get_width()/2, pct, 
                   f'{pct:.1f}%', 
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    for i, (pct, bar) in enumerate(zip(five_year_pct, bars3)):
        if pct > 0:
            ax.text(bar.get_x() + bar.get_width()/2, pct, 
                   f'{pct:.1f}%', 
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Comorbidities_Percentage.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    print("  Comorbidity percentage visualization saved!")


def generate_analysis_and_visualizations(patients_with_data):
    """Main function to generate analysis and visualizations"""
    print("\n" + "="*60)
    print("STARTING ANALYSIS AND VISUALIZATION GENERATION")
    print("="*60)
    
    # Load enhanced data
    enhanced_file = os.path.join(OUTPUT_DIR, 'T2D_Enhanced_OMOP.csv')
    
    if not os.path.exists(enhanced_file):
        print(f"Error: Enhanced data file not found at {enhanced_file}")
        return
    
    print(f"\nLoading enhanced data from: {enhanced_file}")
    df = pd.read_csv(enhanced_file)
    print(f"Loaded {len(df):,} patients with {len(df.columns):,} features")
    
    # Clean lab values
    df = clean_lab_values(df)
    
    # Process antibody flags
    df = process_antibody_flags(df)
    
    # Create summary statistics table
    summary_df = create_summary_statistics(df)
    print("\nSummary Statistics Preview:")
    print(summary_df.head(20).to_string(index=False))
    
    # Create visualizations with normalized data
    create_visualizations(df, patients_with_data)
    
    print("\n" + "="*60)
    print("ANALYSIS AND VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - Summary_Statistics_Table.csv")
    print("  - Demographics_Distribution.png")
    print("  - Lab_Measurements_Distribution.png")
    print("  - Antibody_Status_Distribution.png")
    print("  - Medication_Classes_Normalized.png")
    print("  - Comorbidities_Normalized.png")
    print("  - RaceEthnicity_Composite.png")
    print("  - Diagnosis_Temporal_Distribution.png")
    print("  - DKA_at_Diagnosis_Temporal.png")
    print("  - DKA_at_Diagnosis_by_Year.csv")


def main():
    """Main execution function"""
    print("="*60)
    print("STARTING ENHANCED OMOP DATA INTEGRATION")
    print("="*60)
    
    baseline_df = load_baseline_data()
    enhanced_df = extract_demographics(baseline_df)
    enhanced_df = create_race_ethnicity_composite(enhanced_df)
    
    # Extract features and track patients with data
    enhanced_df, med_data_counts, med_patients_with_data = extract_medications_features(enhanced_df)
    enhanced_df, icd_data_counts, icd_patients_with_data = extract_icd_features(enhanced_df)
    
    # Save enhanced data with metadata
    patients_with_data = save_enhanced_data(enhanced_df, med_data_counts, icd_data_counts, 
                                           med_patients_with_data, icd_patients_with_data)
    
    # Generate visualizations with normalized denominators
    generate_analysis_and_visualizations(patients_with_data)
    
    print("\n" + "="*60)
    print("ENHANCEMENT COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()