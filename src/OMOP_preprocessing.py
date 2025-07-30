# Here we will access the tables from OMOP_extraction.py and preprocess them to a machine learning ready dataset ... 
import pandas as pd
import numpy as np
from datetime import datetime
import boto3
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Define medication mappings to drug classes
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

# Define condition mappings with ICD codes
CONDITIONS = {
    'DKA': {
        'ICD9': [
            '250.11',  # Diabetes mellitus with ketoacidosis, Type 1
            '250.13',  # Diabetes mellitus with ketoacidosis, Type 2
            '250.10',  # Diabetes mellitus with ketoacidosis, Type 1, uncontrolled
            '250.12',  # Diabetes mellitus with ketoacidosis, Type 2, uncontrolled
        ],
        'ICD10': [
            # Type 1 Diabetes (DKA)
            'E10.1',   # Type 1 diabetes mellitus with ketoacidosis
            'E10.10',  # Type 1 diabetes mellitus with ketoacidosis, without coma
            'E10.11',  # Type 1 diabetes mellitus with ketoacidosis with coma

            # Type 2 Diabetes (DKA)
            'E11.1',   # Type 2 diabetes mellitus with ketoacidosis
            'E11.10',  # Type 2 diabetes mellitus with ketoacidosis, without coma
            'E11.11',  # Type 2 diabetes mellitus with ketoacidosis with coma

            # Other Forms of Diabetes (DKA)
            'E13.10',  # Other specified diabetes mellitus with ketoacidosis, without coma
            'E13.11',  # Other specified diabetes mellitus with ketoacidosis with coma

            # Secondary Diabetes (DKA)
            'E08.10',  # Diabetes due to underlying condition with ketoacidosis, without coma
            'E08.11',  # Diabetes due to underlying condition with ketoacidosis with coma

            # Coma and Complications
            'E10.9',   # Type 1 diabetes mellitus, unspecified
            'E11.9',   # Type 2 diabetes mellitus, unspecified
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

# T2D ICD codes for diagnosis date
T2D_CODES = {
    'ICD9': ['250.00', '250.02'],
    'ICD10': ['E11.', 'E11.0', 'E11.1', 'E11.2', 'E11.3', 'E11.4', 'E11.5', 'E11.6', 'E11.7', 'E11.8', 'E11.9']
}

def read_s3_csv(s3_client, bucket, key):
    """Read CSV from S3"""
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
    
    # Map race and ethnicity (common OMOP concept IDs)
    race_map = {
        8527: 'White',
        8516: 'Black',
        8515: 'Asian',
        8657: 'Native American',
        8557: 'Pacific Islander',
        0: 'Unknown'
    }
    
    ethnicity_map = {
        38003563: 'Hispanic',
        38003564: 'Not Hispanic',
        0: 'Unknown'
    }
    
    demo_df['race'] = demo_df['RACE_CONCEPT_ID'].map(race_map).fillna('Other')
    demo_df['ethnicity'] = demo_df['ETHNICITY_CONCEPT_ID'].map(ethnicity_map).fillna('Unknown')
    
    # Print summary
    print(f"Birth dates found: {demo_df['birth_date'].notna().sum()}/{len(demo_df)}")
    
    return demo_df[['PERSON_ID', 'sex', 'birth_date', 'race', 'ethnicity']]

    
def get_t2d_diagnosis_date(condition_df):
    """Extract earliest T2D diagnosis date for each patient"""
    print("Finding T2D diagnosis dates...")
    
    # Filter for T2D codes
    t2d_conditions = condition_df[
        condition_df['CONDITION_SOURCE_VALUE'].apply(
            lambda x: check_icd_code(x, T2D_CODES['ICD9'] + T2D_CODES['ICD10'])
        )
    ].copy()
    
    # Convert date columns
    t2d_conditions['condition_date'] = pd.to_datetime(t2d_conditions['CONDITION_START_DATE'])
    
    # Get earliest T2D date per patient
    t2d_dates = t2d_conditions.groupby('PERSON_ID')['condition_date'].min().reset_index()
    t2d_dates.columns = ['PERSON_ID', 't2d_diagnosis_date']
    
    return t2d_dates

def process_medications(drug_df, t2d_dates):
    """Process drug exposure table to extract medication information"""
    print("Processing medications...")
    
    # Convert dates
    drug_df['drug_date'] = pd.to_datetime(drug_df['DRUG_EXPOSURE_START_DATE'])
    
    # Merge with T2D dates
    drug_df = drug_df.merge(t2d_dates, on='PERSON_ID', how='left')
    
    # Calculate years from diagnosis
    drug_df['years_from_diagnosis'] = (
        (drug_df['drug_date'] - drug_df['t2d_diagnosis_date']).dt.days / 365.25
    )
    
    # Initialize result dataframe
    patient_drugs = pd.DataFrame({'PERSON_ID': drug_df['PERSON_ID'].unique()})
    
    # Process each drug class
    for drug_class, drug_list in DRUG_CLASSES.items():
        print(f"  Processing {drug_class}...")
        
        # Create pattern for matching
        pattern = '|'.join(drug_list)
        
        # Find matching drugs
        class_drugs = drug_df[
            drug_df['DRUG_SOURCE_VALUE'].str.lower().str.contains(pattern, na=False, regex=True)
        ].copy()
        
        if len(class_drugs) > 0:
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
            patient_drugs[drug_class] = 0
            patient_drugs[f'{drug_class}_date'] = pd.NaT
            patient_drugs[f'{drug_class}_years_from_diagnosis'] = np.nan
    
    # Fill NaN values
    for drug_class in DRUG_CLASSES.keys():
        patient_drugs[drug_class] = patient_drugs[drug_class].fillna(0).astype(int)
    
    return patient_drugs

def process_conditions(condition_df, t2d_dates):
    """Process condition occurrence table to extract relevant conditions"""
    print("Processing conditions...")
    
    # Convert dates
    condition_df['condition_date'] = pd.to_datetime(condition_df['CONDITION_START_DATE'])
    
    # Merge with T2D dates
    condition_df = condition_df.merge(t2d_dates, on='PERSON_ID', how='left')
    
    # Calculate years from diagnosis
    condition_df['years_from_diagnosis'] = (
        (condition_df['condition_date'] - condition_df['t2d_diagnosis_date']).dt.days / 365.25
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
            patient_conditions[condition_name] = 0
            patient_conditions[f'{condition_name}_date'] = pd.NaT
            patient_conditions[f'{condition_name}_years_from_diagnosis'] = np.nan
    
    # Fill NaN values
    for condition_name in CONDITIONS.keys():
        patient_conditions[condition_name] = patient_conditions[condition_name].fillna(0).astype(int)
    
    return patient_conditions


def process_measurements_chunked(s3_client, bucket, prefix, t2d_dates, chunk_size=100000):
    """Process measurement table in chunks to extract lab values"""
    print("Processing measurements in chunks...")
    
    # Define measurement mappings (simplified version without unit filtering)
    MEASUREMENTS = {
        'fasting_glucose': ['fasting glucose', 'fasting blood glucose', 'fbs', 'fasting plasma glucose', 'fpg', 'fasting blood sugar'],
        'hba1c': ['hba1c', 'hemoglobin a1c', 'a1c', 'glycosylated hemoglobin', 'glycated hemoglobin', 'hgb a1c', 'hb a1c'],
        'c_peptide': ['c-peptide', 'c peptide', 'cpeptide', 'connecting peptide', 'serum c-peptide', 'serum c peptide'],
        'glucose_2h_ogtt': ['2-h glucose', '2 hour glucose', '2h glucose', 'ogtt', 'oral glucose tolerance', '2hr glucose', 'glucose 2 hour'],
        'hdl': ['hdl', 'hdl cholesterol', 'hdl-c', 'high density lipoprotein'],
        'ldl': ['ldl', 'ldl cholesterol', 'ldl-c', 'low density lipoprotein'],
        'triglycerides': ['triglycerides', 'triglyceride', 'trig', 'trigs', 'tg'],
        'alt': ['alt', 'alanine aminotransferase', 'sgpt', 'alanine transaminase'],
        'ast': ['ast', 'aspartate aminotransferase', 'sgot', 'aspartate transaminase'],
        'bun': ['bun', 'blood urea nitrogen', 'urea nitrogen', 'serum urea', 'blood urea'],
        'creatinine': ['creatinine', 'serum creatinine', 'creat', 'cr', 'scr'],
        'egfr': ['egfr', 'estimated gfr', 'glomerular filtration rate', 'gfr', 'calculated gfr'],
        'gad65_antibody': ['gad65', 'gad65 antibody', 'anti-gad65', 'anti gad65', 'gada', 'anti-gad'],
        'ica512_antibody': ['ica512', 'ica512 antibody', 'anti-ica512', 'ia-2', 'ia2', 'ia-2 antibody'],
        'insulin_antibody': ['insulin antibody', 'anti-insulin', 'iaa', 'insulin autoantibody'],
        'znt8_antibody': ['znt8', 'znt8 antibody', 'anti-znt8', 'zinc transporter 8'],
        'urine_microalbumin': ['urine microalbumin', 'microalbumin', 'microalbuminuria', 'urinary microalbumin', 'albumin urine'],
        'urine_creatinine': ['urine creatinine', 'creatinine urine', 'urinary creatinine'],
        'urine_microalbumin_creatinine_ratio': ['microalbumin creatinine ratio', 'albumin creatinine ratio', 'acr', 'uacr', 'alb/cr ratio']
    }
    
    # Initialize accumulator dictionaries for each measurement type
    measurement_accumulators = {}
    for measurement_name in MEASUREMENTS.keys():
        measurement_accumulators[measurement_name] = {}
    
    # Get the S3 object
    measurement_key = f'{prefix}measurement.csv'
    obj = s3_client.get_object(Bucket=bucket, Key=measurement_key)
    
    # Process in chunks
    chunk_count = 0
    for chunk in pd.read_csv(obj['Body'], chunksize=chunk_size):
        chunk_count += 1
        print(f"  Processing chunk {chunk_count} ({len(chunk)} rows)...")
        
        # Convert dates
        chunk['measurement_date'] = pd.to_datetime(chunk['MEASUREMENT_DATE'])
        
        # Merge with T2D dates
        chunk = chunk.merge(t2d_dates, on='PERSON_ID', how='left')
        
        # Calculate years from diagnosis
        chunk['years_from_diagnosis'] = (
            (chunk['measurement_date'] - chunk['t2d_diagnosis_date']).dt.days / 365.25
        )
        
        # Process each measurement type
        for measurement_name, keywords in MEASUREMENTS.items():
            # Create pattern for matching (case-insensitive)
            pattern = '|'.join(keywords)
            
            # Find matching measurements
            measurement_matches = chunk[
                chunk['MEASUREMENT_SOURCE_VALUE'].str.lower().str.contains(pattern, na=False, regex=True)
            ].copy()
            
            if len(measurement_matches) > 0:
                # Group by patient and update accumulators
                for person_id, person_data in measurement_matches.groupby('PERSON_ID'):
                    if person_id not in measurement_accumulators[measurement_name]:
                        measurement_accumulators[measurement_name][person_id] = {
                            'values': [],
                            'dates': [],
                            'years_from_diagnosis': []
                        }
                    
                    # Append values and dates
                    measurement_accumulators[measurement_name][person_id]['values'].extend(
                        person_data['VALUE_AS_NUMBER'].dropna().tolist()
                    )
                    measurement_accumulators[measurement_name][person_id]['dates'].extend(
                        person_data['measurement_date'].tolist()
                    )
                    measurement_accumulators[measurement_name][person_id]['years_from_diagnosis'].extend(
                        person_data['years_from_diagnosis'].dropna().tolist()
                    )
    
    print(f"Finished processing {chunk_count} chunks. Aggregating results...")
    
    # Get unique patient IDs from all measurements
    all_patient_ids = set()
    for measurement_data in measurement_accumulators.values():
        all_patient_ids.update(measurement_data.keys())
    
    # Initialize result dataframe
    patient_measurements = pd.DataFrame({'PERSON_ID': list(all_patient_ids)})
    
    # Aggregate results for each measurement type
    for measurement_name, patient_data in measurement_accumulators.items():
        print(f"  Aggregating {measurement_name}...")
        
        if patient_data:
            # Create summary for each patient
            summary_data = []
            for person_id, data in patient_data.items():
                if data['values']:
                    values = np.array(data['values'])
                    dates = pd.to_datetime(data['dates'])
                    years = np.array(data['years_from_diagnosis'])
                    
                    # Sort by date to get first/last correctly
                    sorted_indices = np.argsort(dates)
                    values_sorted = values[sorted_indices]
                    dates_sorted = dates[sorted_indices]
                    
                    summary_data.append({
                        'PERSON_ID': person_id,
                        f'{measurement_name}_present': 1,
                        f'{measurement_name}_value_first': values_sorted[0],
                        f'{measurement_name}_value_last': values_sorted[-1],
                        f'{measurement_name}_value_mean': np.mean(values),
                        f'{measurement_name}_value_min': np.min(values),
                        f'{measurement_name}_value_max': np.max(values),
                        f'{measurement_name}_date_first': dates_sorted[0],
                        f'{measurement_name}_date_last': dates_sorted[-1],
                        f'{measurement_name}_years_from_diagnosis_first': np.min(years) if len(years) > 0 else np.nan
                    })
            
            # Convert to dataframe and merge
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                patient_measurements = patient_measurements.merge(summary_df, on='PERSON_ID', how='left')
        
        # Fill missing values for patients without this measurement
        if f'{measurement_name}_present' not in patient_measurements.columns:
            patient_measurements[f'{measurement_name}_present'] = 0
            patient_measurements[f'{measurement_name}_value_first'] = np.nan
            patient_measurements[f'{measurement_name}_value_last'] = np.nan
            patient_measurements[f'{measurement_name}_value_mean'] = np.nan
            patient_measurements[f'{measurement_name}_value_min'] = np.nan
            patient_measurements[f'{measurement_name}_value_max'] = np.nan
            patient_measurements[f'{measurement_name}_date_first'] = pd.NaT
            patient_measurements[f'{measurement_name}_date_last'] = pd.NaT
            patient_measurements[f'{measurement_name}_years_from_diagnosis_first'] = np.nan
        
        # Fill NaN values for presence indicator
        patient_measurements[f'{measurement_name}_present'] = patient_measurements[f'{measurement_name}_present'].fillna(0).astype(int)
    
    return patient_measurements

def main():
    # Initialize S3 client
    s3 = boto3.client('s3')
    
    # Define S3 paths
    bucket = 'dsw-sagemaker-dev-s3'
    prefix = 'T2D_Tosur/data/T2D_OMOP_variables/'
    
    print("Reading data from S3...")
    
    # Read the tables
    person_df = read_s3_csv(s3, bucket, f'{prefix}person.csv')
    condition_df = read_s3_csv(s3, bucket, f'{prefix}condition_occurrence.csv')
    drug_df = read_s3_csv(s3, bucket, f'{prefix}drug_exposure.csv')
    # Read the tables (ADD THIS LINE)
    measurement_df = read_s3_csv(s3, bucket, f'{prefix}measurement.csv')
    
    print(f"Loaded {len(person_df)} patients")
    print(f"Loaded {len(condition_df)} condition records")
    print(f"Loaded {len(drug_df)} drug records")
    print(f"Loaded {len(measurement_df)} measurement records")
    
    # Process demographics
    demographics = process_demographics(person_df)
    
    # Get T2D diagnosis dates
    t2d_dates = get_t2d_diagnosis_date(condition_df)
    
    # Calculate age at diagnosis
    demographics = demographics.merge(t2d_dates, on='PERSON_ID', how='left')
    demographics['age_at_diagnosis'] = (
        (demographics['t2d_diagnosis_date'] - demographics['birth_date']).dt.days / 365.25
    )
    
    # Process medications
    medications = process_medications(drug_df, t2d_dates)
    
    # Process conditions
    conditions = process_conditions(condition_df, t2d_dates)

    # Process measurements (ADD THIS SECTION)
    measurements = process_measurements_chunked(s3, bucket, prefix, t2d_dates, chunk_size=100)
    
    
    # Merge all data (MODIFY THIS SECTION)
    print("\nMerging all data...")
    final_df = demographics.merge(medications, on='PERSON_ID', how='left')
    final_df = final_df.merge(conditions, on='PERSON_ID', how='left')
    final_df = final_df.merge(measurements, on='PERSON_ID', how='left')  # ADD THIS LINE
    
    # Reorder columns (MODIFY THIS SECTION)
    base_cols = ['PERSON_ID', 'sex', 'age_at_diagnosis', 'race', 'ethnicity', 
                 'birth_date', 't2d_diagnosis_date']
    
    drug_cols = []
    for drug_class in DRUG_CLASSES.keys():
        drug_cols.extend([drug_class, f'{drug_class}_date', f'{drug_class}_years_from_diagnosis'])
    
    condition_cols = []
    for condition in CONDITIONS.keys():
        condition_cols.extend([condition, f'{condition}_date', f'{condition}_years_from_diagnosis'])
    
    # ADD THIS SECTION for measurement columns
    measurement_cols = []
    measurement_names = ['fasting_glucose', 'hba1c', 'c_peptide', 'glucose_2h_ogtt', 'hdl', 'ldl', 
                        'triglycerides', 'alt', 'ast', 'bun', 'creatinine', 'egfr', 
                        'gad65_antibody', 'ica512_antibody', 'insulin_antibody', 'znt8_antibody',
                        'urine_microalbumin', 'urine_creatinine', 'urine_microalbumin_creatinine_ratio']
    
    for measurement in measurement_names:
        measurement_cols.extend([
            f'{measurement}_present',
            f'{measurement}_value_first', f'{measurement}_value_last', 
            f'{measurement}_value_mean', f'{measurement}_value_min', f'{measurement}_value_max',
            f'{measurement}_date_first', f'{measurement}_date_last',
            f'{measurement}_years_from_diagnosis_first'
        ])
    
    final_cols = base_cols + drug_cols + condition_cols + measurement_cols  # MODIFY THIS LINE
    
    final_df = final_df[final_cols]
    
    # Save to S3
    output_key = f'{prefix}T2D_patient_features.csv'
    csv_buffer = StringIO()
    final_df.to_csv(csv_buffer, index=False)
    
    print(f"\nSaving results to s3://{bucket}/{output_key}")
    s3.put_object(Bucket=bucket, Key=output_key, Body=csv_buffer.getvalue())
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total patients: {len(final_df)}")
    print(f"Patients with T2D diagnosis date: {final_df['t2d_diagnosis_date'].notna().sum()}")
    print(f"Average age at diagnosis: {final_df['age_at_diagnosis'].mean():.1f} years")
    
    print("\n--- Demographics ---")
    print(f"Sex distribution:")
    print(final_df['sex'].value_counts())
    print(f"\nRace distribution:")
    print(final_df['race'].value_counts())
    
    print("\n--- Medication Usage ---")
    for drug_class in DRUG_CLASSES.keys():
        count = final_df[drug_class].sum()
        pct = (count / len(final_df)) * 100
        print(f"{drug_class}: {count} ({pct:.1f}%)")
    
    print("\n--- Condition Prevalence ---")
    for condition in CONDITIONS.keys():
        count = final_df[condition].sum()
        pct = (count / len(final_df)) * 100
        print(f"{condition}: {count} ({pct:.1f}%)")
    
        # Add measurement summary statistics (ADD THIS SECTION)
    print("\n--- Measurement Availability ---")
    for measurement in measurement_names:
        count = final_df[f'{measurement}_present'].sum()
        pct = (count / len(final_df)) * 100
        if count > 0:
            mean_val = final_df[f'{measurement}_value_mean'].mean()
            print(f"{measurement}: {count} ({pct:.1f}%) - Mean: {mean_val:.2f}")
        else:
            print(f"{measurement}: {count} ({pct:.1f}%)")

    print("\n✓ Processing complete!")
    
    return final_df

if __name__ == "__main__":
    final_df = main()