import pandas as pd
import boto3
import os
import gc
from io import StringIO
import numpy as np
import re
from datetime import datetime, timedelta

# Configuration
S3_BUCKET = 'dsw-sagemaker-dev-s3'
S3_PREFIX = 'OMOP_data_extractions/T1D_Mike/'

# Drug classes configuration
DRUG_CLASSES = {
    'Insulins': [
        'insulin aspart', 'insulin degludec', 'insulin detemir', 'insulin glargine',
        'insulin glulisine', 'insulin human', 'insulin regular', 'insulin nph',
        'insulin isophane', 'insulin lispro', 'insulin lispro protamine',
        'inhaled human insulin', 'technosphere insulin', 'insulin lantus', 'insulin tresiba',
        'insulin novolog', 'insulin humalog', 'insulin apidra', 'insulin levemir',
        'insulin toujeo', 'insulin basaglar', 'insulin fiasp'
    ],
    'Insulin_Pump': [
        'insulin pump', 'continuous subcutaneous insulin infusion', 'csii',
        'omnipod', 'medtronic pump', 'tandem pump', 't:slim', 'minimed',
        'insulin infusion pump', 'subcutaneous insulin pump'
    ],
    'CGM_Device': [
        'continuous glucose monitor', 'cgm', 'dexcom', 'freestyle libre',
        'guardian sensor', 'medtronic cgm', 'glucose sensor', 'flash glucose monitor',
        'intermittent glucose monitor', 'real-time cgm', 'rtcgm'
    ],
    'Biguanide': ['metformin', 'metformin hydrochloride', 'metformin extended release'],
    'GLP1_agonists': [
        'albiglutide', 'dulaglutide', 'exenatide', 'liraglutide', 'lixisenatide',
        'semaglutide', 'tirzepatide', 'trulicity', 'ozempic', 'victoza', 'byetta',
        'bydureon', 'rybelsus', 'mounjaro', 'wegovy'
    ],
    'SGLT2_inhibitors': [
        'canagliflozin', 'dapagliflozin', 'empagliflozin', 'ertugliflozin',
        'ipragliflozin', 'luseogliflozin', 'remogliflozin', 'sotagliflozin', 
        'tofogliflozin', 'jardiance', 'farxiga', 'invokana', 'steglatro'
    ],
    'ACE_Inhibitors': [
        'lisinopril', 'enalapril', 'captopril', 'benazepril', 'fosinopril',
        'perindopril', 'quinapril', 'ramipril', 'trandolapril', 'moexipril'
    ],
    'Statins': [
        'atorvastatin', 'rosuvastatin', 'simvastatin', 'pravastatin', 'lovastatin',
        'fluvastatin', 'pitavastatin', 'lipitor', 'crestor', 'zocor', 'pravachol',
        'mevacor', 'lescol', 'livalo'
    ],
    'Amylin_analogue': ['pramlintide', 'symlin']
}

# Condition codes configuration
CONDITION_CODES = {
    'DKA': {
        'ICD9': ['250.11', '250.13', '250.10', '250.12'],
        'ICD10': ['E10.10', 'E10.11', 'E11.10', 'E11.11', 'E13.10', 'E13.11']
    },
    'Ketosis': {
        'ICD9': ['276.2', '790.6'],
        'ICD10': ['E87.2', 'R82.4']
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
        'ICD10': ['E10.31', 'E10.32', 'E10.33', 'E10.34', 'E10.35', 'E10.36', 'E10.37',
                  'E11.31', 'E11.32', 'E11.33', 'E11.34', 'E11.35', 'E11.36', 'E11.37']
    },
    'Microalbuminuria': {
        'ICD9': ['791.06'],
        'ICD10': ['R80.9', 'N18.3']
    },
    'Neuropathy': {
        'ICD9': ['250.61', '250.63', '250.60', '250.62', '357.2'],
        'ICD10': ['E10.40', 'E10.41', 'E10.42', 'E10.43', 'E10.44', 'E10.49',
                  'E11.40', 'E11.41', 'E11.42', 'E11.43', 'E11.44', 'E11.49']
    },
    'Hypoglycemia': {
        'ICD9': ['250.3', '250.8', '251.0', '251.1', '251.2', '270.3', '775.0', '775.6', '962.39'],
        'ICD10': ['E10.641', 'E10.649', 'E11.641', 'E11.649', 'E13.641', 'E13.649',
                  'E15', 'E16.0', 'E16.1', 'E16.2', 'T38.3X1A', 'T38.3X1D', 'T38.3X1S']
    }
}

# Measurement types configuration  
MEASUREMENT_TYPES = {
    'height': {
        'keywords': ['height', 'body height', 'stature', 'standing height', 'ht'],
        'must_have': [],
        'exclude': ['weight', 'sitting', 'fundal']
    },
    'weight': {
        'keywords': ['weight', 'body weight', 'wt', 'body mass'],
        'must_have': [],
        'exclude': ['height', 'birth', 'ideal', 'target', 'gain', 'loss']
    },
    'bmi': {
        'keywords': ['bmi', 'body mass index', 'body-mass index'],
        'must_have': [],
        'exclude': ['percentile', 'z-score', 'zscore']
    },
    'systolic_blood_pressure': {
        'keywords': ['systolic', 'systolic blood pressure', 'systolic bp', 'sbp'],
        'must_have': [],
        'exclude': ['diastolic', 'mean']
    },
    'diastolic_blood_pressure': {
        'keywords': ['diastolic', 'diastolic blood pressure', 'diastolic bp', 'dbp'],
        'must_have': [],
        'exclude': ['systolic', 'mean']
    },
    'hba1c': {
        'keywords': ['hba1c', 'hemoglobin a1c', 'a1c', 'glycosylated hemoglobin'],
        'must_have': [],
        'exclude': ['mean', 'estimated', 'average']
    },
    'c_peptide': {
        'keywords': ['c-peptide', 'c peptide', 'cpeptide', 'connecting peptide'],
        'must_have': [],
        'exclude': ['urine', 'brain', 'bnp']
    },
    'hdl': {
        'keywords': ['hdl', 'hdl cholesterol', 'hdl-c', 'high density lipoprotein'],
        'must_have': [],
        'exclude': ['ratio', 'total', 'non-hdl', 'vldl']
    },
    'ldl': {
        'keywords': ['ldl', 'ldl cholesterol', 'ldl-c', 'low density lipoprotein'],
        'must_have': [],
        'exclude': ['ratio', 'total', 'vldl']
    },
    'triglycerides': {
        'keywords': ['triglycerides', 'triglyceride', 'trig', 'trigs', 'tg'],
        'must_have': [],
        'exclude': ['ratio']
    },
    'creatinine': {
        'keywords': ['creatinine', 'serum creatinine', 'creat', 'cr', 'scr'],
        'must_have': [],
        'exclude': ['urine', 'clearance', 'ratio', 'kinase']
    },
    'egfr': {
        'keywords': ['egfr', 'estimated gfr', 'glomerular filtration rate', 'gfr'],
        'must_have': [],
        'exclude': []
    },
    'urine_microalbumin': {
        'keywords': ['urine microalbumin', 'microalbumin', 'microalbuminuria'],
        'must_have': [],
        'exclude': ['ratio', 'acr', 'uacr']
    }
}

def read_s3_csv(bucket, key, chunksize=None):
    """Read CSV from S3"""
    s3 = boto3.client('s3')
    if chunksize:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj['Body'], chunksize=chunksize)
    else:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj['Body'])

def process_medications(cpt_data, chunk_size=100000):
    """Add OMOP medication variables to CPT data"""
    print("\n" + "="*60)
    print("PROCESSING MEDICATIONS")
    print("="*60)
    
    # Get unique person IDs and diagnosis dates
    person_dates = cpt_data[['PEDSNET_ID', 'Date of Dx']].copy()
    person_dates['Date of Dx'] = pd.to_datetime(person_dates['Date of Dx'])
    
    # Initialize medication columns
    for drug_class in DRUG_CLASSES.keys():
        cpt_data[f'OMOP_{drug_class}_onset'] = 0
        cpt_data[f'OMOP_{drug_class}_anytime'] = 0
    
    # Track medications per patient
    patient_meds_onset = {pid: set() for pid in cpt_data['PEDSNET_ID'].unique()}
    patient_meds_anytime = {pid: set() for pid in cpt_data['PEDSNET_ID'].unique()}
    
    try:
        # Read medication data
        s3 = boto3.client('s3')
        medication_file = None
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=f'{S3_PREFIX}medications/')
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].endswith('.csv'):
                    medication_file = obj['Key']
                    break
        
        if not medication_file:
            print("No medication file found")
            return cpt_data
        
        print(f"Processing {medication_file}")
        chunks = read_s3_csv(S3_BUCKET, medication_file, chunksize=chunk_size)
        
        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}...", end='\r')
            
            # Filter for our patients
            chunk = chunk[chunk['PERSON_ID'].isin(cpt_data['PEDSNET_ID'])]
            
            if 'DRUG_SOURCE_VALUE' not in chunk.columns:
                continue
            
            # Find date column
            date_col = None
            for col in ['DRUG_EXPOSURE_START_DATE', 'Drug_Exposure_Start_Date']:
                if col in chunk.columns:
                    date_col = col
                    break
            
            if date_col:
                chunk[date_col] = pd.to_datetime(chunk[date_col], errors='coerce')
            
            # Check each medication against drug classes
            for _, row in chunk.iterrows():
                if pd.isna(row['DRUG_SOURCE_VALUE']):
                    continue
                
                person_id = row['PERSON_ID']
                drug_name = str(row['DRUG_SOURCE_VALUE']).lower()
                
                # Get diagnosis date for this patient
                dx_date = person_dates[person_dates['PEDSNET_ID'] == person_id]['Date of Dx'].iloc[0]
                
                # Check each drug class
                for drug_class, drugs in DRUG_CLASSES.items():
                    for drug in drugs:
                        if drug.lower() in drug_name:
                            # Add to anytime
                            patient_meds_anytime[person_id].add(drug_class)
                            
                            # Check if within onset window (±3 months)
                            if date_col and pd.notna(row[date_col]) and pd.notna(dx_date):
                                days_diff = abs((row[date_col] - dx_date).days)
                                if days_diff <= 90:  # Within 3 months
                                    patient_meds_onset[person_id].add(drug_class)
                            break
        
        print("\n\nUpdating medication columns...")
        # Update CPT data with medication flags
        for idx, row in cpt_data.iterrows():
            pid = row['PEDSNET_ID']
            for drug_class in patient_meds_onset.get(pid, []):
                cpt_data.at[idx, f'OMOP_{drug_class}_onset'] = 1
            for drug_class in patient_meds_anytime.get(pid, []):
                cpt_data.at[idx, f'OMOP_{drug_class}_anytime'] = 1
        
        # Print summary
        print("\nMedication Summary:")
        for drug_class in DRUG_CLASSES.keys():
            onset_count = cpt_data[f'OMOP_{drug_class}_onset'].sum()
            anytime_count = cpt_data[f'OMOP_{drug_class}_anytime'].sum()
            print(f"  {drug_class}:")
            print(f"    Onset (±3 months): {onset_count} ({onset_count/len(cpt_data)*100:.1f}%)")
            print(f"    Anytime: {anytime_count} ({anytime_count/len(cpt_data)*100:.1f}%)")
    
    except Exception as e:
        print(f"Error processing medications: {e}")
    
    return cpt_data

def process_icd_codes(cpt_data, chunk_size=100000):
    """Add OMOP ICD code variables to CPT data"""
    print("\n" + "="*60)
    print("PROCESSING ICD CODES")
    print("="*60)
    
    # Get unique person IDs and diagnosis dates
    person_dates = cpt_data[['PEDSNET_ID', 'Date of Dx']].copy()
    person_dates['Date of Dx'] = pd.to_datetime(person_dates['Date of Dx'])
    
    # Initialize ICD columns
    for condition in CONDITION_CODES.keys():
        cpt_data[f'OMOP_{condition}_onset'] = 0
        cpt_data[f'OMOP_{condition}_anytime'] = 0
    
    # Track conditions per patient
    patient_conditions_onset = {pid: set() for pid in cpt_data['PEDSNET_ID'].unique()}
    patient_conditions_anytime = {pid: set() for pid in cpt_data['PEDSNET_ID'].unique()}
    
    try:
        print("Processing condition_occurrence.csv")
        chunks = read_s3_csv(S3_BUCKET, f'{S3_PREFIX}icd_codes/condition_occurrence.csv', 
                           chunksize=chunk_size)
        
        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}...", end='\r')
            
            # Filter for our patients
            chunk = chunk[chunk['PERSON_ID'].isin(cpt_data['PEDSNET_ID'])]
            
            # Extract ICD codes
            if 'CONDITION_SOURCE_VALUE' in chunk.columns:
                chunk['extracted_icd'] = chunk['CONDITION_SOURCE_VALUE'].apply(
                    lambda x: x.split('|')[1].strip() if pd.notna(x) and '|' in x else x
                )
            else:
                continue
            
            # Find date column
            date_col = None
            for col in ['CONDITION_START_DATE', 'Condition_Start_Date']:
                if col in chunk.columns:
                    date_col = col
                    break
            
            if date_col:
                chunk[date_col] = pd.to_datetime(chunk[date_col], errors='coerce')
            
            # Check each ICD code against conditions
            for _, row in chunk.iterrows():
                if pd.isna(row['extracted_icd']):
                    continue
                
                person_id = row['PERSON_ID']
                icd_code = str(row['extracted_icd'])
                
                # Get diagnosis date for this patient
                dx_date = person_dates[person_dates['PEDSNET_ID'] == person_id]['Date of Dx'].iloc[0]
                
                # Check each condition
                for condition, codes in CONDITION_CODES.items():
                    matched = False
                    # Check ICD9 codes
                    for code in codes.get('ICD9', []):
                        if icd_code.startswith(code):
                            patient_conditions_anytime[person_id].add(condition)
                            matched = True
                            
                            # Check if within onset window
                            if date_col and pd.notna(row[date_col]) and pd.notna(dx_date):
                                days_diff = abs((row[date_col] - dx_date).days)
                                if days_diff <= 90:
                                    patient_conditions_onset[person_id].add(condition)
                            break
                    
                    if not matched:
                        # Check ICD10 codes
                        for code in codes.get('ICD10', []):
                            if icd_code.startswith(code):
                                patient_conditions_anytime[person_id].add(condition)
                                
                                # Check if within onset window
                                if date_col and pd.notna(row[date_col]) and pd.notna(dx_date):
                                    days_diff = abs((row[date_col] - dx_date).days)
                                    if days_diff <= 90:
                                        patient_conditions_onset[person_id].add(condition)
                                break
        
        print("\n\nUpdating ICD code columns...")
        # Update CPT data with condition flags
        for idx, row in cpt_data.iterrows():
            pid = row['PEDSNET_ID']
            for condition in patient_conditions_onset.get(pid, []):
                cpt_data.at[idx, f'OMOP_{condition}_onset'] = 1
            for condition in patient_conditions_anytime.get(pid, []):
                cpt_data.at[idx, f'OMOP_{condition}_anytime'] = 1
        
        # Print summary
        print("\nICD Code Summary:")
        for condition in CONDITION_CODES.keys():
            onset_count = cpt_data[f'OMOP_{condition}_onset'].sum()
            anytime_count = cpt_data[f'OMOP_{condition}_anytime'].sum()
            print(f"  {condition}:")
            print(f"    Onset (±3 months): {onset_count} ({onset_count/len(cpt_data)*100:.1f}%)")
            print(f"    Anytime: {anytime_count} ({anytime_count/len(cpt_data)*100:.1f}%)")
    
    except Exception as e:
        print(f"Error processing ICD codes: {e}")
    
    return cpt_data

def process_measurements(cpt_data, chunk_size=100000, min_coverage=0.1):
    """Add OMOP measurement variables to CPT data (onset only)"""
    print("\n" + "="*60)
    print("PROCESSING MEASUREMENTS")
    print("="*60)
    
    # Get unique person IDs and diagnosis dates
    person_dates = cpt_data[['PEDSNET_ID', 'Date of Dx']].copy()
    person_dates['Date of Dx'] = pd.to_datetime(person_dates['Date of Dx'])
    
    # Track measurements per patient
    patient_measurements = {measure: {pid: None for pid in cpt_data['PEDSNET_ID'].unique()} 
                           for measure in MEASUREMENT_TYPES.keys()}
    
    try:
        print("Processing measurement.csv")
        chunks = read_s3_csv(S3_BUCKET, f'{S3_PREFIX}measurements/measurement.csv', 
                           chunksize=chunk_size)
        
        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}...", end='\r')
            
            # Filter for our patients
            chunk = chunk[chunk['PERSON_ID'].isin(cpt_data['PEDSNET_ID'])]
            
            if 'MEASUREMENT_SOURCE_VALUE' not in chunk.columns:
                continue
            
            # Find date and value columns
            date_col = None
            for col in ['MEASUREMENT_DATE', 'Measurement_Date']:
                if col in chunk.columns:
                    date_col = col
                    break
            
            value_col = None
            for col in ['VALUE_AS_NUMBER', 'Value_As_Number']:
                if col in chunk.columns:
                    value_col = col
                    break
            
            if date_col:
                chunk[date_col] = pd.to_datetime(chunk[date_col], errors='coerce')
            
            # Process each measurement
            for _, row in chunk.iterrows():
                if pd.isna(row['MEASUREMENT_SOURCE_VALUE']):
                    continue
                
                person_id = row['PERSON_ID']
                measure_name = str(row['MEASUREMENT_SOURCE_VALUE']).lower()
                
                # Get diagnosis date
                dx_date = person_dates[person_dates['PEDSNET_ID'] == person_id]['Date of Dx'].iloc[0]
                
                # Check if within onset window
                if date_col and pd.notna(row[date_col]) and pd.notna(dx_date):
                    days_diff = abs((row[date_col] - dx_date).days)
                    if days_diff <= 90:  # Within 3 months
                        # Check each measurement type
                        for measure_type, config in MEASUREMENT_TYPES.items():
                            matched = False
                            
                            # Check keywords
                            for keyword in config['keywords']:
                                if keyword.lower() in measure_name:
                                    matched = True
                                    break
                            
                            if matched:
                                # Check must_have
                                if config['must_have']:
                                    has_all = all(must.lower() in measure_name for must in config['must_have'])
                                    if not has_all:
                                        matched = False
                                
                                # Check exclusions
                                if matched:
                                    for exclude in config['exclude']:
                                        if exclude.lower() in measure_name:
                                            matched = False
                                            break
                            
                            if matched and value_col and pd.notna(row[value_col]):
                                # Store the measurement value (keeping the latest if multiple)
                                patient_measurements[measure_type][person_id] = row[value_col]
        
        print("\n\nAdding measurement columns...")
        # Calculate coverage and add columns for measurements with sufficient coverage
        measurements_added = []
        measurements_dropped = []
        
        for measure_type in MEASUREMENT_TYPES.keys():
            # Count patients with this measurement
            patients_with = sum(1 for v in patient_measurements[measure_type].values() if v is not None)
            coverage = patients_with / len(cpt_data)
            
            if coverage >= min_coverage:
                # Add column
                col_name = f'OMOP_{measure_type}_onset'
                cpt_data[col_name] = cpt_data['PEDSNET_ID'].apply(
                    lambda pid: patient_measurements[measure_type].get(pid)
                )
                measurements_added.append((measure_type, coverage))
            else:
                measurements_dropped.append((measure_type, coverage))
        
        # Print summary
        print("\nMeasurements Added:")
        for measure, coverage in measurements_added:
            col_name = f'OMOP_{measure}_onset'
            non_null = cpt_data[col_name].notna().sum()
            print(f"  {measure}: {non_null} patients ({coverage*100:.1f}% coverage)")
        
        if measurements_dropped:
            print("\nMeasurements Dropped (< 10% coverage):")
            for measure, coverage in measurements_dropped:
                print(f"  {measure}: {coverage*100:.1f}% coverage")
    
    except Exception as e:
        print(f"Error processing measurements: {e}")
    
    return cpt_data

def main():
    """Main execution function"""
    print("="*60)
    print("T1D CPT DATA PROCESSING WITH OMOP INTEGRATION")
    print("="*60)
    
    # Step 1: Load CPT data
    print("\nStep 1: Loading T1D CPT data...")
    cpt_data = pd.read_csv('T1D_mike_data.csv')
    print(f"Loaded {len(cpt_data)} patients")
    
    # Verify PEDSNET_ID column exists
    if 'PEDSNET_ID' not in cpt_data.columns:
        print("ERROR: PEDSNET_ID column not found in CPT data!")
        return
    
    # Step 2: Add OMOP Medications
    print("\nStep 2: Adding OMOP Medications...")
    cpt_data = process_medications(cpt_data)
    
    # Step 3: Add OMOP ICD Codes
    print("\nStep 3: Adding OMOP ICD Codes...")
    cpt_data = process_icd_codes(cpt_data)
    
    # Step 4: Add OMOP Measurements
    print("\nStep 4: Adding OMOP Measurements...")
    cpt_data = process_measurements(cpt_data)
    
    # Step 5: Remove specified columns
    print("\nStep 5: Removing specified columns...")
    columns_to_remove = [
        'MRN', 'Patient', 'Date of Dx', 'CGM Date', 'PATIENTID',
        'PEDSNET_ID', 'TCH_SOURCE_ID', 'PAT_MRN_ID', 'Source_DS2', 
        'Source_DS1', 'Last Pump Rx Date', 'Retinal Eye Exam Order Dt',
        'Last Endo OV', 'Last Endo Provider', 'Last Endo Dept',
        'Last CDE Enc', 'Last RD Enc', 'Last CDE Dept',
        'Last RD Dept', 'Last SW Enc', 'Last SW Dept', 'Last Psychology Enc Dept',
        'Lst Enc Nutrition', 'Last Canceled Dep', 'Last NoShow Dep', 'Celiac Screen Order Dt',
        'Last Lipid Panel', 'Last LDL Dt', 'Last Microalbumin Dt', 'Last Creatinine Dt',
        'Last BUN Dt', 'Last Ur Micro:Creat Dt', 'RDT ID', 'MyChart Status', 'Pt Comm Pref'
    ]
    
    # Remove columns that exist in the dataframe
    columns_to_drop = [col for col in columns_to_remove if col in cpt_data.columns]
    cpt_data = cpt_data.drop(columns=columns_to_drop)
    print(f"Removed {len(columns_to_drop)} columns")
    
    # Print final dataset info
    print("\n" + "="*60)
    print("FINAL DATASET SUMMARY")
    print("="*60)
    print(f"Total rows: {len(cpt_data)}")
    print(f"Total columns: {len(cpt_data.columns)}")
    
    # Count OMOP columns added
    omop_cols = [col for col in cpt_data.columns if col.startswith('OMOP_')]
    print(f"OMOP columns added: {len(omop_cols)}")
    
    # Step 6: Save final dataset
    print("\nStep 6: Saving final dataset...")
    cpt_data.to_csv('Mike_CPT_OMOP_data.csv', index=False)
    print("Dataset saved as 'Mike_CPT_OMOP_data.csv'")
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()