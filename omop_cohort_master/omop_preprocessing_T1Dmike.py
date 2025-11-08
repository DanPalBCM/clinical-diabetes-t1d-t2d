import pandas as pd
import boto3
import os
import gc
from io import StringIO
import numpy as np
import re
# Configuration
S3_BUCKET = 'dsw-sagemaker-dev-s3'
S3_PREFIX = 'OMOP_data_extractions/T1D_Mike/'

#### input
S3_BUCKET = 'dsw-sagemaker-dev-s3'
S3_PREFIX = 'OMOP_data_extractions/T1D_Mike/'

#### input
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
# Set to None to include all drugs:
# DRUG_CLASSES = None

# Condition codes remain the same as they are relevant for T1D
CONDITION_CODES = {
    'DKA': {
        'ICD9': [
            '250.11',  # Diabetes mellitus with ketoacidosis, Type 1
            '250.13',  # Diabetes mellitus with ketoacidosis, Type 2
            '250.10',  # Diabetes mellitus with ketoacidosis, Type 1, uncontrolled
            '250.12',  # Diabetes mellitus with ketoacidosis, Type 2, uncontrolled
        ],
        'ICD10': [
            # Type 1 Diabetes (DKA) - Primary focus for T1D
            'E10.10',  # Type 1 diabetes mellitus with ketoacidosis, without coma
            'E10.11',  # Type 1 diabetes mellitus with ketoacidosis with coma
            
            # Type 2 Diabetes (DKA) - Less common but possible
            'E11.10',  # Type 2 diabetes mellitus with ketoacidosis, without coma
            'E11.11',  # Type 2 diabetes mellitus with ketoacidosis with coma
            
            # Other Forms of Diabetes (DKA)
            'E13.10',  # Other specified diabetes mellitus with ketoacidosis, without coma
            'E13.11',  # Other specified diabetes mellitus with ketoacidosis with coma
        ]
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
# Set to None to include all conditions:
# CONDITION_CODES = None

# Updated measurement types for T1D with added clinical measurements
MEASUREMENT_TYPES = {
    # Clinical Measurements
    'height': {
        'keywords': [
            'height', 'body height', 'stature', 'standing height', 'ht',
            'patient height', 'measured height', 'height measurement'
        ],
        'must_have': [],
        'exclude': ['weight', 'sitting', 'fundal']
    },
    
    'weight': {
        'keywords': [
            'weight', 'body weight', 'wt', 'body mass', 'patient weight',
            'measured weight', 'weight measurement', 'actual weight', 'bw'
        ],
        'must_have': [],
        'exclude': ['height', 'birth', 'ideal', 'target', 'gain', 'loss']
    },
    
    'waist_circumference': {
        'keywords': [
            'waist circumference', 'waist', 'abdominal circumference',
            'waist measurement', 'waist circ', 'abdominal girth',
            'waist size', 'wc'
        ],
        'must_have': [],
        'exclude': ['hip', 'chest', 'head', 'arm']
    },
    
    'bmi': {
        'keywords': [
            'bmi', 'body mass index', 'body-mass index', 'bodymass index',
            'quetelet index', 'bmi value', 'calculated bmi'
        ],
        'must_have': [],
        'exclude': ['percentile', 'z-score', 'zscore']
    },
    
    'bmi_percentile': {
        'keywords': [
            'bmi percentile', 'body mass index percentile', 'bmi %',
            'bmi %ile', 'bmi percentage', 'pediatric bmi percentile',
            'bmi for age percentile'
        ],
        'must_have': ['percentile', '%'],
        'exclude': ['z-score', 'zscore', 'adult']
    },
    
    # Blood pressure measurements (keeping existing)
    'systolic_blood_pressure': {
        'keywords': [
            'systolic', 'systolic blood pressure', 'systolic bp', 'sbp',
            'systolic pressure', 'sys bp', 'blood pressure systolic'
        ],
        'must_have': [],
        'exclude': ['diastolic', 'mean']
    },
    
    'diastolic_blood_pressure': {
        'keywords': [
            'diastolic', 'diastolic blood pressure', 'diastolic bp', 'dbp',
            'diastolic pressure', 'dias bp', 'blood pressure diastolic'
        ],
        'must_have': [],
        'exclude': ['systolic', 'mean']
    },
    
    # Glucose-related measurements (keeping existing)
    'hba1c': {
        'keywords': [
            'hba1c', 'hemoglobin a1c', 'a1c', 'glycosylated hemoglobin',
            'glycated hemoglobin', 'labhba1c', 'eag', 'hgb a1c', 'hb a1c',
            'glycohemoglobin', 'diabetic control', 'glucose control'
        ],
        'must_have': [],
        'exclude': ['mean', 'estimated', 'average']
    },
    
    'fasting_glucose': {
        'keywords': [
            'fasting', 'fbs', 'fpg', 'fasting glucose', 'fasting blood sugar',
            'fasting plasma glucose', 'glucose fasting', 'fasting blood glucose'
        ],
        'must_have': ['glucose', 'sugar', 'gluc'],
        'exclude': ['ogtt', 'tolerance', 'random', 'postprandial']
    },
    
    'serum_glucose': {
        'keywords': [
            'serum glucose', 'glucose', 'blood glucose', 'plasma glucose',
            'random glucose', 'glucose serum', 'gluc', 'bg', 'blood sugar'
        ],
        'must_have': [],
        'exclude': ['fasting', 'ogtt', 'tolerance', 'urine', 'csf']
    },
    
    'glucose_2h_ogtt': {
        'keywords': [
            '2-h glucose', '2 hour glucose', '2h glucose', 'ogtt',
            'oral glucose tolerance test', 'glucose tolerance test',
            '2-hour post glucose', '2hr glucose', 'glucose 2 hour'
        ],
        'must_have': ['glucose', 'gluc'],
        'exclude': ['fasting', 'baseline']
    },
    
    # C-peptide measurements (important for T1D)
    'c_peptide': {
        'keywords': [
            'c-peptide', 'c peptide', 'cpeptide', 'connecting peptide',
            'serum c peptide', 'plasma c-peptide', 'c peptide serum'
        ],
        'must_have': [],
        'exclude': ['urine', 'brain', 'bnp', 'natriuretic']
    },
    
    # Islet autoantibodies (critical for T1D diagnosis)
    'gad65_antibody': {
        'keywords': [
            'gad65', 'gad65 antibody', 'anti-gad65', 'anti gad65',
            'glutamic acid decarboxylase 65', 'gad antibody',
            'gada', 'anti-gad', 'gad autoantibody', 'gad-65'
        ],
        'must_have': [],
        'exclude': []
    },
    
    'ica512_antibody': {
        'keywords': [
            'ica512', 'ica512 antibody', 'anti-ica512', 'anti ica512',
            'ia-2', 'ia2', 'ia-2 antibody', 'insulinoma antigen 2',
            'islet cell antibody 512', 'anti-ia2', 'ica-512'
        ],
        'must_have': [],
        'exclude': []
    },
    
    'insulin_antibody': {
        'keywords': [
            'insulin antibody', 'anti-insulin', 'anti insulin',
            'iaa', 'insulin autoantibody', 'anti-insulin antibody'
        ],
        'must_have': [],
        'exclude': ['receptor', 'binding']
    },
    
    'znt8_antibody': {
        'keywords': [
            'znt8', 'znt8 antibody', 'anti-znt8', 'anti znt8',
            'zinc transporter 8', 'zinc transporter 8 antibody',
            'znt8a', 'anti-znt8 antibody'
        ],
        'must_have': [],
        'exclude': []
    },
    
    # Lipid panel
    'hdl': {
        'keywords': [
            'hdl', 'hdl cholesterol', 'hdl-c', 'high density lipoprotein',
            'good cholesterol', 'alpha lipoprotein', 'hdl chol'
        ],
        'must_have': [],
        'exclude': ['ratio', 'total', 'non-hdl', 'vldl']
    },
    
    'ldl': {
        'keywords': [
            'ldl', 'ldl cholesterol', 'ldl-c', 'low density lipoprotein',
            'bad cholesterol', 'beta lipoprotein', 'ldl direct', 'ldl calculated'
        ],
        'must_have': [],
        'exclude': ['ratio', 'total', 'vldl', 'oxidized']
    },
    
    'triglycerides': {
        'keywords': [
            'triglycerides', 'triglyceride', 'trig', 'trigs', 'tg',
            'serum triglycerides', 'plasma triglycerides'
        ],
        'must_have': [],
        'exclude': ['ratio']
    },
    
    # Liver function tests
    'alt': {
        'keywords': [
            'alt', 'alanine aminotransferase', 'sgpt', 'alanine transaminase',
            'serum alt', 'liver alt', 'alt liver enzyme', 'gpt'
        ],
        'must_have': [],
        'exclude': ['ratio', 'ast/alt']
    },
    
    'ast': {
        'keywords': [
            'ast', 'aspartate aminotransferase', 'sgot', 'aspartate transaminase',
            'serum ast', 'liver ast', 'ast liver enzyme', 'got'
        ],
        'must_have': [],
        'exclude': ['ratio', 'ast/alt', 'diastolic']
    },
    
    # Kidney function tests
    'bun': {
        'keywords': [
            'bun', 'blood urea nitrogen', 'urea nitrogen', 'serum urea',
            'blood urea', 'urea', 'serum bun', 'plasma urea'
        ],
        'must_have': [],
        'exclude': ['ratio', 'bun/creatinine']
    },
    
    'creatinine': {
        'keywords': [
            'creatinine', 'serum creatinine', 'creat', 'cr', 'scr',
            'plasma creatinine', 'blood creatinine'
        ],
        'must_have': [],
        'exclude': ['urine', 'clearance', 'ratio', 'kinase']
    },
    
    'egfr': {
        'keywords': [
            'egfr', 'estimated gfr', 'estimated glomerular filtration rate',
            'gfr', 'glomerular filtration rate', 'kidney function',
            'egfr mdrd', 'egfr ckd-epi', 'calculated gfr'
        ],
        'must_have': [],
        'exclude': []
    },
    
    # Urine tests
    'urine_microalbumin': {
        'keywords': [
            'urine microalbumin', 'microalbumin', 'microalbuminuria',
            'urinary microalbumin', 'albumin urine', 'urine albumin'
        ],
        'must_have': [],
        'exclude': ['ratio', 'acr', 'uacr']
    },
    
    'urine_creatinine': {
        'keywords': [
            'urine creatinine', 'creatinine urine', 'urinary creatinine',
            'urine creat', '24hr creatinine', 'random urine creatinine'
        ],
        'must_have': ['urine', 'urinary'],
        'exclude': ['ratio', 'albumin', 'serum', 'plasma']
    },
    
    'urine_microalbumin_creatinine_ratio': {
        'keywords': [
            'acr', 'uacr', 'albumin creatinine ratio', 'microalbumin creatinine ratio',
            'albumin/creatinine', 'microalbumin/creatinine', 'alb/cr ratio'
        ],
        'must_have': ['ratio'],
        'exclude': []
    },
    
    'urine_ketone': {
        'keywords': [
            'urine ketone', 'ketones', 'urine ketones', 'urinary ketones',
            'ketone bodies', 'acetoacetate', 'beta-hydroxybutyrate'
        ],
        'must_have': [],
        'exclude': ['serum', 'blood', 'plasma']
    },
    
    # CGM-specific measurement for hypoglycemia risk
    'lbgi': {
        'keywords': [
            'lbgi', 'low blood glucose index', 'low glucose index',
            'hypoglycemia risk index', 'low bg index', 'glucose variability index'
        ],
        'must_have': [],
        'exclude': ['high', 'hbgi']
    },
    
    # Additional T1D relevant measurements
    'time_in_range': {
        'keywords': [
            'time in range', 'tir', 'percent in range', '% in range',
            'glucose time in range', 'cgm time in range', 'time in target'
        ],
        'must_have': [],
        'exclude': ['time below', 'time above']
    },
    
    'glucose_variability': {
        'keywords': [
            'glucose variability', 'glycemic variability', 'cv', 'coefficient of variation',
            'glucose cv', 'standard deviation glucose', 'glucose sd', 'mage',
            'mean amplitude glycemic excursion'
        ],
        'must_have': [],
        'exclude': []
    },
    'free_t3': {
    'keywords': [
        'free t3', 'free triiodothyronine', 'ft3', 'free t-3', 'triiodothyronine free',
        'free tri-iodothyronine', 't3 free', 'thyroid hormone free t3',
        'unbound t3', 'f-t3'
    ],
    'must_have': [],
    'exclude': ['total', 'bound', 'reverse', 'rt3']
},

'glucose_2h_postprandial': {
    'keywords': [
        '2 hour glucose', '2h glucose postprandial', '2hr postprandial glucose',
        'glucose 2 hours', 'postprandial glucose 2h', '2 hour post meal glucose',
        'glucose 2hr post', '2h post glucose', 'glucose 120 min', 
        'glucose 2 hours after meal', 'post meal glucose 2 hour'
    ],
    'must_have': ['glucose', 'gluc'],
    'exclude': ['ogtt', 'tolerance', 'fasting']
},

'insulin_autoantibody': {
    'keywords': [
        'insulin autoantibody', 'insulin ab', 'insulin antibodies',
        'anti-insulin autoantibody', 'insulin autoantibodies',
        'iaa insulin', 'auto insulin antibody', 'insulin ab test'
    ],
    'must_have': [],
    'exclude': ['receptor', 'binding', 'c-peptide']
}
}

# Main diagnosis codes for Type 1 Diabetes
MAIN_DIAGNOSIS = {
    'ICD9': [
        '250.01',  # Type 1 diabetes mellitus without mention of complication
        '250.03',  # Type 1 diabetes mellitus without mention of complication, uncontrolled
        '250.11',  # Type 1 diabetes mellitus with ketoacidosis
        '250.13',  # Type 1 diabetes mellitus with ketoacidosis, uncontrolled
        '250.21',  # Type 1 diabetes mellitus with hyperosmolarity
        '250.31',  # Type 1 diabetes mellitus with other coma
        '250.41',  # Type 1 diabetes mellitus with renal manifestations
        '250.51',  # Type 1 diabetes mellitus with ophthalmic manifestations
        '250.61',  # Type 1 diabetes mellitus with neurological manifestations
        '250.71',  # Type 1 diabetes mellitus with peripheral circulatory disorders
        '250.81',  # Type 1 diabetes mellitus with other specified manifestations
        '250.91'   # Type 1 diabetes mellitus with unspecified complication
    ],
    'ICD10': [
        'E10.',     # Type 1 diabetes mellitus (all subcodes)
        'E10.0',    # Type 1 diabetes mellitus with hyperosmolarity
        'E10.1',    # Type 1 diabetes mellitus with ketoacidosis
        'E10.2',    # Type 1 diabetes mellitus with kidney complications
        'E10.3',    # Type 1 diabetes mellitus with ophthalmic complications
        'E10.4',    # Type 1 diabetes mellitus with neurological complications
        'E10.5',    # Type 1 diabetes mellitus with circulatory complications
        'E10.6',    # Type 1 diabetes mellitus with other specified complications
        'E10.7',    # Type 1 diabetes mellitus with multiple complications
        'E10.8',    # Type 1 diabetes mellitus with unspecified complications
        'E10.9'     # Type 1 diabetes mellitus without complications
    ]
}

# Set to None to include all measurements:
# MEASUREMENT_TYPES = None
# Set to None to include all measurements:
# MEASUREMENT_TYPES = None

# Example: Demographics filter (optional)
DEMOGRAPHICS_FILTER = {
    'age_min': 0,
    'age_max': 20,
    #'gender': ['Male', 'Female'],  # or use concept IDs: [8507, 8532]
    # 'race': ['White', 'Black', 'Asian']  # optional
}
def read_s3_csv(bucket, key, chunksize=None):
    """Read CSV from S3, optionally in chunks"""
    s3 = boto3.client('s3')
    
    if chunksize:
        # For large files, read in chunks
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj['Body'], chunksize=chunksize)
    else:
        # For smaller files, read entire file
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj['Body'])

def analyze_demographics():
    """Analyze demographics/person.csv"""
    print("\n" + "="*60)
    print("DEMOGRAPHICS ANALYSIS")
    print("="*60)
    
    try:
        # Read demographics file
        df = read_s3_csv(S3_BUCKET, f'{S3_PREFIX}demographics/person.csv')
        
        # Print column names
        print("\nColumn Names:")
        print(", ".join(df.columns.tolist()))
        
        # Basic statistics
        print(f"Total patients: {len(df):,}")
        
        # Race count - using uppercase column names
        if 'RACE_CONCEPT_ID' in df.columns or 'RACE_SOURCE_VALUE' in df.columns:
            race_col = 'RACE_SOURCE_VALUE' if 'RACE_SOURCE_VALUE' in df.columns else 'RACE_CONCEPT_ID'
            print("\nRace Distribution:")
            race_counts = df[race_col].value_counts()
            for race, count in race_counts.head(10).items():
                print(f"  {race}: {count:,} ({count/len(df)*100:.2f}%)")
        
        # Gender count - using uppercase column names
        if 'GENDER_CONCEPT_ID' in df.columns or 'GENDER_SOURCE_VALUE' in df.columns:
            gender_col = 'GENDER_SOURCE_VALUE' if 'GENDER_SOURCE_VALUE' in df.columns else 'GENDER_CONCEPT_ID'
            print("\nGender Distribution:")
            gender_counts = df[gender_col].value_counts()
            for gender, count in gender_counts.items():
                print(f"  {gender}: {count:,} ({count/len(df)*100:.2f}%)")
        
        # Ethnicity count - using uppercase column names
        if 'ETHNICITY_CONCEPT_ID' in df.columns or 'ETHNICITY_SOURCE_VALUE' in df.columns:
            ethnicity_col = 'ETHNICITY_SOURCE_VALUE' if 'ETHNICITY_SOURCE_VALUE' in df.columns else 'ETHNICITY_CONCEPT_ID'
            print("\nEthnicity Distribution:")
            ethnicity_counts = df[ethnicity_col].value_counts()
            for ethnicity, count in ethnicity_counts.head(10).items():
                print(f"  {ethnicity}: {count:,} ({count/len(df)*100:.2f}%)")
        
        # Store person_ids for later use - using uppercase column name
        person_ids = set(df['PERSON_ID'].unique()) if 'PERSON_ID' in df.columns else set()
        
        # Clean up
        del df
        gc.collect()
        
        return person_ids
        
    except Exception as e:
        print(f"Error analyzing demographics: {e}")
        return set()

def analyze_icd_codes(person_ids=None, chunk_size=100000):
    """Analyze ICD codes from condition_occurrence.csv in batches"""
    print("\n" + "="*60)
    print("ICD CODES ANALYSIS")
    print("="*60)
    
    try:
        # Prepare ICD code lists for filtering
        all_icd_codes = []
        for condition, codes in CONDITION_CODES.items():
            all_icd_codes.extend(codes.get('ICD9', []))
            all_icd_codes.extend(codes.get('ICD10', []))
        
        # Initialize counters
        icd_encounter_counts = {}
        icd_patient_counts = {}
        filtered_conditions = []
        total_rows = 0
        
        print(f"\nProcessing condition_occurrence.csv in chunks of {chunk_size:,} rows...")
        
        # Process file in chunks
        chunks = read_s3_csv(S3_BUCKET, f'{S3_PREFIX}icd_codes/condition_occurrence.csv', 
                           chunksize=chunk_size)
        
        total_unique_patients = set()

        for i, chunk in enumerate(chunks):
            total_rows += len(chunk)
            print(f"  Processing chunk {i+1} ({total_rows:,} rows processed)...", end='\r')
            
            # Filter by person_ids if provided - using uppercase column name
            if person_ids and 'PERSON_ID' in chunk.columns:
                chunk = chunk[chunk['PERSON_ID'].isin(person_ids)]
            
            # Identify ICD code column - check for uppercase variations
            icd_col = None
            for col in ['CONDITION_SOURCE_VALUE', 'Condition_Source_Value', 'condition_source_value']:
                if col in chunk.columns:
                    icd_col = col
                    break
            
            if icd_col:
                # Extract ICD codes from the pipe-separated format
                # Format is: "description | icd_code"
                chunk['extracted_icd'] = chunk[icd_col].apply(
                    lambda x: x.split('|')[1].strip() if pd.notna(x) and '|' in x else x if pd.notna(x) else None
                )
                
                # Count ICD codes per encounter
                for code in chunk['extracted_icd'].value_counts().index:
                    if pd.notna(code):
                        icd_encounter_counts[code] = icd_encounter_counts.get(code, 0) + \
                                                    chunk[chunk['extracted_icd'] == code].shape[0]
                
                # Count unique patients per ICD code
                if 'PERSON_ID' in chunk.columns:
                    total_unique_patients.update(chunk['PERSON_ID'].unique())
                    for code in chunk['extracted_icd'].unique():
                        if pd.notna(code):
                            patients = chunk[chunk['extracted_icd'] == code]['PERSON_ID'].nunique()
                            if code in icd_patient_counts:
                                icd_patient_counts[code] = max(icd_patient_counts[code], patients)
                            else:
                                icd_patient_counts[code] = patients
                
                # Filter for specific conditions
                for code in all_icd_codes:
                    # Check for exact match or prefix match
                    mask = (chunk['extracted_icd'] == code) | (chunk['extracted_icd'].str.startswith(code, na=False))
                    filtered = chunk[mask]
                    if not filtered.empty:
                        filtered_conditions.append(filtered)
        
        print(f"\n\nTotal rows processed: {total_rows:,}")
        
        # Print top 15 ICD codes by encounter count
        print("\nTop 15 ICD Codes by Encounter Count:")
        sorted_encounters = sorted(icd_encounter_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        for code, count in sorted_encounters:
            print(f"  {code}: {count:,} encounters")
        
        print("\nTop 15 ICD Codes by Patient Count:")
        print(f"Total unique patients: {len(total_unique_patients):,}")
        sorted_patients = sorted(icd_patient_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        for code, count in sorted_patients:
            if len(total_unique_patients) > 0:
                percentage = (count / len(total_unique_patients)) * 100
            else:
                percentage = 0
            print(f"  {code}: {count:,} patients ({percentage:.2f}%)")
        
        # Modify the filtered conditions summary:
        if filtered_conditions:
            filtered_df = pd.concat(filtered_conditions, ignore_index=True)
            print(f"\nFiltered Conditions Summary:")
            print(f"  Total filtered records: {len(filtered_df):,}")
            if 'PERSON_ID' in filtered_df.columns:
                unique_with_condition = filtered_df['PERSON_ID'].nunique()
                if len(total_unique_patients) > 0:
                    percentage = (unique_with_condition / len(total_unique_patients)) * 100
                else:
                    percentage = 0
                print(f"  Unique patients with target conditions: {unique_with_condition:,} ({percentage:.2f}%)")
            del filtered_df
        
        # Clean up
        gc.collect()
        
    except Exception as e:
        print(f"\nError analyzing ICD codes: {e}")


def analyze_measurements(person_ids=None, chunk_size=100000):
    """Analyze measurements from measurement.csv in batches"""
    print("\n" + "="*60)
    print("MEASUREMENTS ANALYSIS")
    print("="*60)
    
    try:
        # Initialize counters
        measurement_counts = {}
        patient_measurement_presence = {}
        total_rows = 0
        unique_patients = set()
        
        print(f"\nProcessing measurement.csv in chunks of {chunk_size:,} rows...")
        
        # Process file in chunks
        chunks = read_s3_csv(S3_BUCKET, f'{S3_PREFIX}measurements/measurement.csv', 
                           chunksize=chunk_size)
        
        for i, chunk in enumerate(chunks):
            total_rows += len(chunk)
            print(f"  Processing chunk {i+1} ({total_rows:,} rows processed)...", end='\r')
            
            # Reset index to ensure proper alignment
            chunk = chunk.reset_index(drop=True)
            
            # Filter by person_ids if provided
            if person_ids and 'PERSON_ID' in chunk.columns:
                chunk = chunk[chunk['PERSON_ID'].isin(person_ids)]
                # Reset index again after filtering
                chunk = chunk.reset_index(drop=True)
            
            # Track unique patients
            if 'PERSON_ID' in chunk.columns:
                unique_patients.update(chunk['PERSON_ID'].unique())
            
            # Use MEASUREMENT_SOURCE_VALUE column specifically
            if 'MEASUREMENT_SOURCE_VALUE' not in chunk.columns:
                print(f"\nWarning: MEASUREMENT_SOURCE_VALUE column not found. Available columns: {chunk.columns.tolist()}")
                continue
            
            # Count measurements
            for measure in chunk['MEASUREMENT_SOURCE_VALUE'].value_counts().index:
                if pd.notna(measure):
                    measurement_counts[measure] = measurement_counts.get(measure, 0) + \
                                                 chunk[chunk['MEASUREMENT_SOURCE_VALUE'] == measure].shape[0]
            
            # Track which patients have which measurements using regex
            if 'PERSON_ID' in chunk.columns:
                for measure_type, config in MEASUREMENT_TYPES.items():
                    # Initialize mask with proper index alignment
                    keyword_mask = pd.Series([False] * len(chunk), index=chunk.index)
                    
                    # Check for keyword matches using regex for flexible matching
                    for keyword in config['keywords']:
                        # Create case-insensitive regex pattern with word boundaries
                        pattern = r'(?i)\b' + re.escape(keyword) + r'\b'
                        temp_mask = chunk['MEASUREMENT_SOURCE_VALUE'].str.contains(
                            pattern, regex=True, na=False, case=False
                        )
                        # Ensure alignment
                        keyword_mask = keyword_mask | temp_mask.reindex(keyword_mask.index, fill_value=False)
                    
                    # Apply must_have filters with regex
                    if config['must_have']:
                        must_have_mask = pd.Series([False] * len(chunk), index=chunk.index)
                        for must in config['must_have']:
                            pattern = r'(?i)\b' + re.escape(must) + r'\b'
                            temp_mask = chunk['MEASUREMENT_SOURCE_VALUE'].str.contains(
                                pattern, regex=True, na=False, case=False
                            )
                            must_have_mask = must_have_mask | temp_mask.reindex(must_have_mask.index, fill_value=False)
                        keyword_mask = keyword_mask & must_have_mask
                    
                    # Apply exclusions with regex
                    for exclude in config['exclude']:
                        pattern = r'(?i)\b' + re.escape(exclude) + r'\b'
                        exclude_mask = chunk['MEASUREMENT_SOURCE_VALUE'].str.contains(
                            pattern, regex=True, na=False, case=False
                        )
                        keyword_mask = keyword_mask & ~exclude_mask.reindex(keyword_mask.index, fill_value=False)
                    
                    # Track patients with this measurement
                    if keyword_mask.any():
                        # Use loc to ensure proper indexing
                        patients_with_measure = chunk.loc[keyword_mask, 'PERSON_ID'].unique()
                        if measure_type not in patient_measurement_presence:
                            patient_measurement_presence[measure_type] = set()
                        patient_measurement_presence[measure_type].update(patients_with_measure)
        
        print(f"\n\nTotal rows processed: {total_rows:,}")
        print(f"Total unique patients: {len(unique_patients):,}")
        
        # Print top 15 measurements
        print("\nTop 15 Measurements by Count:")
        sorted_measurements = sorted(measurement_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        for measure, count in sorted_measurements:
            # Truncate long measurement names for display
            display_name = measure[:60] + "..." if len(str(measure)) > 60 else measure
            print(f"  {display_name}: {count:,}")
        
        # Calculate missing measurement percentages
        print("\nMeasurement Coverage (% of patients with measurement):")
        for measure_type in sorted(patient_measurement_presence.keys()):
            patients_with = patient_measurement_presence[measure_type]
            if unique_patients:
                percentage = (len(patients_with) / len(unique_patients)) * 100
                missing_percentage = 100 - percentage
                print(f"  {measure_type}:")
                print(f"    Patients with measurement: {len(patients_with):,} ({percentage:.2f}%)")
                print(f"    Patients missing measurement: {len(unique_patients) - len(patients_with):,} ({missing_percentage:.2f}%)")
        
        # Clean up
        gc.collect()
        
    except Exception as e:
        print(f"\nError analyzing measurements: {e}")
        import traceback
        traceback.print_exc()

def analyze_medications(person_ids=None, chunk_size=100000):
    """Analyze medications (assuming there's a medications folder with relevant CSV)"""
    print("\n" + "="*60)
    print("MEDICATIONS ANALYSIS")
    print("="*60)
    
    try:
        # Check if medications folder exists and get the CSV file
        s3 = boto3.client('s3')
        
        # Try to find medications file
        medication_file = None
        try:
            response = s3.list_objects_v2(
                Bucket=S3_BUCKET,
                Prefix=f'{S3_PREFIX}medications/'
            )
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('.csv'):
                        medication_file = obj['Key']
                        break
        except:
            print("No medications folder found. Skipping medication analysis.")
            return
        
        if not medication_file:
            print("No medication CSV file found. Skipping medication analysis.")
            return
        
        # Initialize counters
        med_encounter_counts = {}
        med_patient_counts = {}
        drug_class_counts = {}
        drug_class_patients = {drug_class: set() for drug_class in DRUG_CLASSES.keys()}
        total_rows = 0
        
        print(f"\nProcessing medications in chunks of {chunk_size:,} rows...")
        
        # Process file in chunks
        chunks = read_s3_csv(S3_BUCKET, medication_file, chunksize=chunk_size)
        
        total_unique_patients = set()

        for i, chunk in enumerate(chunks):
            total_rows += len(chunk)
            print(f"  Processing chunk {i+1} ({total_rows:,} rows processed)...", end='\r')
            
            # Filter by person_ids if provided
            if person_ids and 'PERSON_ID' in chunk.columns:
                chunk = chunk[chunk['PERSON_ID'].isin(person_ids)]
            
            # Use DRUG_SOURCE_VALUE column specifically
            if 'DRUG_SOURCE_VALUE' not in chunk.columns:
                print(f"\nWarning: DRUG_SOURCE_VALUE column not found. Available columns: {chunk.columns.tolist()}")
                continue
            
            # Count medications per encounter
            for med in chunk['DRUG_SOURCE_VALUE'].value_counts().index:
                if pd.notna(med):
                    med_encounter_counts[med] = med_encounter_counts.get(med, 0) + \
                                               chunk[chunk['DRUG_SOURCE_VALUE'] == med].shape[0]
            
            # Count unique patients per medication
            if 'PERSON_ID' in chunk.columns:
                total_unique_patients.update(chunk['PERSON_ID'].unique())
                for med in chunk['DRUG_SOURCE_VALUE'].unique():
                    if pd.notna(med):
                        patients = chunk[chunk['DRUG_SOURCE_VALUE'] == med]['PERSON_ID'].nunique()
                        if med in med_patient_counts:
                            med_patient_counts[med] = max(med_patient_counts[med], patients)
                        else:
                            med_patient_counts[med] = patients
            
            # Classify drugs using regex for flexible matching
            for drug_class, drugs in DRUG_CLASSES.items():
                # Reset index to ensure alignment
                chunk_reset = chunk.reset_index(drop=True)
                class_mask = pd.Series([False] * len(chunk_reset), index=chunk_reset.index)
                
                for drug in drugs:
                    # Create case-insensitive regex pattern with word boundaries
                    # This allows for "metformin" to match "Metformin HCL 500mg" etc.
                    pattern = r'(?i)\b' + re.escape(drug) + r'\b'
                    drug_mask = chunk_reset['DRUG_SOURCE_VALUE'].str.contains(
                        pattern, regex=True, na=False, case=False
                    )
                    # Ensure the mask has the same index as class_mask
                    class_mask = class_mask | drug_mask
                
                # Count prescriptions for this drug class using loc for proper indexing
                class_count = chunk_reset.loc[class_mask].shape[0]
                drug_class_counts[drug_class] = drug_class_counts.get(drug_class, 0) + class_count
                
                # Track unique patients for this drug class
                if 'PERSON_ID' in chunk_reset.columns and class_mask.any():
                    patients_in_class = chunk_reset.loc[class_mask, 'PERSON_ID'].unique()
                    drug_class_patients[drug_class].update(patients_in_class)
        
        print(f"\n\nTotal rows processed: {total_rows:,}")
        
        # Print top 15 medications by encounter count
        print("\nTop 15 Medications by Encounter Count:")
        sorted_encounters = sorted(med_encounter_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        for med, count in sorted_encounters:
            # Truncate long medication names for display
            display_name = med[:60] + "..." if len(str(med)) > 60 else med
            print(f"  {display_name}: {count:,} encounters")
        
        # Print top 15 medications by patient count
        print("\nTop 15 Medications by Patient Count:")
        sorted_patients = sorted(med_patient_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        for med, count in sorted_patients:
            display_name = med[:60] + "..." if len(str(med)) > 60 else med
            print(f"  {med}: {count:,} patients")
        
        # Print drug class summary with both prescription counts and patient counts
        print("\nDrug Class Summary:")
        print(f"Total unique patients with any medication: {len(total_unique_patients):,}")
        for drug_class in sorted(drug_class_counts.keys(), key=lambda x: drug_class_counts[x], reverse=True):
            prescription_count = drug_class_counts[drug_class]
            patient_count = len(drug_class_patients[drug_class])
            # Calculate percentage
            if len(total_unique_patients) > 0:
                percentage = (patient_count / len(total_unique_patients)) * 100
            else:
                percentage = 0
            print(f"  {drug_class}:")
            print(f"    Prescriptions: {prescription_count:,}")
            print(f"    Unique patients: {patient_count:,} ({percentage:.2f}%)")
        # Clean up
        gc.collect()
        
    except Exception as e:
        print(f"\nError analyzing medications: {e}")


def calculate_age_at_diagnosis(demographics_df, chunk_size=100000):
    """Calculate age at diagnosis based on first occurrence of target ICD codes"""
    print("\n" + "="*60)
    print("CALCULATING AGE AT DIAGNOSIS")
    print("="*60)
    
    try:
        # Prepare target ICD codes from MAIN_DIAGNOSIS
        target_icd_codes = []
        target_icd_codes.extend(MAIN_DIAGNOSIS.get('ICD9', []))
        target_icd_codes.extend(MAIN_DIAGNOSIS.get('ICD10', []))
        
        # Dictionary to store first diagnosis date per patient
        first_diagnosis_dates = {}
        
        print(f"\nProcessing condition_occurrence.csv for diagnosis dates...")
        
        # Process condition file in chunks
        chunks = read_s3_csv(S3_BUCKET, f'{S3_PREFIX}icd_codes/condition_occurrence.csv', 
                           chunksize=chunk_size)
        
        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}...", end='\r')
            
            # Identify ICD code column
            icd_col = None
            for col in ['CONDITION_SOURCE_VALUE', 'Condition_Source_Value', 'condition_source_value']:
                if col in chunk.columns:
                    icd_col = col
                    break
            
            if not icd_col:
                continue
            
            # Extract ICD codes
            chunk['extracted_icd'] = chunk[icd_col].apply(
                lambda x: x.split('|')[1].strip() if pd.notna(x) and '|' in x else x if pd.notna(x) else None
            )
            
            # Find date column
            date_col = None
            for col in ['CONDITION_START_DATE', 'Condition_Start_Date', 'condition_start_date']:
                if col in chunk.columns:
                    date_col = col
                    break
            
            if not date_col:
                print("\nWarning: No date column found in condition_occurrence.csv")
                continue
            
            # Convert date column to datetime
            chunk[date_col] = pd.to_datetime(chunk[date_col], errors='coerce')
            
            # Filter for target ICD codes
            for code in target_icd_codes:
                mask = (chunk['extracted_icd'] == code) | (chunk['extracted_icd'].str.startswith(code, na=False))
                filtered = chunk[mask]
                
                if not filtered.empty and 'PERSON_ID' in filtered.columns:
                    # Group by person and get minimum date
                    person_dates = filtered.groupby('PERSON_ID')[date_col].min()
                    
                    # Update first_diagnosis_dates with earlier dates
                    for person_id, date in person_dates.items():
                        if pd.notna(date):
                            if person_id not in first_diagnosis_dates or date < first_diagnosis_dates[person_id]:
                                first_diagnosis_dates[person_id] = date
        
        print(f"\n\nFound diagnosis dates for {len(first_diagnosis_dates):,} patients")
        
        # Calculate age at diagnosis
        # Ensure PERSON_ID is in demographics_df
        if 'PERSON_ID' not in demographics_df.columns:
            print("Error: PERSON_ID not found in demographics dataframe")
            return demographics_df
        
        # Find birth date column
        birth_col = None
        for col in ['BIRTH_DATETIME', 'Birth_Datetime', 'birth_datetime', 'BIRTH_DATE']:
            if col in demographics_df.columns:
                birth_col = col
                break
        
        if not birth_col:
            print("Warning: No birth date column found in demographics")
            return demographics_df
        
        # Convert birth dates to datetime
        demographics_df[birth_col] = pd.to_datetime(demographics_df[birth_col], errors='coerce')
        
        # Calculate age at diagnosis
        demographics_df['age_at_diagnosis'] = demographics_df.apply(
            lambda row: (first_diagnosis_dates[row['PERSON_ID']] - row[birth_col]).days / 365.25 
            if row['PERSON_ID'] in first_diagnosis_dates and pd.notna(row[birth_col]) 
            else None, axis=1
        )
        
        # Print summary statistics
        print("\nAge at Diagnosis Statistics:")
        if 'age_at_diagnosis' in demographics_df.columns:
            valid_ages = demographics_df['age_at_diagnosis'].dropna()
            if not valid_ages.empty:
                print(f"  Patients with diagnosis: {len(valid_ages):,}")
                print(f"  Mean age: {valid_ages.mean():.2f} years")
                print(f"  Median age: {valid_ages.median():.2f} years")
                print(f"  Min age: {valid_ages.min():.2f} years")
                print(f"  Max age: {valid_ages.max():.2f} years")
        
        return demographics_df
        
    except Exception as e:
        print(f"Error calculating age at diagnosis: {e}")
        return demographics_df

def apply_demographics_filter(demographics_df, filter_config=None):
    """Apply demographics filters to the dataframe"""
    print("\n" + "="*60)
    print("APPLYING DEMOGRAPHICS FILTERS")
    print("="*60)
    
    if filter_config is None:
        filter_config = DEMOGRAPHICS_FILTER
    
    if not filter_config:
        print("No filters to apply")
        return demographics_df
    
    original_count = len(demographics_df)
    filtered_df = demographics_df.copy()
    
    # Filter by age at diagnosis
    if 'age_min' in filter_config and 'age_max' in filter_config:
        if 'age_at_diagnosis' in filtered_df.columns:
            age_min = filter_config['age_min']
            age_max = filter_config['age_max']
            before_filter = len(filtered_df)
            filtered_df = filtered_df[
                (filtered_df['age_at_diagnosis'] >= age_min) & 
                (filtered_df['age_at_diagnosis'] <= age_max)
            ]
            after_filter = len(filtered_df)
            print(f"Age at diagnosis filter ({age_min}-{age_max} years):")
            print(f"  Before: {before_filter:,} patients")
            print(f"  After: {after_filter:,} patients")
            print(f"  Removed: {before_filter - after_filter:,} patients")
        else:
            print("Warning: age_at_diagnosis column not found, skipping age filter")
    
    # Filter by gender if specified
    if 'gender' in filter_config:
        gender_col = None
        for col in ['GENDER_SOURCE_VALUE', 'GENDER_CONCEPT_ID']:
            if col in filtered_df.columns:
                gender_col = col
                break
        
        if gender_col:
            before_filter = len(filtered_df)
            filtered_df = filtered_df[filtered_df[gender_col].isin(filter_config['gender'])]
            after_filter = len(filtered_df)
            print(f"Gender filter ({filter_config['gender']}):")
            print(f"  Before: {before_filter:,} patients")
            print(f"  After: {after_filter:,} patients")
            print(f"  Removed: {before_filter - after_filter:,} patients")
    
    # Filter by race if specified
    if 'race' in filter_config:
        race_col = None
        for col in ['RACE_SOURCE_VALUE', 'RACE_CONCEPT_ID']:
            if col in filtered_df.columns:
                race_col = col
                break
        
        if race_col:
            before_filter = len(filtered_df)
            filtered_df = filtered_df[filtered_df[race_col].isin(filter_config['race'])]
            after_filter = len(filtered_df)
            print(f"Race filter ({filter_config['race']}):")
            print(f"  Before: {before_filter:,} patients")
            print(f"  After: {after_filter:,} patients")
            print(f"  Removed: {before_filter - after_filter:,} patients")
    
    # Summary
    print(f"\nFilter Summary:")
    print(f"  Original patients: {original_count:,}")
    print(f"  Filtered patients: {len(filtered_df):,}")
    print(f"  Total removed: {original_count - len(filtered_df):,}")
    print(f"  Retention rate: {(len(filtered_df)/original_count)*100:.2f}%")
    
    return filtered_df

def main():
    """Main execution function with preprocessing"""
    print("Starting OMOP Data Analysis Pipeline")
    print("="*60)
    
    # Step 1: Load demographics
    print("\n" + "="*60)
    print("LOADING DEMOGRAPHICS")
    print("="*60)
    demographics_df = read_s3_csv(S3_BUCKET, f'{S3_PREFIX}demographics/person.csv')
    print(f"Loaded {len(demographics_df):,} patients")
    
    # Step 2: Calculate age at diagnosis (preprocessing)
    demographics_df = calculate_age_at_diagnosis(demographics_df)
    
    # Step 3: Apply demographics filters (preprocessing)
    demographics_df = apply_demographics_filter(demographics_df, DEMOGRAPHICS_FILTER)
    
    # Get filtered person_ids for subsequent analyses
    person_ids = set(demographics_df['PERSON_ID'].unique()) if 'PERSON_ID' in demographics_df.columns else set()
    
    # Step 4: Analyze demographics (with filtered data)
    print("\n" + "="*60)
    print("DEMOGRAPHICS ANALYSIS (FILTERED)")
    print("="*60)
    
    print(f"Total patients after filtering: {len(demographics_df):,}")
    
    # Print column names
    print("\nColumn Names:")
    print(", ".join(demographics_df.columns.tolist()))
    
    # Race distribution
    if 'RACE_CONCEPT_ID' in demographics_df.columns or 'RACE_SOURCE_VALUE' in demographics_df.columns:
        race_col = 'RACE_SOURCE_VALUE' if 'RACE_SOURCE_VALUE' in demographics_df.columns else 'RACE_CONCEPT_ID'
        print("\nRace Distribution:")
        race_counts = demographics_df[race_col].value_counts()
        for race, count in race_counts.head(10).items():
            print(f"  {race}: {count:,} ({count/len(demographics_df)*100:.2f}%)")
    
    # Gender distribution
    if 'GENDER_CONCEPT_ID' in demographics_df.columns or 'GENDER_SOURCE_VALUE' in demographics_df.columns:
        gender_col = 'GENDER_SOURCE_VALUE' if 'GENDER_SOURCE_VALUE' in demographics_df.columns else 'GENDER_CONCEPT_ID'
        print("\nGender Distribution:")
        gender_counts = demographics_df[gender_col].value_counts()
        for gender, count in gender_counts.items():
            print(f"  {gender}: {count:,} ({count/len(demographics_df)*100:.2f}%)")
    
    # Ethnicity distribution
    if 'ETHNICITY_CONCEPT_ID' in demographics_df.columns or 'ETHNICITY_SOURCE_VALUE' in demographics_df.columns:
        ethnicity_col = 'ETHNICITY_SOURCE_VALUE' if 'ETHNICITY_SOURCE_VALUE' in demographics_df.columns else 'ETHNICITY_CONCEPT_ID'
        print("\nEthnicity Distribution:")
        ethnicity_counts = demographics_df[ethnicity_col].value_counts()
        for ethnicity, count in ethnicity_counts.head(10).items():
            print(f"  {ethnicity}: {count:,} ({count/len(demographics_df)*100:.2f}%)")
    
    # Age at diagnosis distribution (if calculated)
    if 'age_at_diagnosis' in demographics_df.columns:
        print("\nAge at Diagnosis Distribution:")
        age_stats = demographics_df['age_at_diagnosis'].describe()
        print(f"  Mean: {age_stats['mean']:.2f} years")
        print(f"  Std: {age_stats['std']:.2f} years")
        print(f"  Min: {age_stats['min']:.2f} years")
        print(f"  25%: {age_stats['25%']:.2f} years")
        print(f"  50% (Median): {age_stats['50%']:.2f} years")
        print(f"  75%: {age_stats['75%']:.2f} years")
        print(f"  Max: {age_stats['max']:.2f} years")
    
    # Clean up demographics dataframe to save memory
    del demographics_df
    gc.collect()
    
    # Step 5: Analyze ICD codes with filtered person_ids
    analyze_icd_codes(person_ids)
    
    # Step 6: Analyze measurements with filtered person_ids
    analyze_measurements(person_ids)
    
    # Step 7: Analyze medications with filtered person_ids
    analyze_medications(person_ids)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
    