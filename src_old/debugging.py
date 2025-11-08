import pandas as pd
import numpy as np
import gc
from datetime import datetime
import boto3
from io import StringIO
import warnings
import re
warnings.filterwarnings('ignore')
MEASUREMENTS = {
        'hba1c': {
            'keywords': [
                'hba1c', 'hemoglobin a1c', 'a1c', 'glycosylated hemoglobin',
                'glycated hemoglobin', 'labhba1c', 'eag', 'hgb a1c', 'hb a1c',
                'glycohemoglobin', 'diabetic control', 'glucose control',
                'hemoglobin', 'hgb', 'glyco', 'glycated', 'diabetic',
                'hba', 'a1c%', 'hgba1c', 'hb-a1c', 'hemoglobin-a1c'
            ],
            'exclude': ['mean', 'estimated', 'average']
        },
        
        'fasting_glucose': {
            'keywords': [
                'fasting', 'fbs', 'fpg', 'fasting glucose', 'fasting blood sugar',
                'fasting plasma glucose', 'glucose fasting', 'fasting blood glucose'
            ],
            'exclude': ['ogtt', 'tolerance', 'random', 'postprandial', '2h', '2hr', '2 h', 'after', 'post']
        },
        
        'serum_glucose': {
            'keywords': [
                'serum glucose', 'glucose', 'blood glucose', 'plasma glucose',
                'random glucose', 'glucose serum', 'gluc', 'bg', 'blood sugar',
                'rbs', 'glucose level'
            ],
            'exclude': ['fasting', 'ogtt', 'tolerance', '2h', '2hr', 'urine', 'csf']
        },
        
        'c_peptide': {
            'keywords': [
                'c-peptide', 'c peptide', 'cpeptide', 'connecting peptide',
                'serum c peptide', 'plasma c-peptide', 'c peptide serum',
                'c-pep', 'c pep', 'cpep', 'connecting pep', 'c_peptide',
                'c_pep', 'serum peptide', 'plasma peptide', 'blood c-peptide',
                'c-pep level', 'cpeptide level'
            ],
            'exclude': ['urine', 'brain', 'bnp', 'pro', 'natriuretic']
        },
        
        'urine_c_peptide': {
            'keywords': [
                'urine c-peptide', 'urine c peptide', 'urinary c-peptide',
                'c-peptide urine', 'c peptide urine', 'urine cpeptide',
                '24hr c-peptide', '24 hour c-peptide'
            ],
            'exclude': ['serum', 'plasma', 'blood']
        },
        
        'glucose_2h_ogtt': {
            'keywords': [
                '2-h glucose', '2 hour glucose', '2h glucose', 'ogtt',
                'oral glucose tolerance test', 'glucose tolerance test',
                '2-hour post glucose', '2hr glucose', 'glucose 2 hour',
                'glucose tolerance 2h', 'gtt 2 hour', 'post load glucose',
                '120 min', '120 minute', 'two hour glucose'
            ],
            'exclude': ['fasting', 'baseline']
        },
        
        'hdl': {
            'keywords': [
                'hdl', 'hdl cholesterol', 'hdl-c', 'high density lipoprotein',
                'good cholesterol', 'alpha lipoprotein', 'hdl chol'
            ],
            'exclude': ['ratio', 'total', 'non-hdl', 'vldl']
        },
        
        'ldl': {
            'keywords': [
                'ldl', 'ldl cholesterol', 'ldl-c', 'low density lipoprotein',
                'bad cholesterol', 'beta lipoprotein', 'ldl direct', 'ldl calculated',
                'ldl calc', 'ldl chol'
            ],
            'exclude': ['ratio', 'total', 'vldl', 'oxidized']
        },
        
        'triglycerides': {
            'keywords': [
                'triglycerides', 'triglyceride', 'trig', 'trigs', 'tg',
                'serum triglycerides', 'plasma triglycerides', 'tryglyceride'
            ],
            'exclude': ['ratio']
        },
        
        'alt': {
            'keywords': [
                'alt', 'alanine aminotransferase', 'sgpt', 'alanine transaminase',
                'serum alt', 'liver alt', 'alt liver enzyme', 'gpt'
            ],
            'exclude': ['ratio', 'ast/alt']
        },
        
        'ast': {
            'keywords': [
                'ast', 'aspartate aminotransferase', 'sgot', 'aspartate transaminase',
                'serum ast', 'liver ast', 'ast liver enzyme', 'got'
            ],
            'exclude': ['ratio', 'ast/alt']
        },
        
        'bun': {
            'keywords': [
                'bun', 'blood urea nitrogen', 'urea nitrogen', 'serum urea',
                'blood urea', 'urea', 'serum bun', 'plasma urea'
            ],
            'exclude': ['ratio', 'bun/creatinine', 'pre-bun', 'post-bun']
        },
        
        'creatinine': {
            'keywords': [
                'creatinine', 'serum creatinine', 'creat', 'cr', 'scr',
                'plasma creatinine', 'blood creatinine', 'creatinine serum'
            ],
            'exclude': ['urine', 'clearance', 'ratio', 'kinase', 'ck', 'cpk', 'crs']
        },
        
        'egfr': {
            'keywords': [
                'egfr', 'estimated gfr', 'estimated glomerular filtration rate',
                'gfr', 'glomerular filtration rate', 'kidney function',
                'egfr mdrd', 'egfr ckd-epi', 'calculated gfr', 'e-gfr'
            ],
            'exclude': []
        },
        
        'gad65_antibody': {
            'keywords': [
                'gad65', 'gad65 antibody', 'anti-gad65', 'anti gad65',
                'glutamic acid decarboxylase 65', 'gad antibody',
                'gada', 'anti-gad', 'gad autoantibody', 'gad-65'
            ],
            'exclude': []
        },
        
        'ica512_antibody': {
            'keywords': [
                'ica512', 'ica512 antibody', 'anti-ica512', 'anti ica512',
                'ia-2', 'ia2', 'ia-2 antibody', 'insulinoma antigen 2',
                'islet cell antibody 512', 'anti-ia2', 'ica-512'
            ],
            'exclude': []
        },
        
        'insulin_antibody': {
            'keywords': [
                'insulin antibody', 'anti-insulin', 'anti insulin',
                'iaa', 'insulin autoantibody', 'anti-insulin antibody',
                'insulin autoantibodies'
            ],
            'exclude': ['receptor', 'binding']
        },
        
        'znt8_antibody': {
            'keywords': [
                'znt8', 'znt8 antibody', 'anti-znt8', 'anti znt8',
                'zinc transporter 8', 'zinc transporter 8 antibody',
                'znt8a', 'anti-znt8 antibody'
            ],
            'exclude': []
        },
        
        'tsh': {
            'keywords': [
                'tsh', 'thyroid stimulating hormone', 'thyrotropin', 't.s.h.',
                'thyroid stimulating hormone', 'thyrotropic hormone', 'tsh hormone',
                'thyroid stim hormone'
            ],
            'exclude': ['receptor', 'antibody']
        },
        
        'free_t4': {
            'keywords': [
                'free t4', 'free thyroxine', 'ft4', 'free t-4', 'thyroxine free',
                'free tetraiodothyronine', 't4 free', 'thyroid hormone free t4',
                'unbound t4', 'f-t4'
            ],
            'exclude': ['total', 'bound']
        },
        
        't3': {
            'keywords': [
                't3', 'triiodothyronine', 't-3', 'total t3', 'serum t3',
                'thyroid hormone t3', 'tri-iodothyronine', 'liothyronine'
            ],
            'exclude': ['free', 'reverse', 'rt3']
        },
        
        'urine_microalbumin': {
            'keywords': [
                'urine microalbumin', 'microalbumin', 'microalbuminuria',
                'urinary microalbumin', 'albumin urine', 'urine albumin',
                'microalb', 'microalbumin urine', 'albumin microalbumin',
                'urine protein albumin'
            ],
            'exclude': ['ratio', 'acr', 'uacr']
        },
        
        'urine_creatinine': {
            'keywords': [
                'urine creatinine', 'creatinine urine', 'urinary creatinine',
                'urine creat', '24hr creatinine', 'random urine creatinine',
                'spot urine creatinine', '24 hour creatinine'
            ],
            'exclude': ['ratio', 'albumin', 'serum', 'plasma']
        },
        
        'urine_microalbumin_creatinine_ratio': {
            'keywords': [
                'acr', 'uacr', 'albumin creatinine ratio', 'microalbumin creatinine ratio',
                'albumin/creatinine', 'microalbumin/creatinine', 'alb/cr ratio',
                'urine acr', 'urine albumin creatinine'
            ],
            'exclude': []
        },
        
        'urine_ketone': {
            'keywords': [
                'urine ketone', 'ketones', 'urine ketones', 'urinary ketones',
                'ketone bodies', 'acetoacetate', 'beta-hydroxybutyrate',
                'ketone urine', 'urine acetone'
            ],
            'exclude': ['serum', 'blood', 'plasma']
        },
        
        'systolic_blood_pressure': {
            'keywords': [
                'systolic', 'systolic blood pressure', 'systolic bp', 'sbp',
                'systolic pressure', 'sys bp', 'blood pressure systolic'
            ],
            'exclude': ['diastolic', 'mean']
        },
        
        'diastolic_blood_pressure': {
            'keywords': [
                'diastolic', 'diastolic blood pressure', 'diastolic bp', 'dbp',
                'diastolic pressure', 'dias bp', 'blood pressure diastolic'
            ],
            'exclude': ['systolic', 'mean']
        },
        
        'venous_blood_ph': {
            'keywords': [
                'venous blood ph', 'venous ph', 'blood ph', 'ph venous',
                'venous blood hydrogen', 'vbg ph', 'ph blood', 'arterial ph'
            ],
            'exclude': ['urine', 'gastric']
        },
        
        'venous_blood_hco3': {
            'keywords': [
                'venous blood hco3', 'venous hco3', 'blood hco3', 'bicarbonate',
                'hco3', 'venous bicarbonate', 'serum bicarbonate', 'co2',
                'total co2', 'bicarb', 'hco3-'
            ],
            'exclude': ['arterial']
        },
        
        'igf_1_z_score': {
            'keywords': [
                'igf-1 z-score', 'igf1 z-score', 'igf 1 z score', 'igf1 z score',
                'insulin like growth factor 1 z score', 'igf-1 z score',
                'igf1 standard deviation score', 'igf-1 sds', 'igf1 sds',
                'somatomedin c z score'
            ],
            'exclude': []
        },
        
        'igf_bp3_z_score': {
            'keywords': [
                'igf-bp3 z-score', 'igfbp3 z-score', 'igf bp3 z score', 'igfbp3 z score',
                'insulin like growth factor binding protein 3 z score',
                'igf-bp3 z score', 'igfbp-3 z-score', 'igfbp3 sds', 'igf-bp3 sds'
            ],
            'exclude': []
        }
    }
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
            #'E10.1',   # Type 1 diabetes mellitus with ketoacidosis
            'E10.10',  # Type 1 diabetes mellitus with ketoacidosis, without coma
            'E10.11',  # Type 1 diabetes mellitus with ketoacidosis with coma

            # Type 2 Diabetes (DKA)
            #'E11.1',   # Type 2 diabetes mellitus with ketoacidosis
            'E11.10',  # Type 2 diabetes mellitus with ketoacidosis, without coma
            'E11.11',  # Type 2 diabetes mellitus with ketoacidosis with coma

            # Other Forms of Diabetes (DKA)
            'E13.1',  # Other specified diabetes mellitus with ketoacidosis, without coma
            #'E13.11',  # Other specified diabetes mellitus with ketoacidosis with coma

            # Secondary Diabetes (DKA)
            'E08.10',  # Diabetes due to underlying condition with ketoacidosis, without coma
            'E08.11',  # Diabetes due to underlying condition with ketoacidosis with coma

            # Coma and Complications
            'E09.10',   # Type 1 diabetes mellitus, unspecified
            'E09.11',   # Type 2 diabetes mellitus, unspecified
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
    print(f"Birth dates found: {demo_df['birth_date'].notna().sum()}/{len(demo_df)}")
    
    return demo_df[['PERSON_ID', 'sex', 'birth_date', 'race', 'ethnicity']]

def extract_measurements_chunk(df_chunk, keywords, exclude_keywords=None):
    """
    Enhanced extraction with optional exclusion keywords
    """
    # Filter rows by keywords in MEASUREMENT_SOURCE_VALUE
    pattern = '|'.join(keywords)
    filtered = df_chunk[
        df_chunk["MEASUREMENT_SOURCE_VALUE"].str.contains(pattern, case=False, na=False)
    ]
    
    # Apply exclusions if provided
    if exclude_keywords and len(filtered) > 0:
        exclude_pattern = '|'.join(exclude_keywords)
        filtered = filtered[
            ~filtered["MEASUREMENT_SOURCE_VALUE"].str.contains(exclude_pattern, case=False, na=False)
        ]
    
    return filtered

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

        
def flexible_match(text, measurement_config):
    """Enhanced version with better HbA1c matching and debugging"""
    if pd.isna(text):
        return False
        
    # Convert to string and lowercase
    text_str = str(text).strip().lower()
    
    # Handle various data formats - extract the actual measurement name
    if '|' in text_str:
        # Take the part before the pipe (description)
        text_str = text_str.split('|')[0].strip()
    
    # Create multiple normalized versions for matching
    # 1. Clean version with normalized spaces
    text_clean = re.sub(r'[,;_\-/]+', ' ', text_str)
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()
    
    # 2. No spaces version
    text_no_space = text_clean.replace(' ', '')
    
    # 3. Only alphanumeric
    text_alphanum = re.sub(r'[^a-z0-9]', '', text_str)
    
    # 4. With dots preserved (for a1c variations)
    text_with_dots = re.sub(r'[^a-z0-9.]', ' ', text_str).strip()
    
    # Check exclusions first
    for exclude in measurement_config.get('exclude', []):
        exclude_lower = exclude.lower()
        exclude_no_space = exclude_lower.replace(' ', '')
        
        if (exclude_lower in text_clean or 
            exclude_no_space in text_no_space):
            return False
    
    # Check if any keyword is present
    keyword_found = False
    for keyword in measurement_config['keywords']:
        keyword_lower = keyword.lower()
        keyword_no_space = keyword_lower.replace(' ', '')
        keyword_alphanum = re.sub(r'[^a-z0-9]', '', keyword_lower)
        
        # Multiple matching strategies
        if (keyword_lower in text_clean or
            keyword_no_space in text_no_space or
            keyword_alphanum in text_alphanum or
            keyword_lower in text_with_dots):
            keyword_found = True
            break
            
            if keyword_found:
                break
    
    if not keyword_found:
        return False
    
    # Check must_have conditions
    if measurement_config.get('must_have'):
        must_have_found = False
        for must_have in measurement_config['must_have']:
            must_have_lower = must_have.lower()
            if must_have_lower in text_clean:
                must_have_found = True
                break
        
        if not must_have_found:
            return False
    
    return True


def debug_measurement_values(s3_client, bucket, prefix, sample_size=1000):
    """Check what measurement values actually look like in the data"""
    print("Checking measurement values in data...")
    
    measurement_key = f'{prefix}measurement.csv'
    obj = s3_client.get_object(Bucket=bucket, Key=measurement_key)
    
    # Read a sample
    sample_df = pd.read_csv(obj['Body'], nrows=sample_size)
    
    # Get unique values
    unique_measurements = sample_df['MEASUREMENT_SOURCE_VALUE'].dropna().unique()
    
    print(f"\nFound {len(unique_measurements)} unique measurement types in sample")
    print("\nSample of measurement values:")
    for i, val in enumerate(unique_measurements[:50]):
        print(f"  {val}")
    
    # Test specific keywords
    print("\n\nTesting keyword matches:")
    test_keywords = ['glucose', 'hba1c', 'a1c', 'creatinine', 'hdl', 'ldl']
    
    for keyword in test_keywords:
        matches = sample_df[
            sample_df['MEASUREMENT_SOURCE_VALUE'].str.contains(keyword, case=False, na=False)
        ]['MEASUREMENT_SOURCE_VALUE'].unique()
        
        print(f"\n'{keyword}' matches ({len(matches)}):")
        for match in matches[:5]:
            print(f"  - {match}")
    
    return sample_df


def process_measurements_chunked_improved(s3_client, bucket, prefix, t2d_dates, chunk_size=50000):
    """Improved version with debugging output"""
    import gc
    import psutil
    import os
    
    print("Processing measurements in chunks...")
    
    # Initialize result dataframe
    patient_measurements = pd.DataFrame({'PERSON_ID': t2d_dates['PERSON_ID'].unique()})
    
    # Initialize all measurement columns
    for measurement_name in MEASUREMENTS.keys():
        patient_measurements[f'{measurement_name}_present'] = 0
        patient_measurements[f'{measurement_name}_value_first'] = np.nan
        patient_measurements[f'{measurement_name}_value_last'] = np.nan
        patient_measurements[f'{measurement_name}_value_mean'] = np.nan
        patient_measurements[f'{measurement_name}_value_min'] = np.nan
        patient_measurements[f'{measurement_name}_value_max'] = np.nan
        patient_measurements[f'{measurement_name}_date_first'] = pd.NaT
        patient_measurements[f'{measurement_name}_date_last'] = pd.NaT
        patient_measurements[f'{measurement_name}_years_from_diagnosis_first'] = np.nan
    
    # Get the S3 object
    measurement_key = f'{prefix}measurement.csv'
    obj = s3_client.get_object(Bucket=bucket, Key=measurement_key)
    
    # Track matches for debugging
    match_counts = {measurement: 0 for measurement in MEASUREMENTS.keys()}
    total_rows_processed = 0
    
    # Process in chunks
    chunk_count = 0
    for chunk in pd.read_csv(obj['Body'], chunksize=chunk_size):
        chunk_count += 1
        total_rows_processed += len(chunk)
        print(f"\n{'='*70}")
        print(f"Processing chunk {chunk_count} ({len(chunk)} rows)...")
        print(f"{'='*70}")
        
        # Print sample of raw measurement values
        if chunk_count == 1:  # Only for first chunk
            print("\n--- Sample of Raw Measurement Values (First Chunk) ---")
            sample_measurements = chunk['MEASUREMENT_SOURCE_VALUE'].dropna().head(20)
            for i, val in enumerate(sample_measurements):
                print(f"  {i+1}. {val}")
            
            print(f"\n--- Unique Measurement Types in First {min(100, len(chunk))} Rows ---")
            unique_types = chunk.head(100)['MEASUREMENT_SOURCE_VALUE'].dropna().unique()
            for val in unique_types[:30]:  # Show first 30 unique types
                print(f"  - {val}")
        
        # Keep only necessary columns
        chunk = chunk[['PERSON_ID', 'MEASUREMENT_DATE', 'MEASUREMENT_SOURCE_VALUE', 'VALUE_AS_NUMBER']].copy()
        
        # Convert dates
        chunk['measurement_date'] = pd.to_datetime(chunk['MEASUREMENT_DATE'])
        
        # Merge with T2D dates
        chunk = chunk.merge(t2d_dates, on='PERSON_ID', how='inner')
        
        # Calculate years from diagnosis
        chunk['years_from_diagnosis'] = (
            (chunk['measurement_date'] - chunk['t2d_diagnosis_date']).dt.days / 365.25
        )
        
        # Process each measurement type
        print(f"\n--- Successful Extractions in Chunk {chunk_count} ---")
        extraction_summary = []
        
        for measurement_name, measurement_config in MEASUREMENTS.items():
            # Extract matching measurements using the simpler function
            measurement_matches = extract_measurements_chunk(
                chunk, 
                keywords=measurement_config['keywords'],
                exclude_keywords=measurement_config.get('exclude', None)
            )
            
            if len(measurement_matches) > 0:
                match_counts[measurement_name] += len(measurement_matches)
                
                # Print sample of successful matches
                if len(measurement_matches) > 0:
                    extraction_summary.append({
                        'measurement': measurement_name,
                        'count': len(measurement_matches),
                        'samples': measurement_matches['MEASUREMENT_SOURCE_VALUE'].head(10).tolist()
                    })
                
                # Remove rows with missing values
                measurement_matches = measurement_matches.dropna(subset=['VALUE_AS_NUMBER'])
                
                if len(measurement_matches) > 0:
                    # Print value statistics for this measurement type
                    if chunk_count <= 3:  # Only for first 3 chunks
                        value_stats = measurement_matches['VALUE_AS_NUMBER'].describe()
                        print(f"\n  {measurement_name}: {len(measurement_matches)} matches")
                        print(f"    Value range: {value_stats['min']:.2f} - {value_stats['max']:.2f}")
                        print(f"    Mean: {value_stats['mean']:.2f}, Median: {value_stats['50%']:.2f}")
                        
                        # Show first few actual measurement descriptions
                        sample_matches = measurement_matches.head(5)
                        for idx, row in sample_matches.iterrows():
                            print(f"    Sample: '{row['MEASUREMENT_SOURCE_VALUE']}' = {row['VALUE_AS_NUMBER']:.2f}")
                    
                    # Group by patient and calculate statistics
                    grouped = measurement_matches.groupby('PERSON_ID')
                    
                    # Calculate aggregations
                    agg_stats = grouped['VALUE_AS_NUMBER'].agg(['mean', 'min', 'max']).reset_index()
                    agg_stats.columns = ['PERSON_ID', 'mean_val', 'min_val', 'max_val']
                    
                    # Get first and last values
                    first_last = grouped.apply(
                        lambda x: pd.Series({
                            'first_val': x.sort_values('measurement_date').iloc[0]['VALUE_AS_NUMBER'],
                            'last_val': x.sort_values('measurement_date').iloc[-1]['VALUE_AS_NUMBER'],
                            'first_date': x.sort_values('measurement_date').iloc[0]['measurement_date'],
                            'last_date': x.sort_values('measurement_date').iloc[-1]['measurement_date'],
                            'first_years': x.sort_values('measurement_date').iloc[0]['years_from_diagnosis']
                        })
                    ).reset_index()
                    
                    # Merge statistics
                    patient_stats = agg_stats.merge(first_last, on='PERSON_ID')
                    
                    # Update patient_measurements
                    for _, row in patient_stats.iterrows():
                        idx = patient_measurements['PERSON_ID'] == row['PERSON_ID']
                        patient_measurements.loc[idx, f'{measurement_name}_present'] = 1
                        patient_measurements.loc[idx, f'{measurement_name}_value_first'] = row['first_val']
                        patient_measurements.loc[idx, f'{measurement_name}_value_last'] = row['last_val']
                        patient_measurements.loc[idx, f'{measurement_name}_value_mean'] = row['mean_val']
                        patient_measurements.loc[idx, f'{measurement_name}_value_min'] = row['min_val']
                        patient_measurements.loc[idx, f'{measurement_name}_value_max'] = row['max_val']
                        patient_measurements.loc[idx, f'{measurement_name}_date_first'] = row['first_date']
                        patient_measurements.loc[idx, f'{measurement_name}_date_last'] = row['last_date']
                        patient_measurements.loc[idx, f'{measurement_name}_years_from_diagnosis_first'] = row['first_years']
        
        # Print extraction summary for this chunk
        if extraction_summary:
            print(f"\n--- Extraction Summary for Chunk {chunk_count} ---")
            for item in extraction_summary:
                print(f"\n{item['measurement']}: {item['count']} matches")
                print("Sample matches:")
                for i, sample in enumerate(item['samples'][:5]):  # Show up to 5 samples
                    print(f"  {i+1}. {sample}")
        else:
            print(f"\nNo matches found in chunk {chunk_count}")
        
        # Clean up
        del chunk
        gc.collect()
    
    print(f"\n\n{'='*70}")
    print(f"Finished processing {chunk_count} chunks ({total_rows_processed:,} total rows).")
    print(f"{'='*70}")
    
    print("\nFinal Match Counts by Measurement Type:")
    for measurement, count in sorted(match_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_rows_processed) * 100 if total_rows_processed > 0 else 0
        print(f"  {measurement}: {count:,} matches ({pct:.2f}% of rows)")
    
    # Print measurements with no matches
    no_matches = [m for m, c in match_counts.items() if c == 0]
    if no_matches:
        print(f"\nMeasurements with NO matches: {', '.join(no_matches)}")
    
    return patient_measurements


def debug_measurement_search(s3_client, bucket, prefix, search_terms=None):
    """
    Debug function to search for specific measurement terms in the data
    """
    print("\n" + "="*70)
    print("DEBUG: Searching for measurement terms in data")
    print("="*70)
    
    measurement_key = f'{prefix}measurement.csv'
    obj = s3_client.get_object(Bucket=bucket, Key=measurement_key)
    
    # Read first 10000 rows for debugging
    debug_df = pd.read_csv(obj['Body'], nrows=10000)
    
    # Default search terms if none provided
    if search_terms is None:
        search_terms = ['hba1c', 'a1c', 'hemoglobin', 'glucose', 'cholesterol', 'hdl', 'ldl', 
                       'creatinine', 'peptide', 'antibody', 'tsh', 'microalbumin']
    
    print(f"\nSearching for terms: {', '.join(search_terms)}")
    
    for term in search_terms:
        print(f"\n--- Searching for '{term}' ---")
        matches = debug_df[
            debug_df['MEASUREMENT_SOURCE_VALUE'].str.contains(term, case=False, na=False)
        ]
        
        if len(matches) > 0:
            print(f"Found {len(matches)} matches:")
            unique_matches = matches['MEASUREMENT_SOURCE_VALUE'].unique()
            for i, match in enumerate(unique_matches[:10]):  # Show up to 10 unique matches
                print(f"  {i+1}. {match}")
                # Show a few values for this measurement
                sample_values = matches[matches['MEASUREMENT_SOURCE_VALUE'] == match]['VALUE_AS_NUMBER'].dropna().head(3)
                if len(sample_values) > 0:
                    print(f"     Sample values: {', '.join([f'{v:.2f}' for v in sample_values])}")
        else:
            print(f"No matches found for '{term}'")
    
    # Check for measurements that might use different naming conventions
    print("\n--- Checking Alternative Naming Patterns ---")
    
    # Look for patterns with special characters
    special_patterns = ['%', '-', '_', '/', '\\', '|']
    for pattern in special_patterns:
        count = debug_df['MEASUREMENT_SOURCE_VALUE'].str.contains(f'\\{pattern}', na=False, regex=True).sum()
        if count > 0:
            print(f"\nMeasurements containing '{pattern}': {count}")
            samples = debug_df[debug_df['MEASUREMENT_SOURCE_VALUE'].str.contains(f'\\{pattern}', na=False, regex=True)]['MEASUREMENT_SOURCE_VALUE'].unique()[:5]
            for sample in samples:
                print(f"  - {sample}")
    
    return debug_df
    

def main():
    # Initialize S3 client
    s3 = boto3.client('s3')
    
    # Define S3 paths
    bucket = 'dsw-sagemaker-dev-s3'
    prefix = 'T2D_Tosur/data/T2D_OMOP_variables/'
    
    print("Reading data from S3...")
    
    # OPTIONAL: Run measurement exploration first (comment out after first run)
    print("\n" + "="*80)
    print("RUNNING MEASUREMENT DATA EXPLORATION")
    print("="*80)
    explore_measurement_data(s3, bucket, prefix, sample_size=10000)
    
    # OPTIONAL: Run debug search for specific terms
    print("\n" + "="*80) 
    print("RUNNING DEBUG SEARCH")
    print("="*80)
    debug_measurement_search(s3, bucket, prefix, 
                           search_terms=['hba1c', 'a1c', 'glucose', 'peptide', 'antibody'])
    
    # Continue with regular processing...
    print("\n" + "="*80)
    print("STARTING MAIN PROCESSING")
    print("="*80)
    
    # Read the tables
    person_df = read_s3_csv(s3, bucket, f'{prefix}person.csv')
    condition_df = read_s3_csv(s3, bucket, f'{prefix}condition_occurrence.csv')
    drug_df = read_s3_csv(s3, bucket, f'{prefix}drug_exposure.csv')
    
    print(f"Loaded {len(person_df)} patients")
    print(f"Loaded {len(condition_df)} condition records") 
    print(f"Loaded {len(drug_df)} drug records")
    
    # Process demographics
    demographics = process_demographics(person_df)
    del person_df
    gc.collect()
    
    # Get T2D diagnosis dates
    t2d_dates = get_t2d_diagnosis_date(condition_df)
    print(f"Found T2D diagnosis dates for {len(t2d_dates)} patients")

    # Calculate age at diagnosis
    demographics = demographics.merge(t2d_dates, on='PERSON_ID', how='left')
    demographics['age_at_diagnosis'] = (
        (demographics['t2d_diagnosis_date'] - demographics['birth_date']).dt.days / 365.25
    )
    
    # Process medications
    medications = process_medications(drug_df, t2d_dates)
    del drug_df
    gc.collect()
    
    # Process conditions
    conditions = process_conditions(condition_df, t2d_dates)
    del condition_df
    gc.collect()
    
    # Process measurements with the improved function (with debug output)
    print("\n" + "="*50)
    print("Starting measurement processing...")
    print("="*50)
    measurements = process_measurements_chunked_improved(s3, bucket, prefix, t2d_dates, chunk_size=50000)
    gc.collect()
    
    # Rest of your main function continues as before...
    # [Rest of the merging and output code stays the same]

if __name__ == "__main__":
    final_df = main()