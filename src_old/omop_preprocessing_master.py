import pandas as pd
import numpy as np
import boto3
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import gc
from collections import Counter
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

#### input
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
# Set to None to include all drugs:
# DRUG_CLASSES = None

# Example: Filter for specific conditions (optional)
CONDITION_CODES = {
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
        #'E13.10',  # Other specified diabetes mellitus with ketoacidosis, without coma
        #'E13.11',  # Other specified diabetes mellitus with ketoacidosis with coma

        # Secondary Diabetes (DKA)
        #'E08.10',  # Diabetes due to underlying condition with ketoacidosis, without coma
        #'E08.11',  # Diabetes due to underlying condition with ketoacidosis with coma

        # Coma and Complications
        #'E09.10',   # Type 1 diabetes mellitus, unspecified
        #'E09.11',   # Type 2 diabetes mellitus, unspecified
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
# Set to None to include all conditions:
# CONDITION_CODES = None

# Example: Filter for specific measurements (optional)
MEASUREMENT_TYPES = {
    'hba1c': {
        'keywords': [
            'hba1c', 'hemoglobin a1c', 'a1c', 'glycosylated hemoglobin',
            'glycated hemoglobin', 'labhba1c', 'eag', 'hgb a1c', 'hb a1c',
            'glycohemoglobin', 'diabetic control', 'glucose control',
            'hemoglobin', 'hgb', 'glyco', 'glycated', 'diabetic',
            'hba', 'a1c%', 'hgba1c', 'hb-a1c', 'hemoglobin-a1c'
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
        'exclude': ['ogtt', 'tolerance', 'random', 'postprandial', '2h', '2hr', '2 h', 'after', 'post']
    },
    
    'serum_glucose': {
        'keywords': [
            'serum glucose', 'glucose', 'blood glucose', 'plasma glucose',
            'random glucose', 'glucose serum', 'gluc', 'bg', 'blood sugar',
            'rbs', 'glucose level'
        ],
        'must_have': [],
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
        'must_have': [],
        'exclude': ['urine', 'brain', 'bnp', 'pro', 'natriuretic']
    },
    
    'urine_c_peptide': {
        'keywords': [
            'urine c-peptide', 'urine c peptide', 'urinary c-peptide',
            'c-peptide urine', 'c peptide urine', 'urine cpeptide',
            '24hr c-peptide', '24 hour c-peptide'
        ],
        'must_have': ['peptide'],
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
        'must_have': ['glucose', 'gluc'],
        'exclude': ['fasting', 'baseline']
    },
    
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
            'bad cholesterol', 'beta lipoprotein', 'ldl direct', 'ldl calculated',
            'ldl calc', 'ldl chol'
        ],
        'must_have': [],
        'exclude': ['ratio', 'total', 'vldl', 'oxidized']
    },
    
    'triglycerides': {
        'keywords': [
            'triglycerides', 'triglyceride', 'trig', 'trigs', 'tg',
            'serum triglycerides', 'plasma triglycerides', 'tryglyceride'
        ],
        'must_have': [],
        'exclude': ['ratio']
    },
    
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
        'exclude': ['ratio', 'ast/alt', 'DIASTOLIC', 'diastolic']
    },
    
    'bun': {
        'keywords': [
            'bun', 'blood urea nitrogen', 'urea nitrogen', 'serum urea',
            'blood urea', 'urea', 'serum bun', 'plasma urea'
        ],
        'must_have': [],
        'exclude': ['ratio', 'bun/creatinine', 'pre-bun', 'post-bun']
    },
    
    'creatinine': {
        'keywords': [
            'creatinine', 'serum creatinine', 'creat', 'cr', 'scr',
            'plasma creatinine', 'blood creatinine', 'creatinine serum'
        ],
        'must_have': [],
        'exclude': ['urine', 'clearance', 'ratio', 'kinase', 'ck', 'cpk']
    },
    
    'egfr': {
        'keywords': [
            'egfr', 'estimated gfr', 'estimated glomerular filtration rate',
            'gfr', 'glomerular filtration rate', 'kidney function',
            'egfr mdrd', 'egfr ckd-epi', 'calculated gfr', 'e-gfr'
        ],
        'must_have': [],
        'exclude': []
    },
    
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
            'iaa', 'insulin autoantibody', 'anti-insulin antibody',
            'insulin autoantibodies'
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
    
    'tsh': {
        'keywords': [
            'tsh', 'thyroid stimulating hormone', 'thyrotropin', 't.s.h.',
            'thyroid stimulating hormone', 'thyrotropic hormone', 'tsh hormone',
            'thyroid stim hormone'
        ],
        'must_have': [],
        'exclude': ['receptor', 'antibody']
    },
    
    'free_t4': {
        'keywords': [
            'free t4', 'free thyroxine', 'ft4', 'free t-4', 'thyroxine free',
            'free tetraiodothyronine', 't4 free', 'thyroid hormone free t4',
            'unbound t4', 'f-t4'
        ],
        'must_have': [],
        'exclude': ['total', 'bound']
    },
    
    't3': {
        'keywords': [
            't3', 'triiodothyronine', 't-3', 'total t3', 'serum t3',
            'thyroid hormone t3', 'tri-iodothyronine', 'liothyronine'
        ],
        'must_have': [],
        'exclude': ['free', 'reverse', 'rt3']
    },
    
    'urine_microalbumin': {
        'keywords': [
            'urine microalbumin', 'microalbumin', 'microalbuminuria',
            'urinary microalbumin', 'albumin urine', 'urine albumin',
            'microalb', 'microalbumin urine', 'albumin microalbumin',
            'urine protein albumin'
        ],
        'must_have': [],
        'exclude': ['ratio', 'acr', 'uacr']
    },
    
    'urine_creatinine': {
        'keywords': [
            'urine creatinine', 'creatinine urine', 'urinary creatinine',
            'urine creat', '24hr creatinine', 'random urine creatinine',
            'spot urine creatinine', '24 hour creatinine'
        ],
        'must_have': ['urine', 'urinary'],
        'exclude': ['ratio', 'albumin', 'serum', 'plasma']
    },
    
    'urine_microalbumin_creatinine_ratio': {
        'keywords': [
            'acr', 'uacr', 'albumin creatinine ratio', 'microalbumin creatinine ratio',
            'albumin/creatinine', 'microalbumin/creatinine', 'alb/cr ratio',
            'urine acr', 'urine albumin creatinine'
        ],
        'must_have': ['ratio'],
        'exclude': []
    },
    
    'urine_ketone': {
        'keywords': [
            'urine ketone', 'ketones', 'urine ketones', 'urinary ketones',
            'ketone bodies', 'acetoacetate', 'beta-hydroxybutyrate',
            'ketone urine', 'urine acetone'
        ],
        'must_have': [],
        'exclude': ['serum', 'blood', 'plasma']
    },
    
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
    
    'venous_blood_ph': {
        'keywords': [
            'venous blood ph', 'venous ph', 'blood ph', 'ph venous',
            'venous blood hydrogen', 'vbg ph', 'ph blood', 'arterial ph'
        ],
        'must_have': [],
        'exclude': ['urine', 'gastric']
    },
    
    'venous_blood_hco3': {
        'keywords': [
            'venous blood hco3', 'venous hco3', 'blood hco3', 'bicarbonate',
            'hco3', 'venous bicarbonate', 'serum bicarbonate', 'co2',
            'total co2', 'bicarb', 'hco3-'
        ],
        'must_have': [],
        'exclude': ['arterial']
    },
    
    'igf_1_z_score': {
        'keywords': [
            'igf-1 z-score', 'igf1 z-score', 'igf 1 z score', 'igf1 z score',
            'insulin like growth factor 1 z score', 'igf-1 z score',
            'igf1 standard deviation score', 'igf-1 sds', 'igf1 sds',
            'somatomedin c z score'
        ],
        'must_have': [],
        'exclude': []
    },
    
    'igf_bp3_z_score': {
        'keywords': [
            'igf-bp3 z-score', 'igfbp3 z-score', 'igf bp3 z score', 'igfbp3 z score',
            'insulin like growth factor binding protein 3 z score',
            'igf-bp3 z score', 'igfbp-3 z-score', 'igfbp3 sds', 'igf-bp3 sds'
        ],
        'must_have': [],
        'exclude': []
    }
}
MAIN_DIAGNOSIS = {
'ICD9': ['250.00', '250.02'],
'ICD10': ['E11.', 'E11.0', 'E11.1', 'E11.2', 'E11.3', 'E11.4', 'E11.5', 'E11.6', 'E11.7', 'E11.8', 'E11.9']
}
# Set to None to include all measurements:
# MEASUREMENT_TYPES = None

# Example: Demographics filter (optional)
DEMOGRAPHICS_FILTER = {
    'age_min': 0,
    'age_max': 20,
    #'gender': ['Male', 'Female'],  # or use concept IDs: [8507, 8532]
    # 'race': ['White', 'Black', 'Asian']  # optional
}


import functools
import psutil
import gc

def monitor_memory(func):
    """Decorator to monitor memory usage of functions"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024**2  # MB
        
        print(f"\n[Memory Monitor] Starting {func.__name__}")
        print(f"  Memory before: {mem_before:.2f} MB")
        print(f"  Available system memory: {psutil.virtual_memory().available / 1024**2:.2f} MB")
        
        try:
            # Run function
            result = func(*args, **kwargs)
            
            # Get memory after
            mem_after = process.memory_info().rss / 1024**2
            mem_diff = mem_after - mem_before
            
            print(f"[Memory Monitor] Completed {func.__name__}")
            print(f"  Memory after: {mem_after:.2f} MB")
            print(f"  Memory increase: {mem_diff:.2f} MB")
            
            # Force garbage collection if memory usage is high
            if mem_after > 2000:  # If using more than 2GB
                print(f"  High memory usage detected. Running garbage collection...")
                gc.collect()
                mem_after_gc = process.memory_info().rss / 1024**2
                print(f"  Memory after GC: {mem_after_gc:.2f} MB")
            
            return result
            
        except Exception as e:
            print(f"[Memory Monitor] Error in {func.__name__}: {e}")
            raise
    
    return wrapper



class OMOPPreprocessor:
    """
    Tool for preprocessing and analyzing OMOP data distributions
    """
    
    def __init__(self, bucket='dsw-sagemaker-dev-s3', prefix='OMOP_data_extractions/'):
        """
        Initialize the preprocessor
        
        Args:
            bucket: S3 bucket containing OMOP data
            prefix: Prefix for OMOP data in bucket
        """
        self.bucket = bucket
        self.prefix = prefix
        self.s3 = boto3.client('s3')
        
        # Store dataframes
        # Store dataframes
        self.demographics_df = pd.DataFrame()
        self.conditions_df = pd.DataFrame()
        self.medications_df = pd.DataFrame()
        self.measurements_df = pd.DataFrame()

        # Store filtered dataframes
        self.filtered_demographics = pd.DataFrame()
        self.filtered_conditions = pd.DataFrame()
        self.filtered_medications = pd.DataFrame()
        self.filtered_measurements = pd.DataFrame()


    def optimize_dataframes(self):
        """
        Convert string columns to categories to save memory
        """
        print("\nOptimizing dataframe memory usage...")
        
        # Optimize demographics
        if not self.demographics_df.empty:
            initial_mem = self.demographics_df.memory_usage(deep=True).sum() / 1024**2
            
            # Convert object columns with low cardinality to category
            for col in self.demographics_df.select_dtypes(include=['object']).columns:
                num_unique = self.demographics_df[col].nunique()
                num_total = len(self.demographics_df[col])
                if num_unique / num_total < 0.5:  # Less than 50% unique values
                    self.demographics_df[col] = self.demographics_df[col].astype('category')
            
            final_mem = self.demographics_df.memory_usage(deep=True).sum() / 1024**2
            print(f"  Demographics: {initial_mem:.2f} MB → {final_mem:.2f} MB ({(1-final_mem/initial_mem)*100:.1f}% reduction)")
        
        # Optimize conditions
        if not self.conditions_df.empty:
            initial_mem = self.conditions_df.memory_usage(deep=True).sum() / 1024**2
            
            # CONDITION_SOURCE_VALUE is often repeated - make it categorical
            if 'CONDITION_SOURCE_VALUE' in self.conditions_df.columns:
                self.conditions_df['CONDITION_SOURCE_VALUE'] = self.conditions_df['CONDITION_SOURCE_VALUE'].astype('category')
            
            final_mem = self.conditions_df.memory_usage(deep=True).sum() / 1024**2
            print(f"  Conditions: {initial_mem:.2f} MB → {final_mem:.2f} MB ({(1-final_mem/initial_mem)*100:.1f}% reduction)")
        
        # Similar for medications and measurements...
        gc.collect()

    @monitor_memory
    def load_data(self, project_name, chunksize=10000, max_rows=None):
        """
        Load OMOP data from S3 with aggressive memory optimization
        
        Args:
            project_name: Project name in S3
            chunksize: Size of chunks to process (reduced from 50000)
            max_rows: Maximum rows to load per file (None for all)
        """
        print(f"\n{'='*60}")
        print(f"LOADING OMOP DATA")
        print(f"{'='*60}")
        
        full_prefix = f"{self.prefix}{project_name}/"
        
        # More aggressive dtype optimization
        dtype_dict = {
            'PERSON_ID': 'int32',
            'GENDER_CONCEPT_ID': 'int16',
            'RACE_CONCEPT_ID': 'int16',  # Changed from int32
            'ETHNICITY_CONCEPT_ID': 'int16',  # Changed from int32
            'YEAR_OF_BIRTH': 'int16',
            'CONDITION_OCCURRENCE_ID': 'int32',
            'CONDITION_CONCEPT_ID': 'int32',
            'DRUG_EXPOSURE_ID': 'int32',
            'DRUG_CONCEPT_ID': 'int32',
            'MEASUREMENT_ID': 'int32',
            'MEASUREMENT_CONCEPT_ID': 'int32',
            'VALUE_AS_NUMBER': 'float32'
        }
        
        # Load demographics - usually smaller
        try:
            print("Loading demographics...")
            demo_key = f"{full_prefix}demographics/person.csv"
            obj = self.s3.get_object(Bucket=self.bucket, Key=demo_key)
            
            # Read with minimal columns first
            essential_demo_cols = ['PERSON_ID', 'GENDER_CONCEPT_ID', 'RACE_CONCEPT_ID', 
                                'ETHNICITY_CONCEPT_ID', 'YEAR_OF_BIRTH', 'BIRTH_DATETIME']
            
            self.demographics_df = pd.read_csv(
                obj['Body'], 
                dtype=dtype_dict,
                low_memory=False,
                usecols=lambda x: x in essential_demo_cols,  # Only load essential columns
                nrows=max_rows
            )
            
            # Immediately optimize strings
            for col in self.demographics_df.select_dtypes(include=['object']).columns:
                self.demographics_df[col] = self.demographics_df[col].astype('category')
            
            print(f"  ✓ Loaded {len(self.demographics_df):,} patient records")
            print(f"  Memory usage: {self.demographics_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            gc.collect()
            
        except Exception as e:
            print(f"  ⚠ Could not load demographics: {e}")
            self.demographics_df = pd.DataFrame()
        
        # Load conditions with aggressive chunking
        try:
            print("Loading conditions with optimized chunking...")
            cond_key = f"{full_prefix}icd_codes/condition_occurrence.csv"
            
            # First, get file size to estimate chunks needed
            obj_meta = self.s3.head_object(Bucket=self.bucket, Key=cond_key)
            file_size_mb = obj_meta['ContentLength'] / (1024 * 1024)
            print(f"  File size: {file_size_mb:.2f} MB")
            
            # Essential columns for conditions
            essential_cond_cols = ['PERSON_ID', 'CONDITION_OCCURRENCE_ID', 
                                'CONDITION_CONCEPT_ID', 'CONDITION_SOURCE_VALUE', 
                                'CONDITION_START_DATE']
            
            obj = self.s3.get_object(Bucket=self.bucket, Key=cond_key)
            
            # Process smaller chunks with immediate optimization
            chunks = []
            total_rows_processed = 0
            
            for i, chunk in enumerate(pd.read_csv(
                obj['Body'], 
                chunksize=chunksize,
                dtype=dtype_dict,
                usecols=lambda x: x in essential_cond_cols,
                nrows=max_rows
            )):
                # Optimize each chunk immediately
                if 'CONDITION_SOURCE_VALUE' in chunk.columns:
                    chunk['CONDITION_SOURCE_VALUE'] = chunk['CONDITION_SOURCE_VALUE'].astype('category')
                
                # Keep only non-null conditions
                chunk = chunk[chunk['CONDITION_SOURCE_VALUE'].notna()]
                
                chunks.append(chunk)
                total_rows_processed += len(chunk)
                
                # Periodically consolidate chunks to avoid too many small dataframes
                if len(chunks) >= 10:
                    temp_df = pd.concat(chunks, ignore_index=True)
                    chunks = [temp_df]
                    gc.collect()
                
                # Progress update
                if (i + 1) % 10 == 0:
                    print(f"    Processed {total_rows_processed:,} rows...")
                
                # Stop if we've reached max_rows
                if max_rows and total_rows_processed >= max_rows:
                    break
            
            # Combine all chunks
            if chunks:
                self.conditions_df = pd.concat(chunks, ignore_index=True)
                del chunks
            else:
                self.conditions_df = pd.DataFrame()
            
            gc.collect()
            
            print(f"  ✓ Loaded {len(self.conditions_df):,} condition records")
            print(f"  Memory usage: {self.conditions_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
        except Exception as e:
            print(f"  ⚠ Could not load conditions: {e}")
            self.conditions_df = pd.DataFrame()
            gc.collect()
        
        # Load medications with same optimization
        try:
            print("Loading medications with optimization...")
            med_key = f"{full_prefix}medications/drug_exposure.csv"
            
            essential_med_cols = ['PERSON_ID', 'DRUG_EXPOSURE_ID', 'DRUG_CONCEPT_ID', 
                                'DRUG_SOURCE_VALUE', 'DRUG_EXPOSURE_START_DATE', 
                                'DRUG_EXPOSURE_END_DATE']
            
            obj = self.s3.get_object(Bucket=self.bucket, Key=med_key)
            
            chunks = []
            total_rows_processed = 0
            
            for i, chunk in enumerate(pd.read_csv(
                obj['Body'],
                chunksize=chunksize,
                dtype=dtype_dict,
                usecols=lambda x: x in essential_med_cols,
                nrows=max_rows
            )):
                # Optimize each chunk
                if 'DRUG_SOURCE_VALUE' in chunk.columns:
                    chunk['DRUG_SOURCE_VALUE'] = chunk['DRUG_SOURCE_VALUE'].astype('category')
                
                # Keep only non-null medications
                chunk = chunk[chunk['DRUG_SOURCE_VALUE'].notna()]
                
                chunks.append(chunk)
                total_rows_processed += len(chunk)
                
                if len(chunks) >= 10:
                    temp_df = pd.concat(chunks, ignore_index=True)
                    chunks = [temp_df]
                    gc.collect()
                
                if max_rows and total_rows_processed >= max_rows:
                    break
            
            if chunks:
                self.medications_df = pd.concat(chunks, ignore_index=True)
                del chunks
            else:
                self.medications_df = pd.DataFrame()
            
            gc.collect()
            
            print(f"  ✓ Loaded {len(self.medications_df):,} medication records")
            print(f"  Memory usage: {self.medications_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
        except Exception as e:
            print(f"  ⚠ Could not load medications: {e}")
            self.medications_df = pd.DataFrame()
            gc.collect()
        
        # Load measurements with same optimization
        try:
            print("Loading measurements with optimization...")
            meas_key = f"{full_prefix}measurements/measurement.csv"
            
            essential_meas_cols = ['PERSON_ID', 'MEASUREMENT_ID', 'MEASUREMENT_CONCEPT_ID',
                                'MEASUREMENT_SOURCE_VALUE', 'VALUE_AS_NUMBER', 
                                'MEASUREMENT_DATE']
            
            obj = self.s3.get_object(Bucket=self.bucket, Key=meas_key)
            
            chunks = []
            total_rows_processed = 0
            
            for i, chunk in enumerate(pd.read_csv(
                obj['Body'],
                chunksize=chunksize,
                dtype=dtype_dict,
                usecols=lambda x: x in essential_meas_cols,
                nrows=max_rows
            )):
                # Optimize each chunk
                if 'MEASUREMENT_SOURCE_VALUE' in chunk.columns:
                    chunk['MEASUREMENT_SOURCE_VALUE'] = chunk['MEASUREMENT_SOURCE_VALUE'].astype('category')
                
                # Keep only non-null measurements
                chunk = chunk[chunk['MEASUREMENT_SOURCE_VALUE'].notna()]
                
                chunks.append(chunk)
                total_rows_processed += len(chunk)
                
                if len(chunks) >= 10:
                    temp_df = pd.concat(chunks, ignore_index=True)
                    chunks = [temp_df]
                    gc.collect()
                
                if max_rows and total_rows_processed >= max_rows:
                    break
            
            if chunks:
                self.measurements_df = pd.concat(chunks, ignore_index=True)
                del chunks
            else:
                self.measurements_df = pd.DataFrame()
            
            gc.collect()
            
            print(f"  ✓ Loaded {len(self.measurements_df):,} measurement records")
            print(f"  Memory usage: {self.measurements_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
        except Exception as e:
            print(f"  ⚠ Could not load measurements: {e}")
            self.measurements_df = pd.DataFrame()
            gc.collect()
        
        # Final garbage collection
        gc.collect()
        print(f"\nData loading complete. Total memory check:")
        self.get_memory_usage()

    @monitor_memory
    def analyze_demographics(self, MAIN_DIAGNOSIS, df=None):
        """
        Analyze and display demographics distributions
        """
        if df is None:
            df = self.demographics_df
            
        if df.empty:
            print("No demographics data available")
            return
            
        print(f"\n{'='*60}")
        print(f"DEMOGRAPHICS ANALYSIS (n={len(df):,} patients)")
        print(f"{'='*60}")
        
        # Unique patients
        unique_patients = df['PERSON_ID'].nunique()
        print(f"\nUnique patients: {unique_patients:,}")
        
        # Age at Diagnosis calculation
        if ('BIRTH_DATETIME' in df.columns or 'YEAR_OF_BIRTH' in df.columns) and not self.conditions_df.empty:
            print(f"\n--- Age at Diagnosis Distribution ---")
            
            # Define main diagnosis codes (T2D)
            # MAIN_DIAGNOSIS = {
            #     'ICD9': ['250.00', '250.02'],
            #     'ICD10': ['E11.', 'E11.0', 'E11.1', 'E11.2', 'E11.3', 'E11.4', 
            #             'E11.5', 'E11.6', 'E11.7', 'E11.8', 'E11.9']
            # }
            
            # Function to check if a code matches main diagnosis
            def is_main_diagnosis(code):
                if pd.isna(code):
                    return False
                code_str = str(code).strip()
                if '|' in code_str:
                    code = code_str.split('|')[-1].strip()
                else:
                    code = code_str
                code = code.upper()
                
                # Check ICD9 codes
                for pattern in MAIN_DIAGNOSIS['ICD9']:
                    if code == pattern or code.startswith(pattern + '.'):
                        return True
                
                # Check ICD10 codes
                for pattern in MAIN_DIAGNOSIS['ICD10']:
                    if pattern.endswith('.'):
                        if code.startswith(pattern[:-1]):
                            return True
                    else:
                        if code == pattern or code.startswith(pattern + '.'):
                            return True
                return False
            
            # Filter conditions for main diagnosis
            main_diagnosis_conditions = self.conditions_df[
                self.conditions_df['CONDITION_SOURCE_VALUE'].apply(is_main_diagnosis)
            ].copy()
            
            if not main_diagnosis_conditions.empty:
                # Convert condition dates to datetime
                main_diagnosis_conditions['condition_date'] = pd.to_datetime(
                    main_diagnosis_conditions['CONDITION_START_DATE'], 
                    errors='coerce'
                )
                
                # Get earliest diagnosis date per patient
                earliest_diagnosis = main_diagnosis_conditions.groupby('PERSON_ID')['condition_date'].min().reset_index()
                earliest_diagnosis.columns = ['PERSON_ID', 'diagnosis_date']
                
                # Merge with demographics
                df_with_diagnosis = df.merge(earliest_diagnosis, on='PERSON_ID', how='inner')
                
                # Calculate birth year
                if 'BIRTH_DATETIME' in df_with_diagnosis.columns:
                    df_with_diagnosis['birth_date'] = pd.to_datetime(
                        df_with_diagnosis['BIRTH_DATETIME'], 
                        errors='coerce'
                    )
                else:
                    # If only year of birth is available, assume January 1st
                    df_with_diagnosis['birth_date'] = pd.to_datetime(
                        df_with_diagnosis['YEAR_OF_BIRTH'].astype(str) + '-01-01',
                        errors='coerce'
                    )
                
                # Calculate age at diagnosis
                df_with_diagnosis['age_at_diagnosis'] = (
                    (df_with_diagnosis['diagnosis_date'] - df_with_diagnosis['birth_date']).dt.days / 365.25
                )
                
                # Filter valid ages
                valid_ages = df_with_diagnosis['age_at_diagnosis'].dropna()
                valid_ages = valid_ages[(valid_ages >= 0) & (valid_ages <= 120)]  # Sanity check
                
                if len(valid_ages) > 0:
                    print(f"  Patients with T2D diagnosis: {len(valid_ages):,}")
                    print(f"  Mean: {valid_ages.mean():.1f} years")
                    print(f"  Median: {valid_ages.median():.1f} years")
                    print(f"  Std Dev: {valid_ages.std():.1f} years")
                    print(f"  Min: {valid_ages.min():.1f}, Max: {valid_ages.max():.1f}")
                    
                    # Age groups at diagnosis
                    age_groups = pd.cut(valid_ages, 
                                bins=[0, 18, 30, 40, 50, 60, 70, 80, 150],
                                labels=['0-17', '18-29', '30-39', '40-49', 
                                        '50-59', '60-69', '70-79', '80+'])
                    age_dist = age_groups.value_counts().sort_index()
                    print(f"\n  Age Groups at Diagnosis:")
                    for group, count in age_dist.items():
                        pct = (count / len(valid_ages)) * 100
                        print(f"    {group}: {count:,} ({pct:.1f}%)")
                    
                    # Store age at diagnosis in the dataframe for later use
                    df['age_at_diagnosis'] = df['PERSON_ID'].map(
                        df_with_diagnosis.set_index('PERSON_ID')['age_at_diagnosis']
                    )
                else:
                    print("  No valid age at diagnosis data available")
            else:
                print("  No T2D diagnosis records found")
        else:
            if self.conditions_df.empty:
                print("\n--- Age at Diagnosis Distribution ---")
                print("  No conditions data available to calculate age at diagnosis")
            else:
                print("\n--- Age at Diagnosis Distribution ---")
                print("  No birth date information available")
        
        # Gender distribution
        if 'GENDER_CONCEPT_ID' in df.columns:
            print(f"\n--- Gender Distribution ---")
            gender_map = {8507: 'Male', 8532: 'Female'}
            df['gender'] = df['GENDER_CONCEPT_ID'].map(gender_map).fillna('Unknown')
            gender_dist = df['gender'].value_counts()
            for gender, count in gender_dist.items():
                pct = (count / len(df)) * 100
                print(f"  {gender}: {count:,} ({pct:.1f}%)")
        
        # Race distribution
        if 'RACE_CONCEPT_ID' in df.columns:
            print(f"\n--- Race Distribution ---")
            race_map = {
                8527: 'White',
                8516: 'Black',
                8515: 'Asian',
                8657: 'Native American',
                8557: 'Pacific Islander',
                0: 'Unknown'
            }
            df['race'] = df['RACE_CONCEPT_ID'].map(race_map).fillna('Other')
            race_dist = df['race'].value_counts()
            for race, count in race_dist.head(10).items():
                pct = (count / len(df)) * 100
                print(f"  {race}: {count:,} ({pct:.1f}%)")
        
        # Ethnicity distribution
        if 'ETHNICITY_CONCEPT_ID' in df.columns:
            print(f"\n--- Ethnicity Distribution ---")
            ethnicity_map = {
                38003563: 'Hispanic or Latino',
                38003564: 'Not Hispanic or Latino',
                0: 'Unknown'
            }
            df['ethnicity'] = df['ETHNICITY_CONCEPT_ID'].map(ethnicity_map).fillna('Other')
            eth_dist = df['ethnicity'].value_counts()
            for ethnicity, count in eth_dist.items():
                pct = (count / len(df)) * 100
                print(f"  {ethnicity}: {count:,} ({pct:.1f}%)")
    
    @monitor_memory
    def analyze_conditions(self, df=None, top_n=10):
        """
        Analyze condition distributions per encounter and per patient
        """
        if df is None:
            df = self.conditions_df
            
        if df.empty:
            print("No conditions data available")
            return
            
        print(f"\n{'='*60}")
        print(f"CONDITION/ICD CODE ANALYSIS")
        print(f"{'='*60}")
        
        # Per-encounter statistics
        print(f"\n--- PER-ENCOUNTER STATISTICS ---")
        print(f"Total condition encounters: {len(df):,}")
        print(f"Unique patients: {df['PERSON_ID'].nunique():,}")
        print(f"Unique conditions: {df['CONDITION_CONCEPT_ID'].nunique():,}")
        
        # Top conditions by encounter frequency
        print(f"\nTop {top_n} Conditions by Encounter Frequency:")
        if 'CONDITION_SOURCE_VALUE' in df.columns:
            condition_counts = df['CONDITION_SOURCE_VALUE'].value_counts().head(top_n)
            for i, (condition, count) in enumerate(condition_counts.items(), 1):
                pct = (count / len(df)) * 100
                # Truncate long condition names
                condition_str = str(condition)[:60] + '...' if len(str(condition)) > 60 else str(condition)
                print(f"  {i}. {condition_str}: {count:,} ({pct:.1f}%)")
        
        # Per-patient statistics
        print(f"\n--- PER-PATIENT STATISTICS ---")
        patient_condition_counts = df.groupby('PERSON_ID').agg({
            'CONDITION_OCCURRENCE_ID': 'count',
            'CONDITION_CONCEPT_ID': 'nunique'
        }).rename(columns={
            'CONDITION_OCCURRENCE_ID': 'total_conditions',
            'CONDITION_CONCEPT_ID': 'unique_conditions'
        })
        
        print(f"Conditions per patient:")
        print(f"  Mean: {patient_condition_counts['total_conditions'].mean():.1f}")
        print(f"  Median: {patient_condition_counts['total_conditions'].median():.1f}")
        print(f"  Min: {patient_condition_counts['total_conditions'].min()}")
        print(f"  Max: {patient_condition_counts['total_conditions'].max()}")
        
        print(f"\nUnique conditions per patient:")
        print(f"  Mean: {patient_condition_counts['unique_conditions'].mean():.1f}")
        print(f"  Median: {patient_condition_counts['unique_conditions'].median():.1f}")
        print(f"  Min: {patient_condition_counts['unique_conditions'].min()}")
        print(f"  Max: {patient_condition_counts['unique_conditions'].max()}")
        
        # Top conditions by patient count
        print(f"\nTop {top_n} Conditions by Patient Count:")
        if 'CONDITION_SOURCE_VALUE' in df.columns:
            patient_conditions = df.groupby('CONDITION_SOURCE_VALUE')['PERSON_ID'].nunique().sort_values(ascending=False).head(top_n)
            total_patients = df['PERSON_ID'].nunique()
            for i, (condition, count) in enumerate(patient_conditions.items(), 1):
                pct = (count / total_patients) * 100
                condition_str = str(condition)[:60] + '...' if len(str(condition)) > 60 else str(condition)
                print(f"  {i}. {condition_str}: {count:,} patients ({pct:.1f}%)")


    def analyze_conditions_batched(self, df=None, top_n=10, batch_size=100000):
        """
        Analyze condition distributions with batch processing
        """
        if df is None:
            df = self.conditions_df
        
        if df.empty:
            print("No conditions data available")
            return
        
        print(f"\n{'='*60}")
        print(f"CONDITION/ICD CODE ANALYSIS")
        print(f"{'='*60}")
        
        # Process in batches for aggregations
        total_rows = len(df)
        unique_patients = set()
        condition_counter = Counter()
        
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch = df.iloc[start_idx:end_idx]
            
            # Update unique patients
            unique_patients.update(batch['PERSON_ID'].unique())
            
            # Update condition counts
            if 'CONDITION_SOURCE_VALUE' in batch.columns:
                condition_counter.update(batch['CONDITION_SOURCE_VALUE'].value_counts().to_dict())
            
            # Clear batch from memory
            del batch
            gc.collect()
        
        print(f"\nTotal condition encounters: {total_rows:,}")
        print(f"Unique patients: {len(unique_patients):,}")
        
        # Display top conditions
        print(f"\nTop {top_n} Conditions by Encounter Frequency:")
        for i, (condition, count) in enumerate(condition_counter.most_common(top_n), 1):
            pct = (count / total_rows) * 100
            condition_str = str(condition)[:60] + '...' if len(str(condition)) > 60 else str(condition)
            print(f"  {i}. {condition_str}: {count:,} ({pct:.1f}%)")
    
    def analyze_medications(self, df=None, top_n=10):
        """
        Analyze medication distributions per encounter and per patient
        """
        if df is None:
            df = self.medications_df
            
        if df.empty:
            print("No medications data available")
            return
            
        print(f"\n{'='*60}")
        print(f"MEDICATION ANALYSIS")
        print(f"{'='*60}")
        
        # Per-encounter statistics
        print(f"\n--- PER-ENCOUNTER STATISTICS ---")
        print(f"Total medication encounters: {len(df):,}")
        print(f"Unique patients: {df['PERSON_ID'].nunique():,}")
        print(f"Unique medications: {df['DRUG_CONCEPT_ID'].nunique():,}")
        
        # Top medications by prescription frequency
        print(f"\nTop {top_n} Medications by Prescription Frequency:")
        if 'DRUG_SOURCE_VALUE' in df.columns:
            drug_counts = df['DRUG_SOURCE_VALUE'].value_counts().head(top_n)
            for i, (drug, count) in enumerate(drug_counts.items(), 1):
                pct = (count / len(df)) * 100
                drug_str = str(drug)[:50] + '...' if len(str(drug)) > 50 else str(drug)
                print(f"  {i}. {drug_str}: {count:,} ({pct:.1f}%)")
        
        # Per-patient statistics
        print(f"\n--- PER-PATIENT STATISTICS ---")
        patient_drug_counts = df.groupby('PERSON_ID').agg({
            'DRUG_EXPOSURE_ID': 'count',
            'DRUG_CONCEPT_ID': 'nunique'
        }).rename(columns={
            'DRUG_EXPOSURE_ID': 'total_prescriptions',
            'DRUG_CONCEPT_ID': 'unique_medications'
        })
        
        print(f"Prescriptions per patient:")
        print(f"  Mean: {patient_drug_counts['total_prescriptions'].mean():.1f}")
        print(f"  Median: {patient_drug_counts['total_prescriptions'].median():.1f}")
        print(f"  Min: {patient_drug_counts['total_prescriptions'].min()}")
        print(f"  Max: {patient_drug_counts['total_prescriptions'].max()}")
        
        print(f"\nUnique medications per patient:")
        print(f"  Mean: {patient_drug_counts['unique_medications'].mean():.1f}")
        print(f"  Median: {patient_drug_counts['unique_medications'].median():.1f}")
        print(f"  Min: {patient_drug_counts['unique_medications'].min()}")
        print(f"  Max: {patient_drug_counts['unique_medications'].max()}")
        
        # Top medications by patient count
        print(f"\nTop {top_n} Medications by Patient Count:")
        if 'DRUG_SOURCE_VALUE' in df.columns:
            patient_drugs = df.groupby('DRUG_SOURCE_VALUE')['PERSON_ID'].nunique().sort_values(ascending=False).head(top_n)
            total_patients = df['PERSON_ID'].nunique()
            for i, (drug, count) in enumerate(patient_drugs.items(), 1):
                pct = (count / total_patients) * 100
                drug_str = str(drug)[:50] + '...' if len(str(drug)) > 50 else str(drug)
                print(f"  {i}. {drug_str}: {count:,} patients ({pct:.1f}%)")
        
        # Duration statistics
        if 'DRUG_EXPOSURE_START_DATE' in df.columns and 'DRUG_EXPOSURE_END_DATE' in df.columns:
            df['start_date'] = pd.to_datetime(df['DRUG_EXPOSURE_START_DATE'], errors='coerce')
            df['end_date'] = pd.to_datetime(df['DRUG_EXPOSURE_END_DATE'], errors='coerce')
            df['duration_days'] = (df['end_date'] - df['start_date']).dt.days
            
            valid_durations = df['duration_days'].dropna()
            valid_durations = valid_durations[valid_durations > 0]
            
            if len(valid_durations) > 0:
                print(f"\nMedication Duration Statistics:")
                print(f"  Mean: {valid_durations.mean():.1f} days")
                print(f"  Median: {valid_durations.median():.1f} days")
                print(f"  Min: {valid_durations.min():.0f} days")
                print(f"  Max: {valid_durations.max():.0f} days")
    
    def analyze_measurements(self, df=None, top_n=10):
        """
        Analyze measurement distributions per encounter and per patient
        """
        if df is None:
            df = self.measurements_df
            
        if df.empty:
            print("No measurements data available")
            return
            
        print(f"\n{'='*60}")
        print(f"MEASUREMENT ANALYSIS")
        print(f"{'='*60}")
        
        # Per-encounter statistics
        print(f"\n--- PER-ENCOUNTER STATISTICS ---")
        print(f"Total measurement encounters: {len(df):,}")
        print(f"Unique patients: {df['PERSON_ID'].nunique():,}")
        print(f"Unique measurement types: {df['MEASUREMENT_CONCEPT_ID'].nunique():,}")
        
        # Top measurements by frequency
        print(f"\nTop {top_n} Measurements by Frequency:")
        if 'MEASUREMENT_SOURCE_VALUE' in df.columns:
            measurement_counts = df['MEASUREMENT_SOURCE_VALUE'].value_counts().head(top_n)
            for i, (measurement, count) in enumerate(measurement_counts.items(), 1):
                pct = (count / len(df)) * 100
                meas_str = str(measurement)[:50] + '...' if len(str(measurement)) > 50 else str(measurement)
                print(f"  {i}. {meas_str}: {count:,} ({pct:.1f}%)")
        
        # Per-patient statistics
        print(f"\n--- PER-PATIENT STATISTICS ---")
        patient_measurement_counts = df.groupby('PERSON_ID').agg({
            'MEASUREMENT_ID': 'count',
            'MEASUREMENT_CONCEPT_ID': 'nunique'
        }).rename(columns={
            'MEASUREMENT_ID': 'total_measurements',
            'MEASUREMENT_CONCEPT_ID': 'unique_measurements'
        })
        
        print(f"Measurements per patient:")
        print(f"  Mean: {patient_measurement_counts['total_measurements'].mean():.1f}")
        print(f"  Median: {patient_measurement_counts['total_measurements'].median():.1f}")
        print(f"  Min: {patient_measurement_counts['total_measurements'].min()}")
        print(f"  Max: {patient_measurement_counts['total_measurements'].max()}")
        
        print(f"\nUnique measurement types per patient:")
        print(f"  Mean: {patient_measurement_counts['unique_measurements'].mean():.1f}")
        print(f"  Median: {patient_measurement_counts['unique_measurements'].median():.1f}")
        print(f"  Min: {patient_measurement_counts['unique_measurements'].min()}")
        print(f"  Max: {patient_measurement_counts['unique_measurements'].max()}")
        
        # Top measurements by patient count
        print(f"\nTop {top_n} Measurements by Patient Count:")
        if 'MEASUREMENT_SOURCE_VALUE' in df.columns:
            patient_measurements = df.groupby('MEASUREMENT_SOURCE_VALUE')['PERSON_ID'].nunique().sort_values(ascending=False).head(top_n)
            total_patients = df['PERSON_ID'].nunique()
            for i, (measurement, count) in enumerate(patient_measurements.items(), 1):
                pct = (count / total_patients) * 100
                meas_str = str(measurement)[:50] + '...' if len(str(measurement)) > 50 else str(measurement)
                print(f"  {i}. {meas_str}: {count:,} patients ({pct:.1f}%)")
        
        # Value statistics for top measurements
        if 'VALUE_AS_NUMBER' in df.columns:
            print(f"\nValue Statistics for Top 5 Measurements:")
            top_measurements = df['MEASUREMENT_SOURCE_VALUE'].value_counts().head(5).index
            
            for measurement in top_measurements:
                meas_data = df[df['MEASUREMENT_SOURCE_VALUE'] == measurement]['VALUE_AS_NUMBER'].dropna()
                if len(meas_data) > 0:
                    meas_str = str(measurement)[:40] + '...' if len(str(measurement)) > 40 else str(measurement)
                    print(f"\n  {meas_str}:")
                    print(f"    Mean: {meas_data.mean():.2f}")
                    print(f"    Median: {meas_data.median():.2f}")
                    print(f"    Std Dev: {meas_data.std():.2f}")
                    print(f"    Min: {meas_data.min():.2f}, Max: {meas_data.max():.2f}")
    
    def filter_data(self, drug_classes=None, condition_codes=None, measurement_types=None, 
               demographics_filter=None):
        """
        Filter data with memory optimization
        """
        print(f"\n{'='*60}")
        print(f"APPLYING FILTERS")
        print(f"{'='*60}")
        
        # Work with references, not copies initially
        filtered_patient_ids = None
        
        # Filter demographics in-place when possible
        if demographics_filter and not self.demographics_df.empty:
            print("\nApplying demographics filters...")
            
            # Create mask instead of copying
            mask = pd.Series(True, index=self.demographics_df.index)
            
            if 'age_min' in demographics_filter or 'age_max' in demographics_filter:
                # Calculate age only once
                if 'BIRTH_DATETIME' in self.demographics_df.columns:
                    ages = (datetime.now().year - 
                        pd.to_datetime(self.demographics_df['BIRTH_DATETIME'], errors='coerce').dt.year)
                elif 'YEAR_OF_BIRTH' in self.demographics_df.columns:
                    ages = datetime.now().year - self.demographics_df['YEAR_OF_BIRTH']
                else:
                    ages = None
                
                if ages is not None:
                    if 'age_min' in demographics_filter:
                        mask &= (ages >= demographics_filter['age_min'])
                    if 'age_max' in demographics_filter:
                        mask &= (ages <= demographics_filter['age_max'])
                    del ages  # Free memory
            
            # Apply mask
            self.filtered_demographics = self.demographics_df[mask].copy()
            filtered_patient_ids = set(self.filtered_demographics['PERSON_ID'].unique())
            
            # Free memory from mask
            del mask
            gc.collect()
        
        # For large filtering operations, use query() method which is more memory efficient
        if condition_codes and not self.conditions_df.empty:
            print("\nApplying condition filters...")
            
            # Build query string instead of applying function to entire column
            keep_codes = []
            for condition_name, icd_dict in condition_codes.items():
                codes = icd_dict.get('ICD9', []) + icd_dict.get('ICD10', [])
                keep_codes.extend(codes)
            
            # Filter using vectorized operations where possible
            # Process in chunks if dataframe is very large
            if len(self.conditions_df) > 1000000:
                filtered_chunks = []
                for chunk in np.array_split(self.conditions_df, 10):
                    mask = chunk['CONDITION_SOURCE_VALUE'].apply(check_code)
                    filtered_chunks.append(chunk[mask])
                    del chunk, mask
                    gc.collect()
                self.filtered_conditions = pd.concat(filtered_chunks, ignore_index=True)
                del filtered_chunks
            else:
                mask = self.conditions_df['CONDITION_SOURCE_VALUE'].apply(check_code)
                self.filtered_conditions = self.conditions_df[mask].copy()
                del mask
            
            gc.collect()
    
    def compare_distributions(self):
        """
        Compare distributions before and after filtering
        """
        print(f"\n{'='*60}")
        print(f"DISTRIBUTION COMPARISON (BEFORE vs AFTER FILTERING)")
        print(f"{'='*60}")
        
        # Demographics comparison
        if not self.demographics_df.empty and not self.filtered_demographics.empty:
            print("\n--- DEMOGRAPHICS ---")
            print(f"Patients: {self.demographics_df['PERSON_ID'].nunique():,} → "
                  f"{self.filtered_demographics['PERSON_ID'].nunique():,} "
                  f"({self.filtered_demographics['PERSON_ID'].nunique()/self.demographics_df['PERSON_ID'].nunique()*100:.1f}% retained)")
        
        # Conditions comparison
        if not self.conditions_df.empty and not self.filtered_conditions.empty:
            print("\n--- CONDITIONS ---")
            print(f"Records: {len(self.conditions_df):,} → {len(self.filtered_conditions):,} "
                  f"({len(self.filtered_conditions)/len(self.conditions_df)*100:.1f}% retained)")
            print(f"Unique patients: {self.conditions_df['PERSON_ID'].nunique():,} → "
                  f"{self.filtered_conditions['PERSON_ID'].nunique():,}")
            print(f"Unique conditions: {self.conditions_df['CONDITION_CONCEPT_ID'].nunique():,} → "
                  f"{self.filtered_conditions['CONDITION_CONCEPT_ID'].nunique():,}")
        
        # Medications comparison
        if not self.medications_df.empty and not self.filtered_medications.empty:
            print("\n--- MEDICATIONS ---")
            print(f"Records: {len(self.medications_df):,} → {len(self.filtered_medications):,} "
                  f"({len(self.filtered_medications)/len(self.medications_df)*100:.1f}% retained)")
            print(f"Unique patients: {self.medications_df['PERSON_ID'].nunique():,} → "
                  f"{self.filtered_medications['PERSON_ID'].nunique():,}")
            print(f"Unique medications: {self.medications_df['DRUG_CONCEPT_ID'].nunique():,} → "
                  f"{self.filtered_medications['DRUG_CONCEPT_ID'].nunique():,}")
        
        # Measurements comparison
        if not self.measurements_df.empty and not self.filtered_measurements.empty:
            print("\n--- MEASUREMENTS ---")
            print(f"Records: {len(self.measurements_df):,} → {len(self.filtered_measurements):,} "
                  f"({len(self.filtered_measurements)/len(self.measurements_df)*100:.1f}% retained)")
            print(f"Unique patients: {self.measurements_df['PERSON_ID'].nunique():,} → "
                  f"{self.filtered_measurements['PERSON_ID'].nunique():,}")
            print(f"Unique measurements: {self.measurements_df['MEASUREMENT_CONCEPT_ID'].nunique():,} → "
                  f"{self.filtered_measurements['MEASUREMENT_CONCEPT_ID'].nunique():,}")
    
    def save_filtered_data(self, output_prefix, chunksize=100000):
        """
        Save filtered data to S3 with chunked writing for large datasets
        """
        print(f"\n{'='*60}")
        print(f"SAVING FILTERED DATA")
        print(f"{'='*60}")
        
        # For large dataframes, write in chunks
        for name, df, key_suffix in [
            ('demographics', self.filtered_demographics, 'filtered_demographics.csv'),
            ('conditions', self.filtered_conditions, 'filtered_conditions.csv'),
            ('medications', self.filtered_medications, 'filtered_medications.csv'),
            ('measurements', self.filtered_measurements, 'filtered_measurements.csv')
        ]:
            if not df.empty:
                key = f"{output_prefix}{key_suffix}"
                
                if len(df) > 500000:  # Large dataset - use chunked writing
                    print(f"  Writing {name} in chunks...")
                    csv_buffer = StringIO()
                    
                    for i, chunk_start in enumerate(range(0, len(df), chunksize)):
                        chunk_end = min(chunk_start + chunksize, len(df))
                        chunk = df.iloc[chunk_start:chunk_end]
                        
                        # Write header only for first chunk
                        chunk.to_csv(csv_buffer, index=False, header=(i==0), mode='a')
                        
                        # Periodically flush to S3 to avoid memory buildup
                        if csv_buffer.tell() > 50 * 1024 * 1024:  # 50MB buffer
                            self.s3.put_object(
                                Bucket=self.bucket, 
                                Key=key, 
                                Body=csv_buffer.getvalue(),
                                Metadata={'append': 'true'} if i > 0 else {}
                            )
                            csv_buffer = StringIO()
                    
                    # Write remaining buffer
                    if csv_buffer.tell() > 0:
                        self.s3.put_object(Bucket=self.bucket, Key=key, Body=csv_buffer.getvalue())
                else:
                    # Small dataset - write at once
                    csv_buffer = StringIO()
                    df.to_csv(csv_buffer, index=False)
                    self.s3.put_object(Bucket=self.bucket, Key=key, Body=csv_buffer.getvalue())
                
                print(f"  ✓ Saved {name} to s3://{self.bucket}/{key}")

    def get_memory_usage(self):
        """
        Report current memory usage of all dataframes
        """
        total_memory = 0
        print("\n" + "="*60)
        print("MEMORY USAGE REPORT")
        print("="*60)
        
        dfs = [
            ('Demographics', self.demographics_df),
            ('Conditions', self.conditions_df),
            ('Medications', self.medications_df),
            ('Measurements', self.measurements_df),
            ('Filtered Demographics', self.filtered_demographics),
            ('Filtered Conditions', self.filtered_conditions),
            ('Filtered Medications', self.filtered_medications),
            ('Filtered Measurements', self.filtered_measurements)
        ]
        
        for name, df in dfs:
            if df is not None and not df.empty:
                memory_mb = df.memory_usage(deep=True).sum() / 1024**2
                total_memory += memory_mb
                print(f"{name:25s}: {memory_mb:10.2f} MB")
        
        print("-"*60)
        print(f"{'Total':25s}: {total_memory:10.2f} MB")
        print("="*60)
        
        # System memory info
        import psutil
        process = psutil.Process()
        print(f"\nProcess Memory: {process.memory_info().rss / 1024**2:.2f} MB")
        print(f"Available System Memory: {psutil.virtual_memory().available / 1024**2:.2f} MB")

    def cleanup_memory(self, keep_filtered=True):
        """
        Clean up memory by removing unnecessary dataframes
        """
        print("\nCleaning up memory...")
        
        if keep_filtered and self.filtered_demographics is not None:
            del self.demographics_df
            self.demographics_df = pd.DataFrame()
        
        if keep_filtered and self.filtered_conditions is not None:
            del self.conditions_df
            self.conditions_df = pd.DataFrame()
        
        if keep_filtered and self.filtered_medications is not None:
            del self.medications_df
            self.medications_df = pd.DataFrame()
        
        if keep_filtered and self.filtered_measurements is not None:
            del self.measurements_df
            self.measurements_df = pd.DataFrame()
        
        gc.collect()
        print("Memory cleanup complete")

    def run_analysis(self, project_name, drug_classes=None, condition_codes=None, 
                    measurement_types=None, demographics_filter=None, main_diagnosis = None, save_output=True):
        """
        Run complete preprocessing and analysis pipeline
        
        Args:
            project_name: Name of the project in S3
            drug_classes: Optional dictionary of drug classes to filter
            condition_codes: Optional dictionary of condition codes to filter
            measurement_types: Optional dictionary of measurement types to filter
            demographics_filter: Optional demographics filtering criteria
            save_output: Whether to save filtered data to S3
        """
        # Load data
        self.load_data(project_name)
        
        # Analyze raw data
        print(f"\n{'#'*60}")
        print(f"# RAW DATA ANALYSIS")
        print(f"{'#'*60}")
        
        self.analyze_demographics(main_diagnosis)
        self.analyze_conditions()
        self.analyze_medications()
        self.analyze_measurements()
        
        # Apply filters if provided
        if any([drug_classes, condition_codes, measurement_types, demographics_filter]):
            self.filter_data(
                drug_classes=drug_classes,
                condition_codes=condition_codes,
                measurement_types=measurement_types,
                demographics_filter=demographics_filter
            )
            
            # Analyze filtered data
            print(f"\n{'#'*60}")
            print(f"# FILTERED DATA ANALYSIS")
            print(f"{'#'*60}")
            
            self.analyze_demographics(main_diagnosis, self.filtered_demographics)
            self.analyze_conditions(self.filtered_conditions)
            self.analyze_medications(self.filtered_medications)
            self.analyze_measurements(self.filtered_measurements)
            
            # Compare distributions
            self.compare_distributions()
            
            # Save filtered data
            if save_output:
                output_prefix = f"{self.prefix}{project_name}_filtered/"
                self.save_filtered_data(output_prefix)
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*60}\n")


def main():
    """
    Main function to run OMOP preprocessing and analysis
    """
    import sys
    import traceback
    
    # Force output flushing for debugging
    sys.stdout = sys.stderr = open('debug_output.txt', 'w', buffering=1)
    
    try:
        print("Starting OMOP preprocessing...", flush=True)
        
        # ========================================
        # CONFIGURATION SECTION
        # ========================================
        
        # S3 Configuration
        BUCKET = 'dsw-sagemaker-dev-s3'
        PREFIX = 'OMOP_data_extractions/'
        PROJECT_NAME = 'T2D_Tosur'
        
        print(f"Configuration: bucket={BUCKET}, prefix={PREFIX}, project={PROJECT_NAME}", flush=True)
        
        # [Keep your existing filter definitions here]
        # ... DRUG_CLASSES, CONDITION_CODES, etc ...
        
        # ========================================
        # RUN ANALYSIS WITH MEMORY LIMITS
        # ========================================
        
        print("Creating preprocessor instance...", flush=True)
        
        # Create preprocessor instance
        preprocessor = OMOPPreprocessor(bucket=BUCKET, prefix=PREFIX)
        
        print("Starting data load...", flush=True)
        
        # Load data with REDUCED chunk size and row limits for testing
        # Start with smaller limits and increase if successful
        preprocessor.load_data(
            PROJECT_NAME, 
            chunksize=5000,  # Reduced from 50000
            max_rows=100000  # Limit rows for initial testing
        )
        
        print("Data loading complete. Checking memory...", flush=True)
        
        # Check memory after loading
        preprocessor.get_memory_usage()
        
        # Check if we have enough memory to continue
        import psutil
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        if available_memory < 1:
            print(f"WARNING: Low memory available: {available_memory:.2f} GB", flush=True)
            print("Attempting to free memory...", flush=True)
            gc.collect()
        
        print("Optimizing dataframes...", flush=True)
        
        # Optimize memory usage
        preprocessor.optimize_dataframes()
        
        print("Starting analysis...", flush=True)
        
        # Run analysis with your filters
        # Note: Setting save_output=False initially to test without S3 writes
        preprocessor.run_analysis(
            project_name=PROJECT_NAME,
            drug_classes=DRUG_CLASSES,
            condition_codes=CONDITION_CODES,
            measurement_types=MEASUREMENT_TYPES,
            demographics_filter=DEMOGRAPHICS_FILTER,
            main_diagnosis=MAIN_DIAGNOSIS,
            save_output=False  # Disable initially for testing
        )
        
        print("Analysis complete!", flush=True)
        
        # Clean up memory after filtering
        if any([DRUG_CLASSES, CONDITION_CODES, MEASUREMENT_TYPES, DEMOGRAPHICS_FILTER]):
            preprocessor.cleanup_memory(keep_filtered=True)
            preprocessor.get_memory_usage()
        
        return preprocessor
        
    except Exception as e:
        print(f"ERROR occurred: {str(e)}", flush=True)
        print(f"Traceback: {traceback.format_exc()}", flush=True)
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    import sys
    
    # Add immediate output
    print("Script starting...", flush=True)
    
    try:
        preprocessor = main()
        print("Script completed successfully!", flush=True)
    except Exception as e:
        print(f"Script failed with error: {e}", flush=True)
        sys.exit(1)