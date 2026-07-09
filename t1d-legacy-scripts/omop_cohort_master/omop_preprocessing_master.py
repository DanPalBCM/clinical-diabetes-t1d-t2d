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
        self.demographics_df = None
        self.conditions_df = None
        self.medications_df = None
        self.measurements_df = None
        
        # Store filtered dataframes
        self.filtered_demographics = None
        self.filtered_conditions = None
        self.filtered_medications = None
        self.filtered_measurements = None
        
    def load_data(self, project_name):
        """
        Load OMOP data from S3
        
        Args:
            project_name: Name of the project folder in S3
        """
        print(f"\n{'='*60}")
        print(f"LOADING OMOP DATA")
        print(f"{'='*60}")
        
        full_prefix = f"{self.prefix}{project_name}/"
        
        try:
            # Load demographics
            print("Loading demographics...")
            demo_key = f"{full_prefix}demographics/person.csv"
            obj = self.s3.get_object(Bucket=self.bucket, Key=demo_key)
            self.demographics_df = pd.read_csv(obj['Body'])
            print(f"  ✓ Loaded {len(self.demographics_df):,} patient records")
            
        except Exception as e:
            print(f"  ⚠ Could not load demographics: {e}")
            self.demographics_df = pd.DataFrame()
        
        try:
            # Load conditions
            print("Loading conditions...")
            cond_key = f"{full_prefix}icd_codes/condition_occurrence.csv"
            obj = self.s3.get_object(Bucket=self.bucket, Key=cond_key)
            self.conditions_df = pd.read_csv(obj['Body'])
            print(f"  ✓ Loaded {len(self.conditions_df):,} condition records")
            
        except Exception as e:
            print(f"  ⚠ Could not load conditions: {e}")
            self.conditions_df = pd.DataFrame()
        
        try:
            # Load medications
            print("Loading medications...")
            med_key = f"{full_prefix}medications/drug_exposure.csv"
            obj = self.s3.get_object(Bucket=self.bucket, Key=med_key)
            self.medications_df = pd.read_csv(obj['Body'])
            print(f"  ✓ Loaded {len(self.medications_df):,} medication records")
            
        except Exception as e:
            print(f"  ⚠ Could not load medications: {e}")
            self.medications_df = pd.DataFrame()
        
        try:
            # Load measurements
            print("Loading measurements...")
            meas_key = f"{full_prefix}measurements/measurement.csv"
            obj = self.s3.get_object(Bucket=self.bucket, Key=meas_key)
            self.measurements_df = pd.read_csv(obj['Body'])
            print(f"  ✓ Loaded {len(self.measurements_df):,} measurement records")
            
        except Exception as e:
            print(f"  ⚠ Could not load measurements: {e}")
            self.measurements_df = pd.DataFrame()
        
        print(f"{'='*60}\n")
    
    def analyze_demographics(self, df=None):
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
        
        # Age distribution
        if 'BIRTH_DATETIME' in df.columns or 'YEAR_OF_BIRTH' in df.columns:
            if 'BIRTH_DATETIME' in df.columns:
                df['age'] = (datetime.now().year - 
                           pd.to_datetime(df['BIRTH_DATETIME'], errors='coerce').dt.year)
            else:
                df['age'] = datetime.now().year - df['YEAR_OF_BIRTH']
            
            valid_ages = df['age'].dropna()
            if len(valid_ages) > 0:
                print(f"\n--- Age Distribution ---")
                print(f"  Mean: {valid_ages.mean():.1f} years")
                print(f"  Median: {valid_ages.median():.1f} years")
                print(f"  Std Dev: {valid_ages.std():.1f} years")
                print(f"  Min: {valid_ages.min():.0f}, Max: {valid_ages.max():.0f}")
                
                # Age groups
                age_groups = pd.cut(valid_ages, 
                                   bins=[0, 18, 30, 40, 50, 60, 70, 80, 150],
                                   labels=['0-17', '18-29', '30-39', '40-49', 
                                          '50-59', '60-69', '70-79', '80+'])
                age_dist = age_groups.value_counts().sort_index()
                print(f"\n  Age Groups:")
                for group, count in age_dist.items():
                    pct = (count / len(valid_ages)) * 100
                    print(f"    {group}: {count:,} ({pct:.1f}%)")
        
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
        Filter data based on user-defined criteria
        
        Args:
            drug_classes: Dictionary of drug classes and their medications
            condition_codes: Dictionary of condition categories and their ICD codes
            measurement_types: Dictionary of measurement categories and their types
            demographics_filter: Dictionary with demographic criteria
        """
        print(f"\n{'='*60}")
        print(f"APPLYING FILTERS")
        print(f"{'='*60}")
        
        # Initialize filtered dataframes with copies of original
        self.filtered_demographics = self.demographics_df.copy() if not self.demographics_df.empty else pd.DataFrame()
        self.filtered_conditions = self.conditions_df.copy() if not self.conditions_df.empty else pd.DataFrame()
        self.filtered_medications = self.medications_df.copy() if not self.medications_df.empty else pd.DataFrame()
        self.filtered_measurements = self.measurements_df.copy() if not self.measurements_df.empty else pd.DataFrame()
        
        filtered_patient_ids = None
        
        # Filter demographics
        if demographics_filter and not self.filtered_demographics.empty:
            print("\nApplying demographics filters...")
            
            if 'age_min' in demographics_filter or 'age_max' in demographics_filter:
                if 'BIRTH_DATETIME' in self.filtered_demographics.columns:
                    self.filtered_demographics['age'] = (datetime.now().year - 
                                                        pd.to_datetime(self.filtered_demographics['BIRTH_DATETIME'], errors='coerce').dt.year)
                elif 'YEAR_OF_BIRTH' in self.filtered_demographics.columns:
                    self.filtered_demographics['age'] = datetime.now().year - self.filtered_demographics['YEAR_OF_BIRTH']
                
                if 'age_min' in demographics_filter:
                    self.filtered_demographics = self.filtered_demographics[
                        self.filtered_demographics['age'] >= demographics_filter['age_min']
                    ]
                    print(f"  - Age >= {demographics_filter['age_min']}")
                
                if 'age_max' in demographics_filter:
                    self.filtered_demographics = self.filtered_demographics[
                        self.filtered_demographics['age'] <= demographics_filter['age_max']
                    ]
                    print(f"  - Age <= {demographics_filter['age_max']}")
            
            if 'gender' in demographics_filter:
                gender_map = {'M': 8507, 'Male': 8507, 'F': 8532, 'Female': 8532}
                gender_ids = [gender_map.get(g, g) for g in demographics_filter['gender']]
                self.filtered_demographics = self.filtered_demographics[
                    self.filtered_demographics['GENDER_CONCEPT_ID'].isin(gender_ids)
                ]
                print(f"  - Gender in {demographics_filter['gender']}")
            
            if 'race' in demographics_filter:
                race_map = {
                    'White': 8527,
                    'Black': 8516,
                    'Asian': 8515,
                    'Native American': 8657,
                    'Pacific Islander': 8557
                }
                race_ids = [race_map.get(r, r) for r in demographics_filter['race']]
                self.filtered_demographics = self.filtered_demographics[
                    self.filtered_demographics['RACE_CONCEPT_ID'].isin(race_ids)
                ]
                print(f"  - Race in {demographics_filter['race']}")
            
            filtered_patient_ids = set(self.filtered_demographics['PERSON_ID'].unique())
            print(f"  ✓ {len(filtered_patient_ids):,} patients after demographic filtering")
        
        # Filter medications
        if drug_classes and not self.filtered_medications.empty:
            print("\nApplying medication filters...")
            
            # Create a list of all drug names to keep
            keep_drugs = []
            for drug_class, drug_list in drug_classes.items():
                keep_drugs.extend(drug_list)
                print(f"  - Including {drug_class}: {len(drug_list)} drugs")
            
            # Filter medications
            pattern = '|'.join(keep_drugs)
            mask = self.filtered_medications['DRUG_SOURCE_VALUE'].str.lower().str.contains(
                pattern, na=False, regex=True
            )
            self.filtered_medications = self.filtered_medications[mask]
            
            # Get patient IDs with these medications
            med_patient_ids = set(self.filtered_medications['PERSON_ID'].unique())
            
            if filtered_patient_ids is not None:
                filtered_patient_ids = filtered_patient_ids.intersection(med_patient_ids)
            else:
                filtered_patient_ids = med_patient_ids
            
            print(f"  ✓ {len(self.filtered_medications):,} medication records")
            print(f"  ✓ {len(med_patient_ids):,} patients with target medications")
        
        # Filter conditions
        if condition_codes and not self.filtered_conditions.empty:
            print("\nApplying condition filters...")
            
            # Create a list of all ICD codes to keep
            keep_codes = []
            for condition_name, icd_dict in condition_codes.items():
                codes = icd_dict.get('ICD9', []) + icd_dict.get('ICD10', [])
                keep_codes.extend(codes)
                print(f"  - Including {condition_name}: {len(codes)} codes")
            
            # Filter conditions using the check_icd_code function
            def check_code(code):
                if pd.isna(code):
                    return False
                code_str = str(code).strip()
                if '|' in code_str:
                    code = code_str.split('|')[-1].strip()
                else:
                    code = code_str
                code = code.upper()
                
                for pattern in keep_codes:
                    if pattern.endswith('.'):
                        if code.startswith(pattern[:-1]):
                            return True
                    else:
                        if code == pattern or code.startswith(pattern + '.'):
                            return True
                return False
            
            mask = self.filtered_conditions['CONDITION_SOURCE_VALUE'].apply(check_code)
            self.filtered_conditions = self.filtered_conditions[mask]
            
            # Get patient IDs with these conditions
            cond_patient_ids = set(self.filtered_conditions['PERSON_ID'].unique())
            
            if filtered_patient_ids is not None:
                filtered_patient_ids = filtered_patient_ids.intersection(cond_patient_ids)
            else:
                filtered_patient_ids = cond_patient_ids
            
            print(f"  ✓ {len(self.filtered_conditions):,} condition records")
            print(f"  ✓ {len(cond_patient_ids):,} patients with target conditions")
        
        # Filter measurements
        if measurement_types and not self.filtered_measurements.empty:
            print("\nApplying measurement filters...")
            
            # Create pattern for measurement filtering
            keep_measurements = []
            for meas_name, meas_config in measurement_types.items():
                keep_measurements.extend(meas_config.get('keywords', []))
                print(f"  - Including {meas_name}")
            
            if keep_measurements:
                pattern = '|'.join(keep_measurements)
                mask = self.filtered_measurements['MEASUREMENT_SOURCE_VALUE'].str.lower().str.contains(
                    pattern, na=False, regex=True
                )
                self.filtered_measurements = self.filtered_measurements[mask]
            
            # Get patient IDs with these measurements
            meas_patient_ids = set(self.filtered_measurements['PERSON_ID'].unique())
            
            if filtered_patient_ids is not None:
                filtered_patient_ids = filtered_patient_ids.intersection(meas_patient_ids)
            else:
                filtered_patient_ids = meas_patient_ids
            
            print(f"  ✓ {len(self.filtered_measurements):,} measurement records")
            print(f"  ✓ {len(meas_patient_ids):,} patients with target measurements")
        
        # Apply final patient ID filter to all dataframes
        if filtered_patient_ids is not None:
            print(f"\nApplying final patient filter...")
            print(f"  Total patients after all filters: {len(filtered_patient_ids):,}")
            
            if not self.filtered_demographics.empty:
                self.filtered_demographics = self.filtered_demographics[
                    self.filtered_demographics['PERSON_ID'].isin(filtered_patient_ids)
                ]
            
            if not self.filtered_conditions.empty:
                self.filtered_conditions = self.filtered_conditions[
                    self.filtered_conditions['PERSON_ID'].isin(filtered_patient_ids)
                ]
            
            if not self.filtered_medications.empty:
                self.filtered_medications = self.filtered_medications[
                    self.filtered_medications['PERSON_ID'].isin(filtered_patient_ids)
                ]
            
            if not self.filtered_measurements.empty:
                self.filtered_measurements = self.filtered_measurements[
                    self.filtered_measurements['PERSON_ID'].isin(filtered_patient_ids)
                ]
        
        print(f"{'='*60}\n")
    
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
    
    def save_filtered_data(self, output_prefix):
        """
        Save filtered data to S3
        
        Args:
            output_prefix: Prefix for output files in S3
        """
        print(f"\n{'='*60}")
        print(f"SAVING FILTERED DATA")
        print(f"{'='*60}")
        
        # Save demographics
        if not self.filtered_demographics.empty:
            demo_key = f"{output_prefix}filtered_demographics.csv"
            csv_buffer = StringIO()
            self.filtered_demographics.to_csv(csv_buffer, index=False)
            self.s3.put_object(Bucket=self.bucket, Key=demo_key, Body=csv_buffer.getvalue())
            print(f"  ✓ Saved demographics to s3://{self.bucket}/{demo_key}")
        
        # Save conditions
        if not self.filtered_conditions.empty:
            cond_key = f"{output_prefix}filtered_conditions.csv"
            csv_buffer = StringIO()
            self.filtered_conditions.to_csv(csv_buffer, index=False)
            self.s3.put_object(Bucket=self.bucket, Key=cond_key, Body=csv_buffer.getvalue())
            print(f"  ✓ Saved conditions to s3://{self.bucket}/{cond_key}")
        
        # Save medications
        if not self.filtered_medications.empty:
            med_key = f"{output_prefix}filtered_medications.csv"
            csv_buffer = StringIO()
            self.filtered_medications.to_csv(csv_buffer, index=False)
            self.s3.put_object(Bucket=self.bucket, Key=med_key, Body=csv_buffer.getvalue())
            print(f"  ✓ Saved medications to s3://{self.bucket}/{med_key}")
        
        # Save measurements
        if not self.filtered_measurements.empty:
            meas_key = f"{output_prefix}filtered_measurements.csv"
            csv_buffer = StringIO()
            self.filtered_measurements.to_csv(csv_buffer, index=False)
            self.s3.put_object(Bucket=self.bucket, Key=meas_key, Body=csv_buffer.getvalue())
            print(f"  ✓ Saved measurements to s3://{self.bucket}/{meas_key}")
        
        print(f"{'='*60}\n")
    
    def run_analysis(self, project_name, drug_classes=None, condition_codes=None, 
                    measurement_types=None, demographics_filter=None, save_output=True):
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
        
        self.analyze_demographics()
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
            
            self.analyze_demographics(self.filtered_demographics)
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
    # ========================================
    # CONFIGURATION SECTION
    # ========================================
    
    # S3 Configuration
    BUCKET = 'dsw-sagemaker-dev-s3'
    PREFIX = 'OMOP_data_extractions/'
    PROJECT_NAME = 'T2D_Tosur'  # Change this to your project name
    
    # ========================================
    # OPTIONAL USER-DEFINED FILTERS
    # ========================================
    # Set any of these to None to skip that filter
    
    # Example: Filter for specific drug classes (optional)
    DRUG_CLASSES = {
        'Metformin': ['metformin'],
        'Insulins': [
            'insulin aspart', 'insulin degludec', 'insulin detemir', 
            'insulin glargine', 'insulin glulisine', 'insulin human'
        ],
        'GLP1_agonists': [
            'dulaglutide', 'exenatide', 'liraglutide', 'semaglutide'
        ]
    }
    # Set to None to include all drugs:
    # DRUG_CLASSES = None
    
    # Example: Filter for specific conditions (optional)
    CONDITION_CODES = {
        'Diabetes': {
            'ICD9': ['250.'],
            'ICD10': ['E10.', 'E11.']
        },
        'Hypertension': {
            'ICD9': ['401.', '402.'],
            'ICD10': ['I10.', 'I11.']
        }
    }
    # Set to None to include all conditions:
    # CONDITION_CODES = None
    
    # Example: Filter for specific measurements (optional)
    MEASUREMENT_TYPES = {
        'hba1c': {
            'keywords': ['hba1c', 'hemoglobin a1c', 'a1c', 'glycated hemoglobin']
        },
        'glucose': {
            'keywords': ['glucose', 'blood sugar', 'fasting glucose']
        },
        'blood_pressure': {
            'keywords': ['systolic', 'diastolic', 'blood pressure']
        }
    }
    # Set to None to include all measurements:
    # MEASUREMENT_TYPES = None
    
    # Example: Demographics filter (optional)
    DEMOGRAPHICS_FILTER = {
        'age_min': 18,
        'age_max': 85,
        'gender': ['Male', 'Female'],  # or use concept IDs: [8507, 8532]
        # 'race': ['White', 'Black', 'Asian']  # optional
    }
    # Set to None to include all demographics:
    # DEMOGRAPHICS_FILTER = None
    
    # ========================================
    # RUN ANALYSIS
    # ========================================
    
    # Create preprocessor instance
    preprocessor = OMOPPreprocessor(bucket=BUCKET, prefix=PREFIX)
    
    # Run analysis with optional filtering
    preprocessor.run_analysis(
        project_name=PROJECT_NAME,
        drug_classes=DRUG_CLASSES,
        condition_codes=CONDITION_CODES,
        measurement_types=MEASUREMENT_TYPES,
        demographics_filter=DEMOGRAPHICS_FILTER,
        save_output=True  # Set to False if you don't want to save filtered data
    )
    
    return preprocessor


if __name__ == "__main__":
    preprocessor = main()