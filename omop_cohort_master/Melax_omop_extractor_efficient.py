import pandas as pd
import boto3
import os
from io import StringIO, BytesIO
import numpy as np
import gc
import tempfile
import shutil
from collections import Counter
from datetime import datetime
import json
import random

class OMOPDataExtractor:
    """
    Tool for extracting and analyzing OMOP data for a cohort of patients
    """
    
    def __init__(self, source_bucket='dsw-melax-dev-s3', source_prefix='omop/',
                 dest_bucket='dsw-sagemaker-dev-s3', project_name='default_project'):
        """
        Initialize the OMOP Data Extractor
        
        Args:
            source_bucket: S3 bucket containing source OMOP data
            source_prefix: Prefix for OMOP data in source bucket
            dest_bucket: S3 bucket for saving extracted data
            project_name: Name of the project for organizing outputs
        """
        self.source_bucket = source_bucket
        self.source_prefix = source_prefix
        self.dest_bucket = dest_bucket
        self.dest_prefix = f'OMOP_data_extractions/{project_name}/'
        self.s3 = boto3.client('s3')
        
        # Store distributions for analysis
        self.distributions = {}
        
        # Memory management parameters
        self.chunk_size = 25000  # Reduced chunk size
        self.max_samples = 100000  # Maximum samples to keep for distributions
        
    def extract_patient_ids(self, patient_file_path):
        """
        Extract unique patient IDs from the input file
        
        Args:
            patient_file_path: Path to the patient population file
        
        Returns:
            set: Set of unique patient IDs
        """
        print(f"Reading patient population file: {patient_file_path}")
        
        # Determine file type and read accordingly
        if patient_file_path.endswith('.xlsx'):
            df = pd.read_excel(patient_file_path)
        elif patient_file_path.endswith('.csv'):
            df = pd.read_csv(patient_file_path)
        else:
            raise ValueError("Unsupported file format. Use .xlsx or .csv")
        
        # Print column information
        print(f"\nColumns in patient file:")
        for col in df.columns:
            print(f"  - {col}")
        
        # Extract person_id (handle different possible column names)
        id_column = None
        for col in ['person_id', 'PERSON_ID']:
            if col in df.columns:
                id_column = col
                break
        
        if id_column is None:
            raise ValueError("Could not find patient ID column in the input file")
        
        unique_ids = df[id_column].unique()
        print(f"\nFound {len(unique_ids):,} unique patient IDs")
        
        return set(unique_ids)
        
    def process_demographics(self, patient_ids):
        """
        Extract and analyze demographics data with improved memory handling
        """
        print("\n" + "="*60)
        print("PROCESSING DEMOGRAPHICS (person.csv)")
        print("="*60)
        
        table_name = 'person.csv'
        source_key = f"{self.source_prefix}{table_name}"
        dest_key = f"{self.dest_prefix}demographics/{table_name}"
        
        # Use Counters for categorical data only
        gender_counter = Counter()
        race_counter = Counter()
        ethnicity_counter = Counter()
        birth_year_counter = Counter()
        
        try:
            obj = self.s3.get_object(Bucket=self.source_bucket, Key=source_key)
            
            # Use temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
            temp_filename = temp_file.name
            
            first_chunk = True
            total_rows = 0
            filtered_count = 0
            
            for chunk in pd.read_csv(obj['Body'], chunksize=self.chunk_size, low_memory=False):
                total_rows += len(chunk)
                
                # Filter for our patient IDs
                filtered_chunk = chunk[chunk['PERSON_ID'].isin(patient_ids)]
                
                if len(filtered_chunk) > 0:
                    # Write to temp file
                    filtered_chunk.to_csv(temp_file, index=False, header=first_chunk, mode='a')
                    first_chunk = False
                    filtered_count += len(filtered_chunk)
                    
                    # Collect birth year distribution
                    if 'BIRTH_DATETIME' in filtered_chunk.columns:
                        birth_years = pd.to_datetime(filtered_chunk['BIRTH_DATETIME'], errors='coerce').dt.year
                        birth_year_counter.update(birth_years.dropna().astype(int).tolist())
                    elif 'YEAR_OF_BIRTH' in filtered_chunk.columns:
                        birth_year_counter.update(
                            filtered_chunk['YEAR_OF_BIRTH'].dropna().astype(int).tolist()
                        )
                    
                    # Update categorical counters
                    if 'GENDER_CONCEPT_ID' in filtered_chunk.columns:
                        gender_counter.update(filtered_chunk['GENDER_CONCEPT_ID'].dropna().astype(str).tolist())
                    
                    if 'RACE_CONCEPT_ID' in filtered_chunk.columns:
                        race_counter.update(filtered_chunk['RACE_CONCEPT_ID'].dropna().astype(str).tolist())
                    
                    if 'ETHNICITY_CONCEPT_ID' in filtered_chunk.columns:
                        ethnicity_counter.update(filtered_chunk['ETHNICITY_CONCEPT_ID'].dropna().astype(str).tolist())
                
                print(f"  Processed {total_rows:,} rows, found {filtered_count:,} matches", end='\r')
                
                # Explicitly delete chunk to free memory
                del chunk
                del filtered_chunk
                gc.collect()
            
            print()
            temp_file.close()
            
            if filtered_count > 0:
                # Upload temp file to S3
                with open(temp_filename, 'rb') as f:
                    self.s3.put_object(Bucket=self.dest_bucket, Key=dest_key, Body=f)
                
                print(f"  ✓ Saved {filtered_count:,} demographic records")
                
                # Store only top items from counters
                self.distributions['demographics'] = {
                    'birth_year_counter': dict(birth_year_counter.most_common(50)),
                    'gender_counter': dict(gender_counter.most_common(20)),
                    'race_counter': dict(race_counter.most_common(20)),
                    'ethnicity_counter': dict(ethnicity_counter.most_common(20))
                }
                
                self._display_demographics_distributions()
            
            # Clean up temp file
            os.unlink(temp_filename)
            
        except Exception as e:
            print(f"  ✗ Error processing demographics: {str(e)}")
            if 'temp_filename' in locals() and os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    # def process_icd_codes(self, patient_ids):
    #     """
    #     Extract and analyze ICD codes with improved memory handling
    #     """
    #     print("\n" + "="*60)
    #     print("PROCESSING ICD CODES (condition_occurrence.csv)")
    #     print("="*60)
        
    #     table_name = 'condition_occurrence.csv'
    #     source_key = f"{self.source_prefix}{table_name}"
    #     dest_key = f"{self.dest_prefix}icd_codes/{table_name}"
        
    #     # Use Counters instead of lists
    #     condition_concept_counter = Counter()
    #     condition_source_counter = Counter()
        
    #     try:
    #         obj = self.s3.get_object(Bucket=self.source_bucket, Key=source_key)
            
    #         # Use temporary file instead of accumulating in memory
    #         temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
    #         temp_filename = temp_file.name
            
    #         first_chunk = True
    #         total_rows = 0
    #         filtered_count = 0
            
    #         for chunk in pd.read_csv(obj['Body'], chunksize=self.chunk_size, low_memory=False):
    #             total_rows += len(chunk)
                
    #             # Filter for our patient IDs
    #             filtered_chunk = chunk[chunk['PERSON_ID'].isin(patient_ids)]
                
    #             if len(filtered_chunk) > 0:
    #                 # Write to temp file
    #                 filtered_chunk.to_csv(temp_file, index=False, header=first_chunk, mode='a')
    #                 first_chunk = False
    #                 filtered_count += len(filtered_chunk)
                    
    #                 # Update counters directly (not storing lists)
    #                 if 'CONDITION_CONCEPT_ID' in filtered_chunk.columns:
    #                     condition_concept_counter.update(
    #                         filtered_chunk['CONDITION_CONCEPT_ID'].dropna().astype(str).tolist()
    #                     )
                    
    #                 if 'CONDITION_SOURCE_VALUE' in filtered_chunk.columns:
    #                     condition_source_counter.update(
    #                         filtered_chunk['CONDITION_SOURCE_VALUE'].dropna().astype(str).tolist()
    #                     )
                
    #             print(f"  Processed {total_rows:,} rows, found {filtered_count:,} matches", end='\r')
                
    #             # Force garbage collection
    #             del chunk
    #             del filtered_chunk
    #             gc.collect()
            
    #         print()
    #         temp_file.close()
            
    #         if filtered_count > 0:
    #             # Upload temp file to S3
    #             with open(temp_filename, 'rb') as f:
    #                 self.s3.put_object(Bucket=self.dest_bucket, Key=dest_key, Body=f)
                
    #             print(f"  ✓ Saved {filtered_count:,} condition records")
                
    #             # Store only top items from counters
    #             self.distributions['icd_codes'] = {
    #                 'condition_concepts': dict(condition_concept_counter.most_common(100)),
    #                 'condition_sources': dict(condition_source_counter.most_common(100)),
    #                 'total_conditions': sum(condition_concept_counter.values())
    #             }
                
    #             self._display_icd_distributions()
            
    #         # Clean up temp file
    #         os.unlink(temp_filename)
            
    #     except Exception as e:
    #         print(f"  ✗ Error processing ICD codes: {str(e)}")
    #         if 'temp_filename' in locals() and os.path.exists(temp_filename):
    #             os.unlink(temp_filename)
    
    def process_medications(self, patient_ids):
        """
        Extract and analyze medications with improved memory handling
        """
        print("\n" + "="*60)
        print("PROCESSING MEDICATIONS (drug_exposure.csv)")
        print("="*60)
        
        table_name = 'drug_exposure.csv'
        source_key = f"{self.source_prefix}{table_name}"
        dest_key = f"{self.dest_prefix}medications/{table_name}"
        
        # Use Counters instead of lists
        drug_concept_counter = Counter()
        drug_source_counter = Counter()
        
        try:
            obj = self.s3.get_object(Bucket=self.source_bucket, Key=source_key)
            
            # Use temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
            temp_filename = temp_file.name
            
            first_chunk = True
            total_rows = 0
            filtered_count = 0
            
            for chunk in pd.read_csv(obj['Body'], chunksize=self.chunk_size, low_memory=False):
                total_rows += len(chunk)
                
                # Filter for our patient IDs
                filtered_chunk = chunk[chunk['PERSON_ID'].isin(patient_ids)]
                
                if len(filtered_chunk) > 0:
                    # Write to temp file
                    filtered_chunk.to_csv(temp_file, index=False, header=first_chunk, mode='a')
                    first_chunk = False
                    filtered_count += len(filtered_chunk)
                    
                    # Update counters directly
                    if 'DRUG_CONCEPT_ID' in filtered_chunk.columns:
                        drug_concept_counter.update(
                            filtered_chunk['DRUG_CONCEPT_ID'].dropna().astype(str).tolist()
                        )
                    
                    if 'DRUG_SOURCE_VALUE' in filtered_chunk.columns:
                        drug_source_counter.update(
                            filtered_chunk['DRUG_SOURCE_VALUE'].dropna().astype(str).tolist()
                        )
                
                print(f"  Processed {total_rows:,} rows, found {filtered_count:,} matches", end='\r')
                
                # Force garbage collection
                del chunk
                del filtered_chunk
                gc.collect()
            
            print()
            temp_file.close()
            
            if filtered_count > 0:
                # Upload temp file to S3
                with open(temp_filename, 'rb') as f:
                    self.s3.put_object(Bucket=self.dest_bucket, Key=dest_key, Body=f)
                
                print(f"  ✓ Saved {filtered_count:,} medication records")
                
                # Store only top items from counters
                self.distributions['medications'] = {
                    'drug_concepts': dict(drug_concept_counter.most_common(100)),
                    'drug_sources': dict(drug_source_counter.most_common(100)),
                    'total_prescriptions': sum(drug_concept_counter.values())
                }
                
                self._display_medication_distributions()
            
            # Clean up temp file
            os.unlink(temp_filename)
            
        except Exception as e:
            print(f"  ✗ Error processing medications: {str(e)}")
            if 'temp_filename' in locals() and os.path.exists(temp_filename):
                os.unlink(temp_filename)
        
    # def process_measurements(self, patient_ids):
    #     """
    #     Extract and analyze measurements with improved memory handling
    #     """
    #     print("\n" + "="*60)
    #     print("PROCESSING MEASUREMENTS (measurement.csv)")
    #     print("="*60)
        
    #     table_name = 'measurement.csv'
    #     source_key = f"{self.source_prefix}{table_name}"
    #     dest_key = f"{self.dest_prefix}measurements/{table_name}"
        
    #     # Use Counter for concepts
    #     measurement_counter = Counter()
        
    #     # For values, use running statistics with limited sampling
    #     measurement_stats = {}  # concept_id -> {count, sum, sum_sq, min, max, samples}
    #     MAX_SAMPLES_PER_CONCEPT = 500  # Reduced from 1000
        
    #     try:
    #         obj = self.s3.get_object(Bucket=self.source_bucket, Key=source_key)
            
    #         temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
    #         temp_filename = temp_file.name
            
    #         first_chunk = True
    #         total_rows = 0
    #         filtered_count = 0
            
    #         for chunk in pd.read_csv(obj['Body'], chunksize=self.chunk_size, low_memory=False):
    #             total_rows += len(chunk)
                
    #             # Filter for our patient IDs
    #             filtered_chunk = chunk[chunk['PERSON_ID'].isin(patient_ids)]
                
    #             if len(filtered_chunk) > 0:
    #                 # Write to temp file
    #                 filtered_chunk.to_csv(temp_file, index=False, header=first_chunk, mode='a')
    #                 first_chunk = False
    #                 filtered_count += len(filtered_chunk)
                    
    #                 # Update concept counter
    #                 if 'MEASUREMENT_CONCEPT_ID' in filtered_chunk.columns:
    #                     measurement_counter.update(
    #                         filtered_chunk['MEASUREMENT_CONCEPT_ID'].dropna().astype(str).tolist()
    #                     )
                    
    #                 # Process numeric values with running statistics
    #                 if 'VALUE_AS_NUMBER' in filtered_chunk.columns and 'MEASUREMENT_CONCEPT_ID' in filtered_chunk.columns:
    #                     # Process only top concepts to save memory
    #                     top_concepts = set(str(c) for c, _ in measurement_counter.most_common(50))
                        
    #                     for concept_id in filtered_chunk['MEASUREMENT_CONCEPT_ID'].dropna().unique():
    #                         concept_str = str(concept_id)
                            
    #                         # Only track stats for top concepts
    #                         if concept_str not in top_concepts and len(measurement_stats) >= 50:
    #                             continue
                            
    #                         values = filtered_chunk[
    #                             filtered_chunk['MEASUREMENT_CONCEPT_ID'] == concept_id
    #                         ]['VALUE_AS_NUMBER'].dropna()
                            
    #                         if len(values) > 0:
    #                             if concept_str not in measurement_stats:
    #                                 measurement_stats[concept_str] = {
    #                                     'count': 0,
    #                                     'sum': 0,
    #                                     'sum_sq': 0,
    #                                     'min': float('inf'),
    #                                     'max': float('-inf'),
    #                                     'samples': []
    #                                 }
                                
    #                             stats = measurement_stats[concept_str]
                                
    #                             # Update running statistics
    #                             for val in values:
    #                                 stats['count'] += 1
    #                                 stats['sum'] += val
    #                                 stats['sum_sq'] += val ** 2
    #                                 stats['min'] = min(stats['min'], val)
    #                                 stats['max'] = max(stats['max'], val)
                                    
    #                                 # Reservoir sampling with smaller sample size
    #                                 if len(stats['samples']) < MAX_SAMPLES_PER_CONCEPT:
    #                                     stats['samples'].append(val)
    #                                 else:
    #                                     # Random replacement with decreasing probability
    #                                     replace_idx = np.random.randint(0, stats['count'])
    #                                     if replace_idx < MAX_SAMPLES_PER_CONCEPT:
    #                                         stats['samples'][replace_idx] = val
                
    #             print(f"  Processed {total_rows:,} rows, found {filtered_count:,} matches", end='\r')
                
    #             # Force garbage collection
    #             del chunk
    #             del filtered_chunk
    #             gc.collect()
            
    #         print()
    #         temp_file.close()
            
    #         if filtered_count > 0:
    #             # Upload temp file to S3
    #             with open(temp_filename, 'rb') as f:
    #                 self.s3.put_object(Bucket=self.dest_bucket, Key=dest_key, Body=f)
                
    #             print(f"  ✓ Saved {filtered_count:,} measurement records")
                
    #             # Store only statistics, not raw data
    #             self.distributions['measurements'] = {
    #                 'measurement_counter': dict(measurement_counter.most_common(50)),
    #                 'measurement_stats': measurement_stats
    #             }
                
    #             self._display_measurement_distributions()
            
    #         # Clean up
    #         os.unlink(temp_filename)
            
    #     except Exception as e:
    #         print(f"  ✗ Error processing measurements: {str(e)}")
    #         if 'temp_filename' in locals() and os.path.exists(temp_filename):
    #             os.unlink(temp_filename)
    def process_icd_codes(self, patient_ids):
        """
        Extract and analyze ICD codes with improved memory handling and file splitting
        """
        print("\n" + "="*60)
        print("PROCESSING ICD CODES (condition_occurrence.csv)")
        print("="*60)
        
        table_name = 'condition_occurrence.csv'
        source_key = f"{self.source_prefix}{table_name}"
        
        # Use Counters instead of lists
        condition_concept_counter = Counter()
        condition_source_counter = Counter()
        
        # File splitting parameters - much smaller to avoid S3 limit
        MAX_ROWS_PER_FILE = 10_000_000  # Reduced from 100M to 10M rows
        current_part = 1
        rows_in_current_file = 0
        
        try:
            obj = self.s3.get_object(Bucket=self.source_bucket, Key=source_key)
            
            # Initialize first temp file
            temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
            temp_filename = temp_file.name
            temp_files = []  # Keep track of uploaded parts
            
            first_chunk = True
            total_rows = 0
            filtered_count = 0
            
            for chunk in pd.read_csv(obj['Body'], chunksize=self.chunk_size, low_memory=False):
                total_rows += len(chunk)
                
                # Filter for our patient IDs
                filtered_chunk = chunk[chunk['PERSON_ID'].isin(patient_ids)]
                
                if len(filtered_chunk) > 0:
                    # Check if we need to start a new file
                    if rows_in_current_file >= MAX_ROWS_PER_FILE:
                        # Close current file and upload it
                        temp_file.close()
                        
                        # Upload current part to S3
                        dest_key = f"{self.dest_prefix}icd_codes/condition_occurrence_part{current_part:03d}.csv"
                        with open(temp_filename, 'rb') as f:
                            self.s3.put_object(Bucket=self.dest_bucket, Key=dest_key, Body=f)
                        print(f"\n  ✓ Uploaded part {current_part} with {rows_in_current_file:,} rows")
                        temp_files.append(dest_key)
                        
                        # Delete the temp file immediately after upload to save disk space
                        os.unlink(temp_filename)
                        
                        # Start new file
                        current_part += 1
                        rows_in_current_file = 0
                        temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
                        temp_filename = temp_file.name
                        first_chunk = True  # Reset header flag for new file
                    
                    # Write to current temp file
                    filtered_chunk.to_csv(temp_file, index=False, header=first_chunk, mode='a')
                    if first_chunk:
                        first_chunk = False
                    
                    rows_in_current_file += len(filtered_chunk)
                    filtered_count += len(filtered_chunk)
                    
                    # Update counters
                    if 'CONDITION_CONCEPT_ID' in filtered_chunk.columns:
                        condition_concept_counter.update(
                            filtered_chunk['CONDITION_CONCEPT_ID'].dropna().astype(str).tolist()
                        )
                    
                    if 'CONDITION_SOURCE_VALUE' in filtered_chunk.columns:
                        condition_source_counter.update(
                            filtered_chunk['CONDITION_SOURCE_VALUE'].dropna().astype(str).tolist()
                        )
                
                print(f"  Processed {total_rows:,} rows, found {filtered_count:,} matches", end='\r')
                
                # Force garbage collection
                del chunk
                del filtered_chunk
                gc.collect()
            
            print()
            temp_file.close()
            
            if filtered_count > 0:
                # Upload the last file
                if rows_in_current_file > 0:
                    dest_key = f"{self.dest_prefix}icd_codes/condition_occurrence_part{current_part:03d}.csv"
                    
                    with open(temp_filename, 'rb') as f:
                        self.s3.put_object(Bucket=self.dest_bucket, Key=dest_key, Body=f)
                    print(f"  ✓ Uploaded final part {current_part} with {rows_in_current_file:,} rows")
                    temp_files.append(dest_key)
                
                print(f"  ✓ Total: Saved {filtered_count:,} condition records in {current_part} part(s)")
                
                # Store only top items from counters
                self.distributions['icd_codes'] = {
                    'condition_concepts': dict(condition_concept_counter.most_common(100)),
                    'condition_sources': dict(condition_source_counter.most_common(100)),
                    'total_conditions': sum(condition_concept_counter.values()),
                    'parts_created': current_part,
                    'part_files': temp_files
                }
                
                self._display_icd_distributions()
            
            # Clean up the last temp file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
            
        except Exception as e:
            print(f"  ✗ Error processing ICD codes: {str(e)}")
            # Clean up any temp files on error
            if 'temp_filename' in locals() and os.path.exists(temp_filename):
                os.unlink(temp_filename)


    def process_measurements(self, patient_ids):
        """
        Extract and analyze measurements with improved memory handling and file splitting
        """
        print("\n" + "="*60)
        print("PROCESSING MEASUREMENTS (measurement.csv)")
        print("="*60)
        
        table_name = 'measurement.csv'
        source_key = f"{self.source_prefix}{table_name}"
        
        # Use Counter for concepts
        measurement_counter = Counter()
        
        # For values, use running statistics with limited sampling
        measurement_stats = {}
        MAX_SAMPLES_PER_CONCEPT = 500
        
        # File splitting parameters - much smaller to avoid disk space issues
        MAX_ROWS_PER_FILE = 10_000_000  # Reduced from 1B to 10M rows
        current_part = 1
        rows_in_current_file = 0
        
        try:
            obj = self.s3.get_object(Bucket=self.source_bucket, Key=source_key)
            
            # Initialize first temp file
            temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
            temp_filename = temp_file.name
            temp_files = []  # Keep track of uploaded parts
            
            first_chunk = True
            total_rows = 0
            filtered_count = 0
            
            for chunk in pd.read_csv(obj['Body'], chunksize=self.chunk_size, low_memory=False):
                total_rows += len(chunk)
                
                # Filter for our patient IDs
                filtered_chunk = chunk[chunk['PERSON_ID'].isin(patient_ids)]
                
                if len(filtered_chunk) > 0:
                    # Check if we need to start a new file
                    if rows_in_current_file >= MAX_ROWS_PER_FILE:
                        # Close current file and upload it
                        temp_file.close()
                        
                        # Upload current part to S3
                        dest_key = f"{self.dest_prefix}measurements/measurement_part{current_part:03d}.csv"
                        with open(temp_filename, 'rb') as f:
                            self.s3.put_object(Bucket=self.dest_bucket, Key=dest_key, Body=f)
                        print(f"\n  ✓ Uploaded part {current_part} with {rows_in_current_file:,} rows")
                        temp_files.append(dest_key)
                        
                        # Delete the temp file immediately after upload to save disk space
                        os.unlink(temp_filename)
                        
                        # Start new file
                        current_part += 1
                        rows_in_current_file = 0
                        temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
                        temp_filename = temp_file.name
                        first_chunk = True  # Reset header flag for new file
                    
                    # Write to current temp file
                    filtered_chunk.to_csv(temp_file, index=False, header=first_chunk, mode='a')
                    if first_chunk:
                        first_chunk = False
                    
                    rows_in_current_file += len(filtered_chunk)
                    filtered_count += len(filtered_chunk)
                    
                    # Update concept counter
                    if 'MEASUREMENT_CONCEPT_ID' in filtered_chunk.columns:
                        measurement_counter.update(
                            filtered_chunk['MEASUREMENT_CONCEPT_ID'].dropna().astype(str).tolist()
                        )
                    
                    # Process numeric values with running statistics
                    if 'VALUE_AS_NUMBER' in filtered_chunk.columns and 'MEASUREMENT_CONCEPT_ID' in filtered_chunk.columns:
                        # Process only top concepts to save memory
                        top_concepts = set(str(c) for c, _ in measurement_counter.most_common(50))
                        
                        for concept_id in filtered_chunk['MEASUREMENT_CONCEPT_ID'].dropna().unique():
                            concept_str = str(concept_id)
                            
                            # Only track stats for top concepts
                            if concept_str not in top_concepts and len(measurement_stats) >= 50:
                                continue
                            
                            values = filtered_chunk[
                                filtered_chunk['MEASUREMENT_CONCEPT_ID'] == concept_id
                            ]['VALUE_AS_NUMBER'].dropna()
                            
                            if len(values) > 0:
                                if concept_str not in measurement_stats:
                                    measurement_stats[concept_str] = {
                                        'count': 0,
                                        'sum': 0,
                                        'sum_sq': 0,
                                        'min': float('inf'),
                                        'max': float('-inf'),
                                        'samples': []
                                    }
                                
                                stats = measurement_stats[concept_str]
                                
                                # Update running statistics
                                for val in values:
                                    stats['count'] += 1
                                    stats['sum'] += val
                                    stats['sum_sq'] += val ** 2
                                    stats['min'] = min(stats['min'], val)
                                    stats['max'] = max(stats['max'], val)
                                    
                                    # Reservoir sampling
                                    if len(stats['samples']) < MAX_SAMPLES_PER_CONCEPT:
                                        stats['samples'].append(val)
                                    else:
                                        replace_idx = np.random.randint(0, stats['count'])
                                        if replace_idx < MAX_SAMPLES_PER_CONCEPT:
                                            stats['samples'][replace_idx] = val
                
                print(f"  Processed {total_rows:,} rows, found {filtered_count:,} matches", end='\r')
                
                # Force garbage collection
                del chunk
                del filtered_chunk
                gc.collect()
            
            print()
            temp_file.close()
            
            if filtered_count > 0:
                # Upload the last file
                if rows_in_current_file > 0:
                    dest_key = f"{self.dest_prefix}measurements/measurement_part{current_part:03d}.csv"
                    
                    with open(temp_filename, 'rb') as f:
                        self.s3.put_object(Bucket=self.dest_bucket, Key=dest_key, Body=f)
                    print(f"  ✓ Uploaded final part {current_part} with {rows_in_current_file:,} rows")
                    temp_files.append(dest_key)
                
                print(f"  ✓ Total: Saved {filtered_count:,} measurement records in {current_part} part(s)")
                
                # Store only statistics, not raw data
                self.distributions['measurements'] = {
                    'measurement_counter': dict(measurement_counter.most_common(50)),
                    'measurement_stats': measurement_stats,
                    'parts_created': current_part,
                    'part_files': temp_files
                }
                
                self._display_measurement_distributions()
            
            # Clean up the last temp file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
            
        except Exception as e:
            print(f"  ✗ Error processing measurements: {str(e)}")
            # Clean up any temp files on error
            if 'temp_filename' in locals() and os.path.exists(temp_filename):
                os.unlink(temp_filename)   


    def _display_demographics_distributions(self):
        """Display demographics distributions"""
        print("\n--- Demographics Distributions ---")
        
        # Birth year distribution
        if 'birth_year_counter' in self.distributions['demographics']:
            birth_year_data = self.distributions['demographics']['birth_year_counter']
            total = sum(birth_year_data.values())
            if total > 0:
                print(f"\nBirth Year Distribution (n={total:,}):")
                if birth_year_data:
                    print(f"  Earliest: {min(birth_year_data.keys())}")
                    print(f"  Latest: {max(birth_year_data.keys())}")
                    print(f"  Most common birth years:")
                    for year, count in list(birth_year_data.items())[:5]:
                        print(f"    {year}: {count:,} patients ({count/total*100:.1f}%)")
        
        # Gender distribution
        if 'gender_counter' in self.distributions['demographics']:
            gender_data = self.distributions['demographics']['gender_counter']
            total = sum(gender_data.values())
            print(f"\nGender Distribution (n={total:,}):")
            for gender_id, count in list(gender_data.items())[:10]:
                print(f"  Concept ID {gender_id}: {count:,} ({count/total*100:.1f}%)")
        
        # Race distribution
        if 'race_counter' in self.distributions['demographics']:
            race_data = self.distributions['demographics']['race_counter']
            total = sum(race_data.values())
            print(f"\nRace Distribution (n={total:,}):")
            for race_id, count in list(race_data.items())[:10]:
                print(f"  Concept ID {race_id}: {count:,} ({count/total*100:.1f}%)")
        
        # Ethnicity distribution
        if 'ethnicity_counter' in self.distributions['demographics']:
            ethnicity_data = self.distributions['demographics']['ethnicity_counter']
            total = sum(ethnicity_data.values())
            print(f"\nEthnicity Distribution (n={total:,}):")
            for ethnicity_id, count in list(ethnicity_data.items())[:10]:
                print(f"  Concept ID {ethnicity_id}: {count:,} ({count/total*100:.1f}%)")

    def _display_measurement_distributions(self):
        """Display measurement distributions from statistics"""
        print("\n--- Measurement Distributions ---")
        
        # Measurement concepts
        if 'measurement_counter' in self.distributions['measurements']:
            counter_data = self.distributions['measurements']['measurement_counter']
            total = sum(counter_data.values())
            print(f"\nTop 10 Measurement Types (n={total:,} total measurements):")
            for concept_id, count in list(counter_data.items())[:10]:
                print(f"  Concept ID {concept_id}: {count:,} measurements ({count/total*100:.1f}%)")
        
        # Measurement value statistics
        if 'measurement_stats' in self.distributions['measurements']:
            stats_dict = self.distributions['measurements']['measurement_stats']
            
            # Get top 5 concepts by frequency
            top_concepts = sorted(stats_dict.keys(), 
                                key=lambda x: stats_dict[x]['count'], 
                                reverse=True)[:5]
            
            print(f"\nValue Statistics for Top 5 Measurements:")
            for concept_id in top_concepts:
                stats = stats_dict[concept_id]
                if stats['count'] > 0:
                    mean = stats['sum'] / stats['count']
                    variance = (stats['sum_sq'] / stats['count']) - (mean ** 2)
                    std_dev = np.sqrt(max(0, variance))
                    
                    # Calculate median from samples if available
                    median = np.median(stats['samples']) if stats['samples'] else mean
                    
                    print(f"\n  Concept ID {concept_id} (n={stats['count']:,}):")
                    print(f"    Mean: {mean:.2f}")
                    print(f"    Median (sampled): {median:.2f}")
                    print(f"    Std Dev: {std_dev:.2f}")
                    print(f"    Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
    
    def _display_icd_distributions(self):
        """Display ICD code distributions"""
        print("\n--- ICD Code Distributions ---")
        
        # Condition concepts
        if 'condition_concepts' in self.distributions['icd_codes']:
            concepts = self.distributions['icd_codes']['condition_concepts']
            total = self.distributions['icd_codes'].get('total_conditions', 0)
            print(f"\nTop 10 Condition Concepts (n={total:,} total occurrences):")
            for concept_id, count in list(concepts.items())[:10]:
                print(f"  Concept ID {concept_id}: {count:,} occurrences ({count/total*100:.1f}%)")
        
        # Condition source values (ICD codes)
        if 'condition_sources' in self.distributions['icd_codes']:
            sources = self.distributions['icd_codes']['condition_sources']
            total = self.distributions['icd_codes'].get('total_conditions', 0)
            print(f"\nTop 10 ICD Codes (n={total:,} total occurrences):")
            for icd_code, count in list(sources.items())[:10]:
                print(f"  {icd_code}: {count:,} occurrences ({count/total*100:.1f}%)")
    
    def _display_medication_distributions(self):
        """Display medication distributions"""
        print("\n--- Medication Distributions ---")
        
        # Drug concepts
        if 'drug_concepts' in self.distributions['medications']:
            concepts = self.distributions['medications']['drug_concepts']
            total = self.distributions['medications'].get('total_prescriptions', 0)
            print(f"\nTop 10 Drug Concepts (n={total:,} total prescriptions):")
            for concept_id, count in list(concepts.items())[:10]:
                print(f"  Concept ID {concept_id}: {count:,} prescriptions ({count/total*100:.1f}%)")
        
        # Drug source values
        if 'drug_sources' in self.distributions['medications']:
            sources = self.distributions['medications']['drug_sources']
            total = self.distributions['medications'].get('total_prescriptions', 0)
            print(f"\nTop 10 Drug Names (n={total:,} total prescriptions):")
            for drug_name, count in list(sources.items())[:10]:
                print(f"  {drug_name}: {count:,} prescriptions ({count/total*100:.1f}%)")
    
    def save_distribution_summary(self):
        """Save distribution summary to S3"""
        summary = {
            'extraction_date': datetime.now().isoformat(),
            'total_patients': len(self.patient_ids) if hasattr(self, 'patient_ids') else 0,
            'distributions': {}
        }
        
        # Summarize distributions
        for category in self.distributions:
            summary['distributions'][category] = {}
            for key, values in self.distributions[category].items():
                if isinstance(values, dict):
                    # For measurement_stats
                    if key == 'measurement_stats':
                        summary['distributions'][category][key] = {
                            'measurement_types': len(values),
                            'total_measurements': sum(v['count'] for v in values.values())
                        }
                    # For counter dictionaries
                    else:
                        summary['distributions'][category][key] = {
                            'unique_values': len(values),
                            'total_count': sum(values.values()) if values else 0
                        }
                elif isinstance(values, (int, float)):
                    # For total counts
                    summary['distributions'][category][key] = values
        
        # Save to S3
        summary_key = f"{self.dest_prefix}extraction_summary.json"
        self.s3.put_object(
            Bucket=self.dest_bucket,
            Key=summary_key,
            Body=json.dumps(summary, indent=2)
        )
        
        print(f"\n✓ Saved extraction summary to s3://{self.dest_bucket}/{summary_key}")
    
    def run_extraction(self, patient_file_path):
        """
        Run the complete extraction pipeline
        
        Args:
            patient_file_path: Path to the patient population file
        """
        print(f"\n{'='*60}")
        print(f"OMOP DATA EXTRACTION TOOL")
        print(f"{'='*60}")
        print(f"Project: {self.dest_prefix.split('/')[-2]}")
        print(f"Output: s3://{self.dest_bucket}/{self.dest_prefix}")
        
        # Step 1: Extract patient IDs
        self.patient_ids = self.extract_patient_ids(patient_file_path)
        
        # Step 2: Process each data category
        self.process_demographics(self.patient_ids)
        
        # Force garbage collection between large operations
        gc.collect()
        
        self.process_icd_codes(self.patient_ids)
        gc.collect()
        
        self.process_medications(self.patient_ids)
        gc.collect()
        
        self.process_measurements(self.patient_ids)
        gc.collect()
        
        # Step 3: Save distribution summary
        self.save_distribution_summary()
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Processed {len(self.patient_ids):,} patients")
        print(f"✓ Extracted: Demographics, ICD Codes, Medications, Measurements")
        print(f"✓ Saved to: s3://{self.dest_bucket}/{self.dest_prefix}")
        print(f"{'='*60}\n")


def main():
    """
    Main function to run the OMOP data extraction for multiple projects.
    """
    # Configuration
    SOURCE_BUCKET = 'dsw-melax-dev-s3'
    SOURCE_PREFIX = 'omop/'
    DEST_BUCKET = 'dsw-sagemaker-dev-s3'

    # Project configurations
    projects = [
        # {
        #     'project_name': 'T2D_Tosur',
        #     'patient_file': "../data/T2DPopulation_7242025.xlsx"
        # }, #completed!
        # {
        #     'project_name': 'T1D_Tosur',
        #     'patient_file': "../data/CROSSWALK_PATINENTIDS_person_id.csv"
        # },
        # {
        #     'project_name': 'T1D_Mike',
        #     'patient_file': "../data/Mike_T1D_person_id.csv"
        # }
        # {
        #     'project_name': 'Rett_syndrome',
        #     'patient_file': "../data/Rett_syndrome/Rett_syndrome_subset_person_id.csv"
        # },
        {
            'project_name': 'Melax_Vishnu',
            'patient_file': "../data/Melax_sleeping/Melax_person_id.csv"
        }

        # Add more projects as needed
    ]

    for project in projects:
        # Create extractor instance for each project
        extractor = OMOPDataExtractor(
            source_bucket=SOURCE_BUCKET,
            source_prefix=SOURCE_PREFIX,
            dest_bucket=DEST_BUCKET,
            project_name=project['project_name']
        )

        # Run extraction for the current project
        print(f"Running extraction for project: {project['project_name']}")
        extractor.run_extraction(project['patient_file'])

if __name__ == "__main__":
    main()