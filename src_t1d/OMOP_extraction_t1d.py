import pandas as pd
import boto3
import os
from io import StringIO
import numpy as np
import gc
import tempfile
import shutil
from datetime import datetime

def preprocess_t1d_patients():
    """
    Preprocess T1D patient lists:
    1. Filter CROSSWALK patients to only include those in t1d_latest
    2. Return both filtered T1D list and T1D Hypoglycemia list
    """
    print("=" * 60)
    print("PREPROCESSING T1D PATIENT LISTS")
    print("=" * 60)
    
    # Read the CROSSWALK file
    print("\n1. Reading CROSSWALK_PATINENTIDS.csv...")
    crosswalk_path = "/home/sagemaker-user/T2D/data/CROSSWALK_PATINENTIDS.csv"
    crosswalk_df = pd.read_csv(crosswalk_path)
    print(f"   Total patients in CROSSWALK: {len(crosswalk_df)}")
    print(f"   Columns: {list(crosswalk_df.columns)}")
    
    # Read the t1d_latest file (filtered list)
    print("\n2. Reading t1d_latest.xlsx...")
    t1d_latest_path = "/home/sagemaker-user/T2D/data/t1d_latest.xlsx"
    t1d_latest_df = pd.read_excel(t1d_latest_path)
    print(f"   Total patients in t1d_latest: {len(t1d_latest_df)}")
    print(f"   Columns: {list(t1d_latest_df.columns)}")
    
    # Get the PAT_IDs from t1d_latest (these match PATIENTID in CROSSWALK)
    t1d_latest_ids = set(t1d_latest_df['PAT_ID'].unique())
    print(f"   Unique PAT_IDs in t1d_latest: {len(t1d_latest_ids)}")
    
    # Filter CROSSWALK to only include patients whose PATIENTID is in t1d_latest PAT_IDs
    print("\n3. Filtering CROSSWALK patients...")
    t1d_filtered_df = crosswalk_df[crosswalk_df['PATIENTID'].isin(t1d_latest_ids)]
    print(f"   Filtered T1D patients: {len(t1d_filtered_df)}")
    print(f"   Removed {len(crosswalk_df) - len(t1d_filtered_df)} patients not in t1d_latest")
    
    # Get unique PEDSNET_IDs (these are person_ids in OMOP)
    t1d_person_ids = set(t1d_filtered_df['PEDSNET_ID'].unique())
    print(f"   Unique PEDSNET_IDs for extraction: {len(t1d_person_ids)}")
    
    # Read the T1D Hypoglycemia file
    print("\n4. Reading T1D Hypoglycemia patients...")
    t1d_hypo_path = "/home/sagemaker-user/T2D/data/CROSSWALK_PATINENTIDS_08052025.csv"
    t1d_hypo_df = pd.read_csv(t1d_hypo_path)
    print(f"   Total T1D Hypoglycemia patients: {len(t1d_hypo_df)}")
    print(f"   Columns: {list(t1d_hypo_df.columns)}")
    
    # Get unique PEDSNET_IDs for hypoglycemia patients
    t1d_hypo_person_ids = set(t1d_hypo_df['PEDSNET_ID'].unique())
    print(f"   Unique PEDSNET_IDs for hypoglycemia: {len(t1d_hypo_person_ids)}")
    
    return t1d_person_ids, t1d_hypo_person_ids, t1d_filtered_df, t1d_hypo_df


def extract_omop_data_for_cohort(person_ids, cohort_name, dest_prefix, s3, 
                                 source_bucket='dsw-melax-dev-s3', 
                                 dest_bucket='dsw-sagemaker-dev-s3'):
    """
    Extract OMOP data for a specific cohort of patients
    """
    print(f"\n{'=' * 60}")
    print(f"EXTRACTING DATA FOR: {cohort_name}")
    print(f"{'=' * 60}")
    print(f"Processing {len(person_ids)} unique person IDs")
    print(f"Destination: s3://{dest_bucket}/{dest_prefix}")
    
    # Convert to set for faster lookup
    person_id_set = set(person_ids)
    
    # Tables to process with their ID columns and expected dtypes
    tables = [
        ('person.csv', 'PERSON_ID', None),
        ('condition_occurrence.csv', 'PERSON_ID', None),
        ('drug_exposure.csv', 'PERSON_ID', {
            'DRUG_EXPOSURE_ID': 'int64',
            'PERSON_ID': 'int64',
            'DRUG_CONCEPT_ID': 'int64',
            'DRUG_EXPOSURE_START_DATE': 'str',
            'DRUG_EXPOSURE_START_DATETIME': 'str',
            'DRUG_EXPOSURE_END_DATE': 'str',
            'DRUG_EXPOSURE_END_DATETIME': 'str',
            'VERBATIM_END_DATE': 'str',
            'DRUG_TYPE_CONCEPT_ID': 'int64',
            'STOP_REASON': 'str',
            'REFILLS': 'float64',
            'QUANTITY': 'float64',
            'DAYS_SUPPLY': 'float64',
            'SIG': 'str',
            'ROUTE_CONCEPT_ID': 'float64',
            'LOT_NUMBER': 'str',
            'PROVIDER_ID': 'float64',
            'VISIT_OCCURRENCE_ID': 'float64',
            'VISIT_DETAIL_ID': 'float64',
            'DRUG_SOURCE_VALUE': 'str',
            'DRUG_SOURCE_CONCEPT_ID': 'float64',
            'ROUTE_SOURCE_VALUE': 'str',
            'DOSE_UNIT_SOURCE_VALUE': 'str'
        }),
        #('measurement.csv', 'PERSON_ID', None),
        #('observation.csv', 'PERSON_ID', None),  # Added observation table
        #('procedure_occurrence.csv', 'PERSON_ID', None),  # Added procedure table
        #('visit_occurrence.csv', 'PERSON_ID', None)  # Added visit table
    ]
    
    source_prefix = 'omop/'
    results = {}
    
    # Process each table
    for table_name, id_column, dtype_dict in tables:
        print(f"\n📊 Processing {table_name}...")
        
        try:
            # Create a temporary directory for intermediate files
            temp_dir = tempfile.mkdtemp()
            temp_file_paths = []
            
            # Read from S3 in chunks to handle large files
            source_key = f"{source_prefix}{table_name}"
            print(f"   Reading from s3://{source_bucket}/{source_key}")
            
            # Get object to read in chunks
            obj = s3.get_object(Bucket=source_bucket, Key=source_key)
            
            # Process in smaller chunks to reduce memory usage
            chunk_size = 50000
            total_rows = 0
            filtered_rows = 0
            chunk_count = 0
            
            # Read CSV in chunks
            chunk_iterator = pd.read_csv(
                obj['Body'], 
                chunksize=chunk_size,
                dtype=dtype_dict,
                low_memory=False if dtype_dict is None else True
            )
            
            for chunk in chunk_iterator:
                total_rows += len(chunk)
                
                # Filter chunk to include only our person IDs
                filtered_chunk = chunk[chunk[id_column].isin(person_id_set)]
                filtered_rows += len(filtered_chunk)
                
                if len(filtered_chunk) > 0:
                    # Save filtered chunk to temporary file
                    temp_file = os.path.join(temp_dir, f'chunk_{chunk_count}.csv')
                    filtered_chunk.to_csv(temp_file, index=False)
                    temp_file_paths.append(temp_file)
                    chunk_count += 1
                
                # Clear memory
                del chunk
                del filtered_chunk
                
                # Force garbage collection every 10 chunks
                if chunk_count % 10 == 0:
                    gc.collect()
                
                print(f"   Processed {total_rows:,} rows, kept {filtered_rows:,} rows", end='\r')
            
            print()  # New line after progress
            
            if temp_file_paths:
                # Upload chunks directly to S3 using multipart upload
                dest_key = f"{dest_prefix}{table_name}"
                print(f"   Uploading to s3://{dest_bucket}/{dest_key}")
                
                # Initialize multipart upload
                multipart_upload = s3.create_multipart_upload(
                    Bucket=dest_bucket,
                    Key=dest_key
                )
                
                parts = []
                part_number = 1
                
                try:
                    # First, upload the header
                    first_chunk = pd.read_csv(temp_file_paths[0], nrows=0)
                    header_bytes = ','.join(first_chunk.columns).encode('utf-8') + b'\n'
                    
                    # Create a buffer for accumulating data
                    buffer = header_bytes
                    min_part_size = 5 * 1024 * 1024  # 5MB minimum part size for S3
                    
                    # Process each temporary file
                    for i, temp_file in enumerate(temp_file_paths):
                        print(f"   Uploading chunk {i+1}/{len(temp_file_paths)}", end='\r')
                        
                        # Read chunk without header
                        chunk_df = pd.read_csv(temp_file)
                        chunk_bytes = chunk_df.to_csv(index=False, header=False).encode('utf-8')
                        
                        buffer += chunk_bytes
                        
                        # If buffer is large enough, upload as a part
                        if len(buffer) >= min_part_size or i == len(temp_file_paths) - 1:
                            response = s3.upload_part(
                                Bucket=dest_bucket,
                                Key=dest_key,
                                PartNumber=part_number,
                                UploadId=multipart_upload['UploadId'],
                                Body=buffer
                            )
                            
                            parts.append({
                                'ETag': response['ETag'],
                                'PartNumber': part_number
                            })
                            
                            part_number += 1
                            buffer = b''
                        
                        # Remove processed temp file to free disk space
                        os.remove(temp_file)
                        gc.collect()
                    
                    print()  # New line after progress
                    
                    # Complete multipart upload
                    s3.complete_multipart_upload(
                        Bucket=dest_bucket,
                        Key=dest_key,
                        UploadId=multipart_upload['UploadId'],
                        MultipartUpload={'Parts': parts}
                    )
                    
                    print(f"   ✅ Successfully uploaded {table_name} with {filtered_rows:,} rows")
                    results[table_name] = {'status': 'success', 'rows': filtered_rows}
                    
                except Exception as e:
                    # Abort multipart upload on error
                    s3.abort_multipart_upload(
                        Bucket=dest_bucket,
                        Key=dest_key,
                        UploadId=multipart_upload['UploadId']
                    )
                    raise e
                
            else:
                print(f"   ⚠️  No matching records found in {table_name}")
                results[table_name] = {'status': 'no_data', 'rows': 0}
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
                
        except Exception as e:
            print(f"   ❌ Error processing {table_name}: {str(e)}")
            results[table_name] = {'status': 'error', 'error': str(e)}
            # Clean up on error
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            continue
    
    return results


def save_patient_lists(t1d_df, t1d_hypo_df, s3, dest_bucket='dsw-sagemaker-dev-s3'):
    """
    Save the patient lists to S3 for reference
    """
    print("\n📁 Saving patient lists to S3...")
    
    # Save T1D filtered patient list
    t1d_csv = t1d_df.to_csv(index=False)
    s3.put_object(
        Bucket=dest_bucket,
        Key='T1D_Tosur/data/patient_lists/t1d_filtered_patients.csv',
        Body=t1d_csv.encode('utf-8')
    )
    print("   ✅ Saved T1D filtered patient list")
    
    # Save T1D Hypoglycemia patient list
    t1d_hypo_csv = t1d_hypo_df.to_csv(index=False)
    s3.put_object(
        Bucket=dest_bucket,
        Key='T1D_Tosur/data/patient_lists/t1d_hypoglycemia_patients.csv',
        Body=t1d_hypo_csv.encode('utf-8')
    )
    print("   ✅ Saved T1D Hypoglycemia patient list")


def generate_summary_report(t1d_ids, t1d_hypo_ids, t1d_results, t1d_hypo_results, s3, 
                           dest_bucket='dsw-sagemaker-dev-s3'):
    """
    Generate and save a summary report of the extraction
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = []
    report.append("=" * 70)
    report.append("T1D DATA EXTRACTION SUMMARY REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    
    report.append("\n📊 PATIENT COHORTS:")
    report.append(f"   • T1D Filtered Patients: {len(t1d_ids):,} unique IDs")
    report.append(f"   • T1D Hypoglycemia Patients: {len(t1d_hypo_ids):,} unique IDs")
    report.append(f"   • Overlap: {len(t1d_ids.intersection(t1d_hypo_ids)):,} patients")
    
    report.append("\n📁 T1D FILTERED COHORT EXTRACTION RESULTS:")
    for table, result in t1d_results.items():
        if result['status'] == 'success':
            report.append(f"   ✅ {table}: {result['rows']:,} rows extracted")
        elif result['status'] == 'no_data':
            report.append(f"   ⚠️  {table}: No matching records found")
        else:
            report.append(f"   ❌ {table}: Error - {result.get('error', 'Unknown error')}")
    
    report.append("\n📁 T1D HYPOGLYCEMIA COHORT EXTRACTION RESULTS:")
    for table, result in t1d_hypo_results.items():
        if result['status'] == 'success':
            report.append(f"   ✅ {table}: {result['rows']:,} rows extracted")
        elif result['status'] == 'no_data':
            report.append(f"   ⚠️  {table}: No matching records found")
        else:
            report.append(f"   ❌ {table}: Error - {result.get('error', 'Unknown error')}")
    
    report.append("\n📍 OUTPUT LOCATIONS:")
    report.append(f"   • T1D Filtered Data: s3://{dest_bucket}/T1D_Tosur/data/T1D_OMOP_variables/")
    report.append(f"   • T1D Hypoglycemia Data: s3://{dest_bucket}/T1D_Tosur/data/T1D_Hypoglycemia_OMOP_variables/")
    report.append(f"   • Patient Lists: s3://{dest_bucket}/T1D_Tosur/data/patient_lists/")
    
    report.append("\n" + "=" * 70)
    report.append("EXTRACTION COMPLETE")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    
    # Print to console
    print("\n" + report_text)
    
    # Save to S3
    s3.put_object(
        Bucket=dest_bucket,
        Key=f'T1D_Tosur/data/extraction_reports/summary_report_{timestamp}.txt',
        Body=report_text.encode('utf-8')
    )
    print(f"\n📄 Report saved to s3://{dest_bucket}/T1D_Tosur/data/extraction_reports/summary_report_{timestamp}.txt")


def main():
    """
    Main function to orchestrate T1D data extraction
    """
    print("\n" + "🏥 " * 20)
    print("T1D PATIENT DATA EXTRACTION PIPELINE")
    print("🏥 " * 20)
    
    start_time = datetime.now()
    
    try:
        # Step 1: Preprocess patient lists
        t1d_ids, t1d_hypo_ids, t1d_df, t1d_hypo_df = preprocess_t1d_patients()
        
        # Initialize S3 client
        print("\n🔗 Initializing S3 connection...")
        s3 = boto3.client('s3')
        
        # Step 2: Extract data for T1D filtered cohort
        t1d_results = extract_omop_data_for_cohort(
            person_ids=t1d_ids,
            cohort_name="T1D Filtered Cohort",
            dest_prefix='T1D_Tosur/data/T1D_OMOP_variables/',
            s3=s3
        )
        
        # Step 3: Extract data for T1D Hypoglycemia cohort
        t1d_hypo_results = extract_omop_data_for_cohort(
            person_ids=t1d_hypo_ids,
            cohort_name="T1D Hypoglycemia Cohort",
            dest_prefix='T1D_Tosur/data/T1D_Hypoglycemia_OMOP_variables/',
            s3=s3
        )
        
        # Step 4: Save patient lists to S3
        save_patient_lists(t1d_df, t1d_hypo_df, s3)
        
        # Step 5: Generate summary report
        generate_summary_report(t1d_ids, t1d_hypo_ids, t1d_results, t1d_hypo_results, s3)
        
        # Calculate execution time
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n⏱️  Total execution time: {duration}")
        print("\n✨ Pipeline completed successfully! ✨")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        raise e


if __name__ == "__main__":
    main()