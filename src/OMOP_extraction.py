import pandas as pd
import boto3
import os
from io import StringIO
import numpy as np
import gc
import tempfile
import shutil

def main():
    # Step 1: Read the Excel file
    print("Reading T2D Population file...")
    t2d_pop = pd.read_excel("../data/T2DPopulation_7242025.xlsx")
    
    # Step 2: Print all column names
    print("\nColumn names in T2DPopulation_7242025.xlsx:")
    for col in t2d_pop.columns:
        print(f"  - {col}")
    
    # Step 3: Extract unique OMOP IDs
    print(f"\nExtracting unique OMOP IDs...")
    unique_omop_ids = t2d_pop['person_id'].unique()
    print(f"Found {len(unique_omop_ids)} unique OMOP IDs")
    
    # Convert to set for faster lookup
    omop_id_set = set(unique_omop_ids)
    
    # Initialize S3 client
    s3 = boto3.client('s3')
    
    # Define source and destination S3 paths
    source_bucket = 'dsw-melax-dev-s3'
    source_prefix = 'omop/'
    dest_bucket = 'dsw-sagemaker-dev-s3'
    dest_prefix = 'T2D_Tosur/data/T2D_OMOP_variables/'
    
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
        ('measurement.csv', 'PERSON_ID', None)
    ]
    
    # Process each table
    for table_name, id_column, dtype_dict in tables:
        print(f"\nProcessing {table_name}...")
        
        try:
            # Create a temporary directory for intermediate files
            temp_dir = tempfile.mkdtemp()
            temp_file_paths = []
            
            # Read from S3 in chunks to handle large files
            source_key = f"{source_prefix}{table_name}"
            print(f"  Reading from s3://{source_bucket}/{source_key}")
            
            # Get object to read in chunks
            obj = s3.get_object(Bucket=source_bucket, Key=source_key)
            
            # Process in smaller chunks to reduce memory usage
            chunk_size = 50000  # Reduced from 100000
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
                
                # Filter chunk to include only our OMOP IDs
                filtered_chunk = chunk[chunk[id_column].isin(omop_id_set)]
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
                
                print(f"  Processed {total_rows:,} rows, kept {filtered_rows:,} rows (saved {chunk_count} chunks)", end='\r')
            
            print()  # New line after progress
            
            if temp_file_paths:
                # Upload chunks directly to S3 using multipart upload
                dest_key = f"{dest_prefix}{table_name}"
                print(f"  Uploading to s3://{dest_bucket}/{dest_key}")
                
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
                        print(f"  Uploading chunk {i+1}/{len(temp_file_paths)}", end='\r')
                        
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
                    
                    print(f"  ✓ Successfully uploaded {table_name} with {filtered_rows:,} rows")
                    
                except Exception as e:
                    # Abort multipart upload on error
                    s3.abort_multipart_upload(
                        Bucket=dest_bucket,
                        Key=dest_key,
                        UploadId=multipart_upload['UploadId']
                    )
                    raise e
                
            else:
                print(f"  ⚠ No matching records found in {table_name}")
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
                
        except Exception as e:
            print(f"  ✗ Error processing {table_name}: {str(e)}")
            # Clean up on error
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            continue
    
    print("\n✓ Processing complete!")
    
    # Summary report
    print("\nSummary:")
    print(f"- Source T2D population: {len(t2d_pop):,} records")
    print(f"- Unique OMOP IDs: {len(unique_omop_ids):,}")
    print(f"- Output location: s3://{dest_bucket}/{dest_prefix}")


def process_table_in_batches(s3, source_bucket, source_key, dest_bucket, dest_key, 
                           id_column, omop_id_set, batch_size=1000000):
    """
    Alternative approach: Process table in batches of person IDs to reduce memory usage
    """
    print(f"  Processing {source_key} in batches...")
    
    # Convert set to sorted list for batching
    omop_id_list = sorted(list(omop_id_set))
    total_ids = len(omop_id_list)
    
    # Process in batches
    all_parts = []
    part_number = 1
    
    # Initialize multipart upload
    multipart_upload = s3.create_multipart_upload(
        Bucket=dest_bucket,
        Key=dest_key
    )
    
    try:
        for i in range(0, total_ids, batch_size):
            batch_ids = set(omop_id_list[i:i+batch_size])
            print(f"  Processing IDs {i+1} to {min(i+batch_size, total_ids)} of {total_ids}")
            
            # Read and filter data for this batch of IDs
            obj = s3.get_object(Bucket=source_bucket, Key=source_key)
            
            filtered_chunks = []
            for chunk in pd.read_csv(obj['Body'], chunksize=50000):
                filtered_chunk = chunk[chunk[id_column].isin(batch_ids)]
                if len(filtered_chunk) > 0:
                    filtered_chunks.append(filtered_chunk)
            
            if filtered_chunks:
                # Combine chunks for this batch
                batch_df = pd.concat(filtered_chunks, ignore_index=True)
                
                # Convert to CSV and upload as part
                csv_bytes = batch_df.to_csv(index=False, header=(i==0)).encode('utf-8')
                
                response = s3.upload_part(
                    Bucket=dest_bucket,
                    Key=dest_key,
                    PartNumber=part_number,
                    UploadId=multipart_upload['UploadId'],
                    Body=csv_bytes
                )
                
                all_parts.append({
                    'ETag': response['ETag'],
                    'PartNumber': part_number
                })
                
                part_number += 1
                
                # Clear memory
                del batch_df
                del filtered_chunks
                gc.collect()
        
        # Complete multipart upload
        s3.complete_multipart_upload(
            Bucket=dest_bucket,
            Key=dest_key,
            UploadId=multipart_upload['UploadId'],
            MultipartUpload={'Parts': all_parts}
        )
        
        print(f"  ✓ Successfully uploaded using batch processing")
        
    except Exception as e:
        # Abort multipart upload on error
        s3.abort_multipart_upload(
            Bucket=dest_bucket,
            Key=dest_key,
            UploadId=multipart_upload['UploadId']
        )
        raise e


if __name__ == "__main__":
    main()