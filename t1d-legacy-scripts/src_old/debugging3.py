import boto3
import pandas as pd
import gc
from io import BytesIO
import sys

def process_large_s3_csv(bucket_name, key, chunk_size=10**6):
    """
    Process a large CSV file from S3 in chunks to extract unique values
    from MEASUREMENT_SOURCE_VALUE column.
    
    Args:
        bucket_name: S3 bucket name
        key: S3 object key (file path)
        chunk_size: Number of rows to process at a time (default: 1 million)
    """
    
    # Initialize S3 client
    s3 = boto3.client('s3')
    
    # Initialize variables
    unique_values = set()
    total_rows = 0
    chunk_count = 0
    
    try:
        # Get object size to track progress
        response = s3.head_object(Bucket=bucket_name, Key=key)
        file_size = response['ContentLength']
        print(f"File size: {file_size:,} bytes ({file_size/1e9:.2f} GB)")
        
        # Stream the file from S3
        print("Starting to process file in chunks...")
        response = s3.get_object(Bucket=bucket_name, Key=key)
        
        # Read CSV in chunks
        for chunk in pd.read_csv(response['Body'], 
                                 chunksize=chunk_size,
                                 usecols=['MEASUREMENT_SOURCE_VALUE'],  # Only load needed column
                                 dtype={'MEASUREMENT_SOURCE_VALUE': 'object'},  # Specify dtype to avoid inference
                                 low_memory=False):
            
            chunk_count += 1
            chunk_rows = len(chunk)
            total_rows += chunk_rows
            
            # Extract unique values from this chunk
            chunk_unique = chunk['MEASUREMENT_SOURCE_VALUE'].dropna().unique()
            unique_values.update(chunk_unique)
            
            # Print progress
            print(f"Processed chunk {chunk_count}: {chunk_rows:,} rows. "
                  f"Total rows so far: {total_rows:,}. "
                  f"Unique values found: {len(unique_values):,}")
            
            # Clean up chunk from memory
            del chunk
            del chunk_unique
            
            # Force garbage collection every 10 chunks
            if chunk_count % 10 == 0:
                gc.collect()
        
        print(f"\nProcessing complete!")
        print(f"Total rows processed: {total_rows:,}")
        print(f"Total unique MEASUREMENT_SOURCE_VALUE values: {len(unique_values):,}")
        
        # Convert set to sorted list for saving
        unique_values_list = sorted(list(unique_values))
        
        # Save unique values to file
        output_filename = 'unique_measurement_source_values.txt'
        with open(output_filename, 'w', encoding='utf-8') as f:
            for value in unique_values_list:
                f.write(f"{value}\n")
        
        print(f"\nUnique values saved to: {output_filename}")
        
        # Clean up
        del unique_values
        del unique_values_list
        gc.collect()
        
        return total_rows
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise
    finally:
        # Final cleanup
        gc.collect()


def main():
    # S3 configuration
    bucket_name = 'dsw-melax-dev-s3'
    key = 'omop/measurement.csv'
    
    # Process the file with different chunk sizes based on available memory
    # Adjust chunk_size based on your system's memory
    # Smaller chunks = less memory usage but slower processing
    chunk_size = 500000  # 500K rows per chunk (conservative for 411GB file)
    
    try:
        total_rows = process_large_s3_csv(bucket_name, key, chunk_size)
        print(f"\nFinal total row count: {total_rows:,}")
    except Exception as e:
        print(f"Failed to process file: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()