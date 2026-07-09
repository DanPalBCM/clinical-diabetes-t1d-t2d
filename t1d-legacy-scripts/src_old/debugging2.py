import pandas as pd
import boto3
from datetime import datetime
import gc
import os

def extract_unique_measurements(s3_client, bucket, key, output_filename='unique_measurements.txt', chunk_size=50000):
    """
    Read measurement data in chunks and save all unique measurement names to a text file
    Memory-efficient version that processes data in chunks
    """
    print("="*60)
    print("EXTRACTING UNIQUE MEASUREMENT NAMES (Memory Efficient)")
    print("="*60)
    
    try:
        # Get the S3 object
        print(f"Reading data from s3://{bucket}/{key}")
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        
        # Initialize variables
        unique_measurements = set()
        total_rows = 0
        non_null_count = 0
        chunk_count = 0
        
        print(f"Processing in chunks of {chunk_size:,} rows...")
        
        # Read CSV in chunks
        for chunk in pd.read_csv(obj['Body'], chunksize=chunk_size):
            chunk_count += 1
            total_rows += len(chunk)
            
            # Extract non-null measurements from this chunk
            chunk_measurements = chunk['MEASUREMENT_SOURCE_VALUE'].dropna()
            non_null_count += len(chunk_measurements)
            
            # Add to unique set
            unique_measurements.update(chunk_measurements.unique())
            
            # Print progress
            print(f"  Chunk {chunk_count}: Processed {total_rows:,} rows total, "
                  f"found {len(unique_measurements):,} unique measurements so far")
            
            # Clean up chunk memory
            del chunk
            del chunk_measurements
            gc.collect()
        
        print(f"\nTotal rows processed: {total_rows:,}")
        print(f"Non-null measurement values: {non_null_count:,}")
        print(f"Found {len(unique_measurements):,} unique measurement names")
        
        # Convert to sorted list
        unique_measurements_sorted = sorted(list(unique_measurements))
        
        # Clear the set to free memory
        del unique_measurements
        gc.collect()
        
        # Save to text file
        print(f"\nSaving to {output_filename}...")
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            # Write header with metadata
            f.write(f"Unique Measurement Names\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source: s3://{bucket}/{key}\n")
            f.write(f"Total unique measurements: {len(unique_measurements_sorted):,}\n")
            f.write(f"Total rows processed: {total_rows:,}\n")
            f.write(f"Chunk size used: {chunk_size:,}\n")
            f.write("="*80 + "\n\n")
            
            # Write each unique measurement name
            for i, measurement in enumerate(unique_measurements_sorted, 1):
                f.write(f"{i:6d}. {measurement}\n")
        
        print(f"✓ Successfully saved {len(unique_measurements_sorted):,} unique measurement names to '{output_filename}'")
        
        # Show preview of first 10 measurements
        print("\nPreview of first 10 measurements:")
        for i, measurement in enumerate(unique_measurements_sorted[:10], 1):
            print(f"  {i:2d}. {measurement}")
        
        if len(unique_measurements_sorted) > 10:
            print(f"  ... and {len(unique_measurements_sorted) - 10:,} more")
        
        return unique_measurements_sorted
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
    finally:
        # Final garbage collection
        gc.collect()


def extract_unique_measurements_local(csv_file_path, output_filename='unique_measurements.txt', chunk_size=50000):
    """
    Read local CSV file in chunks and save all unique measurement names to a text file
    Memory-efficient version that processes data in chunks
    """
    print("="*60)
    print("EXTRACTING UNIQUE MEASUREMENT NAMES (Local File - Memory Efficient)")
    print("="*60)
    
    try:
        # Check file size
        file_size = os.path.getsize(csv_file_path) / (1024 * 1024)  # Size in MB
        print(f"File size: {file_size:.2f} MB")
        print(f"Processing in chunks of {chunk_size:,} rows...")
        
        # Initialize variables
        unique_measurements = set()
        total_rows = 0
        non_null_count = 0
        chunk_count = 0
        
        # Read CSV in chunks
        for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size):
            chunk_count += 1
            total_rows += len(chunk)
            
            # Extract non-null measurements from this chunk
            chunk_measurements = chunk['MEASUREMENT_SOURCE_VALUE'].dropna()
            non_null_count += len(chunk_measurements)
            
            # Add to unique set
            unique_measurements.update(chunk_measurements.unique())
            
            # Print progress
            print(f"  Chunk {chunk_count}: Processed {total_rows:,} rows total, "
                  f"found {len(unique_measurements):,} unique measurements so far")
            
            # Clean up chunk memory
            del chunk
            del chunk_measurements
            gc.collect()
        
        print(f"\nTotal rows processed: {total_rows:,}")
        print(f"Non-null measurement values: {non_null_count:,}")
        print(f"Found {len(unique_measurements):,} unique measurement names")
        
        # Convert to sorted list
        unique_measurements_sorted = sorted(list(unique_measurements))
        
        # Clear the set to free memory
        del unique_measurements
        gc.collect()
        
        # Save to text file
        print(f"\nSaving to {output_filename}...")
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            # Write header with metadata
            f.write(f"Unique Measurement Names\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source: {csv_file_path}\n")
            f.write(f"Total unique measurements: {len(unique_measurements_sorted):,}\n")
            f.write(f"Total rows processed: {total_rows:,}\n")
            f.write(f"Chunk size used: {chunk_size:,}\n")
            f.write("="*80 + "\n\n")
            
            # Write each unique measurement name
            for i, measurement in enumerate(unique_measurements_sorted, 1):
                f.write(f"{i:6d}. {measurement}\n")
        
        print(f"✓ Successfully saved {len(unique_measurements_sorted):,} unique measurement names to '{output_filename}'")
        
        # Show preview of first 10 measurements
        print("\nPreview of first 10 measurements:")
        for i, measurement in enumerate(unique_measurements_sorted[:10], 1):
            print(f"  {i:2d}. {measurement}")
        
        if len(unique_measurements_sorted) > 10:
            print(f"  ... and {len(unique_measurements_sorted) - 10:,} more")
        
        return unique_measurements_sorted
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
    finally:
        # Final garbage collection
        gc.collect()


def estimate_optimal_chunk_size():
    """
    Estimate optimal chunk size based on available memory
    """
    try:
        import psutil
        available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
        # Use about 25% of available memory for safety
        chunk_size = int((available_memory * 0.25 * 1024 * 1024 * 1024) / (100 * 8))  # Rough estimate
        chunk_size = max(10000, min(chunk_size, 100000))  # Keep between 10k and 100k
        return chunk_size
    except:
        return 50000  # Default if psutil not available


def run_melax_dataset():
    """
    Run the extraction specifically for the dsw-melax-dev-s3 dataset
    """
    print("Running extraction for dsw-melax-dev-s3/omop/measurement.csv")
    
    # Try to estimate optimal chunk size
    try:
        optimal_chunk_size = estimate_optimal_chunk_size()
        print(f"Estimated optimal chunk size: {optimal_chunk_size:,} rows")
    except:
        optimal_chunk_size = 50000
    
    # Ask for chunk size
    chunk_input = input(f"\nEnter chunk size (press Enter for default {optimal_chunk_size:,}): ").strip()
    if chunk_input:
        try:
            chunk_size = int(chunk_input)
        except:
            chunk_size = optimal_chunk_size
    else:
        chunk_size = optimal_chunk_size
    
    # Initialize S3 client
    s3 = boto3.client('s3')
    
    # Define S3 paths for the new dataset
    bucket = 'dsw-melax-dev-s3'
    key = 'omop/measurement.csv'
    
    # Extract unique measurements
    unique_measurements = extract_unique_measurements(s3, bucket, key, 
                                                    output_filename='melax_unique_measurements.txt',
                                                    chunk_size=chunk_size)
    
    return unique_measurements


if __name__ == "__main__":
    # Try to estimate optimal chunk size
    try:
        optimal_chunk_size = estimate_optimal_chunk_size()
        print(f"Estimated optimal chunk size: {optimal_chunk_size:,} rows")
    except:
        optimal_chunk_size = 50000
    
    print("\nChoose your data source:")
    print("1. S3 bucket (original T2D dataset)")
    print("2. Local CSV file")
    print("3. S3: dsw-melax-dev-s3/omop/measurement.csv (NEW)")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    # Ask for chunk size
    chunk_input = input(f"\nEnter chunk size (press Enter for default {optimal_chunk_size:,}): ").strip()
    if chunk_input:
        try:
            chunk_size = int(chunk_input)
        except:
            chunk_size = optimal_chunk_size
    else:
        chunk_size = optimal_chunk_size
    
    if choice == "1":
        # Initialize S3 client
        s3 = boto3.client('s3')
        
        # Define S3 paths (original)
        bucket = 'dsw-sagemaker-dev-s3'
        prefix = 'T2D_Tosur/data/T2D_OMOP_variables/'
        key = f'{prefix}measurement.csv'
        
        # Extract unique measurements
        unique_measurements = extract_unique_measurements(s3, bucket, key, chunk_size=chunk_size)
        
    elif choice == "2":
        # For local CSV file
        csv_path = input("Enter path to your CSV file: ").strip()
        unique_measurements = extract_unique_measurements_local(csv_path, chunk_size=chunk_size)
        
    elif choice == "3":
        # NEW: dsw-melax-dev-s3 dataset
        unique_measurements = run_melax_dataset()
        
    else:
        print("Invalid choice. Please run again and choose 1, 2, or 3.")
    
    print("\nDone! Check the output file in your current directory.")
    
    # Final cleanup
    gc.collect()