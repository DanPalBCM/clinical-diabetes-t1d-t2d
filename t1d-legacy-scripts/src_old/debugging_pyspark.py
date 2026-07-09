import boto3
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, collect_set
from pyspark import SparkContext, SparkConf
import os
import sys

def create_spark_session_sagemaker(app_name="ExtractUniqueMeasurements"):
    """
    Create and configure a Spark session optimized for SageMaker environment
    """
    # SageMaker-specific Spark configuration
    conf = SparkConf() \
        .setAppName(app_name) \
        .set("spark.sql.adaptive.enabled", "true") \
        .set("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .set("spark.sql.adaptive.skewJoin.enabled", "true") \
        .set("spark.sql.shuffle.partitions", "200") \
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .set("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .set("spark.dynamicAllocation.enabled", "true") \
        .set("spark.dynamicAllocation.minExecutors", "1") \
        .set("spark.dynamicAllocation.maxExecutors", "10")
    
    # Create SparkContext first (required in SageMaker)
    sc = SparkContext(conf=conf)
    
    # Create SparkSession from SparkContext
    spark = SparkSession(sc)
    
    # Configure S3 access for SageMaker
    # SageMaker uses IAM roles, so we use the IAM credentials provider
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    hadoop_conf.set("fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    hadoop_conf.set("fs.s3a.aws.credentials.provider", 
                    "com.amazonaws.auth.InstanceProfileCredentialsProvider,com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
    hadoop_conf.set("fs.s3a.endpoint", "s3.amazonaws.com")
    hadoop_conf.set("fs.s3a.connection.ssl.enabled", "true")
    
    # Optimize S3 access
    hadoop_conf.set("fs.s3a.connection.maximum", "100")
    hadoop_conf.set("fs.s3a.fast.upload", "true")
    hadoop_conf.set("fs.s3a.fast.upload.buffer", "bytebuffer")
    hadoop_conf.set("fs.s3a.multipart.size", "104857600")  # 100MB
    
    return spark

def extract_unique_measurements_spark(spark, data_path, output_path=None, save_to_s3=True):
    """
    Extract unique measurement names using PySpark in SageMaker
    
    Args:
        spark: SparkSession
        data_path: S3 path to input CSV
        output_path: S3 path for output (if save_to_s3=True) or local path
        save_to_s3: Whether to save output to S3 or locally
    """
    print("="*60)
    print("EXTRACTING UNIQUE MEASUREMENT NAMES (PySpark on SageMaker)")
    print("="*60)
    
    try:
        print(f"Reading data from {data_path}")
        
        # Read CSV file with optimized settings
        df = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .option("multiLine", "false") \
            .csv(data_path)
        
        # Repartition for better parallelism based on data size
        # Get approximate size first
        initial_partitions = df.rdd.getNumPartitions()
        print(f"Initial partitions: {initial_partitions}")
        
        # Repartition if needed (for large files)
        if initial_partitions < 50:
            df = df.repartition(50)
            print(f"Repartitioned to 50 partitions for better parallelism")
        
        # Cache the dataframe
        df.cache()
        
        # Get total row count
        total_rows = df.count()
        print(f"Total rows in dataset: {total_rows:,}")
        
        # Filter out null values and get count
        non_null_df = df.filter(col("MEASUREMENT_SOURCE_VALUE").isNotNull())
        non_null_count = non_null_df.count()
        print(f"Non-null measurement values: {non_null_count:,}")
        
        # Get unique measurements using distinct()
        print("\nExtracting unique measurements...")
        unique_measurements_df = non_null_df.select("MEASUREMENT_SOURCE_VALUE").distinct()
        
        # Coalesce to reduce partitions before collect for efficiency
        unique_measurements_df = unique_measurements_df.coalesce(1)
        
        # Collect results
        unique_measurements_rows = unique_measurements_df.collect()
        unique_measurements = sorted([row[0] for row in unique_measurements_rows])
        unique_count = len(unique_measurements)
        
        print(f"Found {unique_count:,} unique measurement names")
        
        # Prepare output content
        output_lines = []
        output_lines.append("Unique Measurement Names")
        output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output_lines.append(f"Source: {data_path}")
        output_lines.append(f"Total unique measurements: {unique_count:,}")
        output_lines.append(f"Total rows processed: {total_rows:,}")
        output_lines.append(f"Processing engine: PySpark on SageMaker")
        output_lines.append("="*80)
        output_lines.append("")
        
        # Add each unique measurement
        for i, measurement in enumerate(unique_measurements, 1):
            output_lines.append(f"{i:6d}. {measurement}")
        
        output_content = '\n'.join(output_lines)
        
        # Save output
        if save_to_s3 and output_path:
            print(f"\nSaving to S3: {output_path}")
            
            # Create a DataFrame with the output content
            output_df = spark.createDataFrame([(output_content,)], ["content"])
            
            # Write to S3 as text file
            output_df.coalesce(1).write.mode("overwrite").text(output_path)
            
            print(f"✓ Successfully saved to S3: {output_path}")
        else:
            # Save locally
            local_filename = output_path or 'unique_measurements.txt'
            print(f"\nSaving locally to {local_filename}...")
            
            with open(local_filename, 'w', encoding='utf-8') as f:
                f.write(output_content)
            
            print(f"✓ Successfully saved {unique_count:,} unique measurement names to '{local_filename}'")
        
        # Show preview
        print("\nPreview of first 10 measurements:")
        for i, measurement in enumerate(unique_measurements[:10], 1):
            print(f"  {i:2d}. {measurement}")
        
        if unique_count > 10:
            print(f"  ... and {unique_count - 10:,} more")
        
        # Unpersist the cached dataframe
        df.unpersist()
        
        return unique_measurements
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def extract_unique_measurements_spark_with_stats(spark, data_path, output_path=None, save_to_s3=True):
    """
    Extract unique measurement names with frequency statistics using PySpark in SageMaker
    """
    print("="*60)
    print("EXTRACTING UNIQUE MEASUREMENT NAMES WITH STATISTICS (PySpark on SageMaker)")
    print("="*60)
    
    try:
        print(f"Reading data from {data_path}")
        
        # Read CSV file
        df = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv(data_path)
        
        # Repartition for better parallelism
        df = df.repartition(100)
        
        # Get total row count
        total_rows = df.count()
        print(f"Total rows in dataset: {total_rows:,}")
        
        # Filter out null values
        non_null_df = df.filter(col("MEASUREMENT_SOURCE_VALUE").isNotNull())
        non_null_count = non_null_df.count()
        print(f"Non-null measurement values: {non_null_count:,}")
        
        # Get unique measurements with counts
        print("\nExtracting unique measurements with frequencies...")
        measurement_stats_df = non_null_df.groupBy("MEASUREMENT_SOURCE_VALUE") \
            .agg(count("*").alias("frequency")) \
            .orderBy("MEASUREMENT_SOURCE_VALUE")
        
        # Cache the stats dataframe as we'll use it multiple times
        measurement_stats_df.cache()
        
        # Collect results
        measurement_stats = measurement_stats_df.collect()
        unique_count = len(measurement_stats)
        print(f"Found {unique_count:,} unique measurement names")
        
        # Calculate statistics
        frequencies = [row['frequency'] for row in measurement_stats]
        max_freq = max(frequencies) if frequencies else 0
        min_freq = min(frequencies) if frequencies else 0
        avg_freq = sum(frequencies) / len(frequencies) if frequencies else 0
        
        # Get top measurements
        top_measurements = measurement_stats_df \
            .orderBy(col("frequency").desc()) \
            .limit(10) \
            .collect()
        
        # Prepare output content
        output_lines = []
        output_lines.append("Unique Measurement Names with Statistics")
        output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output_lines.append(f"Source: {data_path}")
        output_lines.append(f"Total unique measurements: {unique_count:,}")
        output_lines.append(f"Total rows processed: {total_rows:,}")
        output_lines.append(f"Processing engine: PySpark on SageMaker")
        output_lines.append(f"\nFrequency Statistics:")
        output_lines.append(f"  Maximum frequency: {max_freq:,}")
        output_lines.append(f"  Minimum frequency: {min_freq:,}")
        output_lines.append(f"  Average frequency: {avg_freq:,.2f}")
        output_lines.append("="*80)
        output_lines.append("")
        output_lines.append(f"{'#':>6} | {'Measurement Name':<50} | {'Frequency':>10}")
        output_lines.append("-"*80)
        
        # Add measurements with frequencies
        for i, row in enumerate(measurement_stats, 1):
            measurement = row['MEASUREMENT_SOURCE_VALUE']
            frequency = row['frequency']
            output_lines.append(f"{i:6d} | {measurement:<50} | {frequency:10,}")
        
        output_content = '\n'.join(output_lines)
        
        # Save output
        if save_to_s3 and output_path:
            print(f"\nSaving to S3: {output_path}")
            
            # Create DataFrame and save
            output_df = spark.createDataFrame([(output_content,)], ["content"])
            output_df.coalesce(1).write.mode("overwrite").text(output_path)
            
            print(f"✓ Successfully saved to S3: {output_path}")
        else:
            # Save locally
            local_filename = output_path or 'unique_measurements_with_stats.txt'
            print(f"\nSaving locally to {local_filename}...")
            
            with open(local_filename, 'w', encoding='utf-8') as f:
                f.write(output_content)
            
            print(f"✓ Successfully saved to '{local_filename}'")
        
        # Show preview of top 10
        print("\nTop 10 most frequent measurements:")
        for i, row in enumerate(top_measurements, 1):
            print(f"  {i:2d}. {row['MEASUREMENT_SOURCE_VALUE']} (frequency: {row['frequency']:,})")
        
        # Unpersist
        measurement_stats_df.unpersist()
        
        return measurement_stats
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Main function for SageMaker PySpark job
    """
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        # Running as SageMaker Processing Job
        input_path = sys.argv[1] if len(sys.argv) > 1 else None
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        include_stats = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False
        
        if not input_path:
            print("Error: Input path required")
            sys.exit(1)
            
        # Create Spark session
        spark = create_spark_session_sagemaker()
        
        try:
            if include_stats:
                extract_unique_measurements_spark_with_stats(spark, input_path, output_path, save_to_s3=True)
            else:
                extract_unique_measurements_spark(spark, input_path, output_path, save_to_s3=True)
        finally:
            spark.stop()
    
    else:
        # Interactive mode
        spark = create_spark_session_sagemaker()
        
        try:
            print("\nSageMaker PySpark - Extract Unique Measurements")
            print("\nChoose your data source:")
            print("1. S3: dsw-sagemaker-dev-s3 (original T2D dataset)")
            print("2. S3: dsw-melax-dev-s3/omop/measurement.csv")
            print("3. Custom S3 path")
            
            choice = input("Enter choice (1-3): ").strip()
            
            if choice == "1":
                data_path = "s3://dsw-sagemaker-dev-s3/T2D_Tosur/data/T2D_OMOP_variables/measurement.csv"
                output_base = "s3://dsw-sagemaker-dev-s3/T2D_Tosur/outputs/t2d_unique_measurements"
            elif choice == "2":
                data_path = "s3://dsw-melax-dev-s3/omop/measurement.csv"
                output_base = "s3://dsw-melax-dev-s3/outputs/melax_unique_measurements"
            elif choice == "3":
                bucket = input("Enter S3 bucket name: ").strip()
                key = input("Enter S3 key/path: ").strip()
                data_path = f"s3://{bucket}/{key}"
                output_base = f"s3://{bucket}/outputs/custom_unique_measurements"
            else:
                print("Invalid choice.")
                return
            
            # Ask for output preference
            output_choice = input("\nSave output to S3? (y/n): ").strip().lower()
            save_to_s3 = output_choice == 'y'
            
            if save_to_s3:
                output_path = output_base
            else:
                output_path = None  # Will save locally
            
            # Ask if user wants statistics
            stats_choice = input("\nInclude frequency statistics? (y/n): ").strip().lower()
            
            if stats_choice == 'y':
                if save_to_s3:
                    output_path = f"{output_base}_with_stats"
                extract_unique_measurements_spark_with_stats(spark, data_path, output_path, save_to_s3)
            else:
                extract_unique_measurements_spark(spark, data_path, output_path, save_to_s3)
            
        finally:
            spark.stop()
            print("\nSpark session closed.")

if __name__ == "__main__":
    main()
    print("\nDone!")