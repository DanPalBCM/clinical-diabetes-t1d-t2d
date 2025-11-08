import pandas as pd
import boto3
from datetime import datetime
import json
from typing import Dict, Tuple, Optional
import gc

class OMOPDateRangeAnalyzer:
    """
    Tool for analyzing date ranges in OMOP data to understand temporal coverage
    """
    
    def __init__(self, source_bucket='dsw-melax-dev-s3', source_prefix='omop/'):
        """
        Initialize the OMOP Date Range Analyzer
        
        Args:
            source_bucket: S3 bucket containing source OMOP data
            source_prefix: Prefix for OMOP data in source bucket
        """
        self.source_bucket = source_bucket
        self.source_prefix = source_prefix
        self.s3 = boto3.client('s3')
        self.date_ranges = {}
        
    def analyze_table_dates(self, table_name: str, date_columns: list, 
                           sample_size: int = 100000) -> Dict[str, Tuple[str, str]]:
        """
        Analyze date ranges in a specific OMOP table using efficient sampling
        
        Args:
            table_name: Name of the OMOP table (e.g., 'condition_occurrence.csv')
            date_columns: List of date column names to analyze
            sample_size: Number of rows to sample per chunk for efficiency
            
        Returns:
            Dictionary with date column names as keys and (min_date, max_date) tuples as values
        """
        source_key = f"{self.source_prefix}{table_name}"
        results = {}
        
        print(f"\nAnalyzing {table_name}...")
        print(f"  Date columns to check: {', '.join(date_columns)}")
        
        try:
            # Initialize min/max trackers for each date column
            date_trackers = {col: {'min': None, 'max': None} for col in date_columns}
            
            # Get object from S3
            obj = self.s3.get_object(Bucket=self.source_bucket, Key=source_key)
            
            # Process in chunks for memory efficiency
            chunk_size = 50000
            total_rows = 0
            rows_with_dates = 0
            
            for chunk_num, chunk in enumerate(pd.read_csv(obj['Body'], 
                                                         chunksize=chunk_size, 
                                                         low_memory=False)):
                total_rows += len(chunk)
                
                # Sample the chunk if it's large (for extra efficiency)
                if len(chunk) > sample_size and chunk_num % 5 != 0:  # Sample every 5th chunk fully
                    chunk = chunk.sample(n=min(sample_size, len(chunk)))
                
                # Check each date column
                for col in date_columns:
                    if col in chunk.columns:
                        # Convert to datetime, handling different formats
                        dates = pd.to_datetime(chunk[col], errors='coerce')
                        valid_dates = dates.dropna()
                        
                        if len(valid_dates) > 0:
                            rows_with_dates += len(valid_dates)
                            
                            # Update min/max
                            chunk_min = valid_dates.min()
                            chunk_max = valid_dates.max()
                            
                            if date_trackers[col]['min'] is None or chunk_min < date_trackers[col]['min']:
                                date_trackers[col]['min'] = chunk_min
                            
                            if date_trackers[col]['max'] is None or chunk_max > date_trackers[col]['max']:
                                date_trackers[col]['max'] = chunk_max
                
                # Progress indicator
                if chunk_num % 10 == 0:
                    print(f"    Processed {total_rows:,} rows...", end='\r')
                
                # Memory cleanup
                del chunk
                if chunk_num % 20 == 0:
                    gc.collect()
            
            print(f"    Processed {total_rows:,} total rows")
            
            # Format results
            for col, tracker in date_trackers.items():
                if tracker['min'] is not None and tracker['max'] is not None:
                    results[col] = (
                        tracker['min'].strftime('%Y-%m-%d'),
                        tracker['max'].strftime('%Y-%m-%d')
                    )
                    print(f"    {col}: {results[col][0]} to {results[col][1]}")
                    
                    # Calculate span
                    days_span = (tracker['max'] - tracker['min']).days
                    years_span = days_span / 365.25
                    print(f"      Span: {days_span:,} days ({years_span:.1f} years)")
            
            return results
            
        except Exception as e:
            print(f"  ✗ Error analyzing {table_name}: {str(e)}")
            return {}
    
    def get_overall_date_range(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Calculate the overall min and max dates across all analyzed tables
        
        Returns:
            Tuple of (overall_min_date, overall_max_date) as strings
        """
        all_min_dates = []
        all_max_dates = []
        
        for table_ranges in self.date_ranges.values():
            for col, (min_date, max_date) in table_ranges.items():
                all_min_dates.append(pd.to_datetime(min_date))
                all_max_dates.append(pd.to_datetime(max_date))
        
        if all_min_dates and all_max_dates:
            overall_min = min(all_min_dates).strftime('%Y-%m-%d')
            overall_max = max(all_max_dates).strftime('%Y-%m-%d')
            return overall_min, overall_max
        
        return None, None
    
    def analyze_all_tables(self, quick_mode: bool = True):
        """
        Analyze date ranges across all relevant OMOP tables
        
        Args:
            quick_mode: If True, only analyze the most important tables for speed
        """
        print(f"\n{'='*60}")
        print(f"OMOP DATE RANGE ANALYSIS")
        print(f"{'='*60}")
        print(f"Source: s3://{self.source_bucket}/{self.source_prefix}")
        print(f"Mode: {'Quick (essential tables only)' if quick_mode else 'Comprehensive'}")
        
        # Define tables and their date columns
        if quick_mode:
            # Essential tables for quick analysis
            tables_to_analyze = {
                'condition_occurrence.csv': ['CONDITION_START_DATE', 'CONDITION_START_DATETIME', 
                                            'CONDITION_END_DATE', 'CONDITION_END_DATETIME'],
                'drug_exposure.csv': ['DRUG_EXPOSURE_START_DATE', 'DRUG_EXPOSURE_START_DATETIME',
                                     'DRUG_EXPOSURE_END_DATE', 'DRUG_EXPOSURE_END_DATETIME'],
                'measurement.csv': ['MEASUREMENT_DATE', 'MEASUREMENT_DATETIME'],
                'observation.csv': ['OBSERVATION_DATE', 'OBSERVATION_DATETIME'],
            }
        else:
            # Comprehensive analysis including all tables with dates
            tables_to_analyze = {
                'person.csv': ['BIRTH_DATETIME', 'DEATH_DATETIME'],
                'condition_occurrence.csv': ['CONDITION_START_DATE', 'CONDITION_START_DATETIME', 
                                            'CONDITION_END_DATE', 'CONDITION_END_DATETIME'],
                'drug_exposure.csv': ['DRUG_EXPOSURE_START_DATE', 'DRUG_EXPOSURE_START_DATETIME',
                                     'DRUG_EXPOSURE_END_DATE', 'DRUG_EXPOSURE_END_DATETIME'],
                'measurement.csv': ['MEASUREMENT_DATE', 'MEASUREMENT_DATETIME'],
                'observation.csv': ['OBSERVATION_DATE', 'OBSERVATION_DATETIME'],
                'procedure_occurrence.csv': ['PROCEDURE_DATE', 'PROCEDURE_DATETIME'],
                'visit_occurrence.csv': ['VISIT_START_DATE', 'VISIT_START_DATETIME',
                                        'VISIT_END_DATE', 'VISIT_END_DATETIME'],
                'device_exposure.csv': ['DEVICE_EXPOSURE_START_DATE', 'DEVICE_EXPOSURE_START_DATETIME',
                                       'DEVICE_EXPOSURE_END_DATE', 'DEVICE_EXPOSURE_END_DATETIME'],
            }
        
        # Analyze each table
        for table_name, date_columns in tables_to_analyze.items():
            ranges = self.analyze_table_dates(table_name, date_columns)
            if ranges:
                self.date_ranges[table_name] = ranges
        
        # Calculate overall range
        overall_min, overall_max = self.get_overall_date_range()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*60}")
        
        if overall_min and overall_max:
            print(f"\n📊 OVERALL DATE RANGE:")
            print(f"  Earliest date: {overall_min}")
            print(f"  Latest date:   {overall_max}")
            
            # Calculate span
            min_dt = pd.to_datetime(overall_min)
            max_dt = pd.to_datetime(overall_max)
            days_span = (max_dt - min_dt).days
            years_span = days_span / 365.25
            
            print(f"  Total span:    {days_span:,} days ({years_span:.1f} years)")
            
            # Show which tables contributed to min/max
            print(f"\n📋 Tables with earliest dates:")
            for table, ranges in self.date_ranges.items():
                for col, (min_date, max_date) in ranges.items():
                    if min_date == overall_min:
                        print(f"  - {table} ({col})")
            
            print(f"\n📋 Tables with latest dates:")
            for table, ranges in self.date_ranges.items():
                for col, (min_date, max_date) in ranges.items():
                    if max_date == overall_max:
                        print(f"  - {table} ({col})")
        else:
            print("No date ranges found in the analyzed tables.")
        
        print(f"{'='*60}\n")
        
        return self.date_ranges
    
    def save_results(self, output_bucket: str = None, output_key: str = None):
        """
        Save the analysis results to S3 as JSON
        
        Args:
            output_bucket: S3 bucket for output (defaults to source bucket)
            output_key: S3 key for output file
        """
        if not output_bucket:
            output_bucket = self.source_bucket
        
        if not output_key:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_key = f"omop_date_range_analysis_{timestamp}.json"
        
        overall_min, overall_max = self.get_overall_date_range()
        
        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'source_bucket': self.source_bucket,
            'source_prefix': self.source_prefix,
            'overall_date_range': {
                'min_date': overall_min,
                'max_date': overall_max,
                'span_days': (pd.to_datetime(overall_max) - pd.to_datetime(overall_min)).days if overall_min and overall_max else None
            },
            'table_ranges': self.date_ranges
        }
        
        # Save to S3
        self.s3.put_object(
            Bucket=output_bucket,
            Key=output_key,
            Body=json.dumps(results, indent=2)
        )
        
        print(f"\n✓ Results saved to s3://{output_bucket}/{output_key}")
        
        return results


def main():
    """
    Main function to run the OMOP date range analysis
    """
    # Configuration
    SOURCE_BUCKET = 'dsw-melax-dev-s3'
    SOURCE_PREFIX = 'omop/'
    
    # Create analyzer instance
    analyzer = OMOPDateRangeAnalyzer(
        source_bucket=SOURCE_BUCKET,
        source_prefix=SOURCE_PREFIX
    )
    
    # Run analysis in quick mode (set to False for comprehensive analysis)
    # Quick mode analyzes only the most important tables for speed
    results = analyzer.analyze_all_tables(quick_mode=True)
    
    # Optionally save results to S3
    # analyzer.save_results()
    
    # For even faster analysis of just one table (as you suggested):
    # You can uncomment this to analyze only condition_occurrence
    """
    print("\n" + "="*60)
    print("QUICK ANALYSIS: Condition Occurrence Only")
    print("="*60)
    
    condition_ranges = analyzer.analyze_table_dates(
        'condition_occurrence.csv',
        ['CONDITION_START_DATE', 'CONDITION_START_DATETIME', 
         'CONDITION_END_DATE', 'CONDITION_END_DATETIME'],
        sample_size=100000  # Adjust sample size for speed vs accuracy
    )
    
    if condition_ranges:
        for col, (min_date, max_date) in condition_ranges.items():
            print(f"\n{col}:")
            print(f"  Min: {min_date}")
            print(f"  Max: {max_date}")
    """
    
    return results


if __name__ == "__main__":
    main()