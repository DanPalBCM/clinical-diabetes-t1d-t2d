import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import umap
import os
from datetime import datetime
import re

# Create directories for results
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load the data
print("Loading data...")
data = pd.read_excel('../data/T2DPopulation_732025.xlsx')

# Create a timestamp for unique file naming
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Function to parse TXT columns with numeric values and comparison operators
def parse_txt_value(value):
    """
    Parse TXT values that may contain comparison operators.
    Returns: (numeric_value, operator, original_value)
    """
    if pd.isna(value):
        return np.nan, None, value
    
    # Convert to string and strip whitespace
    value_str = str(value).strip()
    
    # Pattern to match comparison operators and numbers
    pattern = r'^([<>]=?)?(\d+\.?\d*)$'
    match = re.match(pattern, value_str)
    
    if match:
        operator = match.group(1) or '='
        numeric_val = float(match.group(2))
        return numeric_val, operator, value_str
    else:
        # Try to parse as a simple number
        try:
            numeric_val = float(value_str)
            return numeric_val, '=', value_str
        except:
            return np.nan, None, value_str

# Function to convert TXT columns to numeric with metadata
def convert_txt_to_numeric(df, column_name):
    """
    Convert a TXT column to numeric values while preserving operator information.
    Returns a DataFrame with numeric values and metadata columns.
    """
    parsed_data = df[column_name].apply(parse_txt_value)
    
    result_df = pd.DataFrame({
        f'{column_name}_numeric': [x[0] for x in parsed_data],
        f'{column_name}_operator': [x[1] for x in parsed_data],
        f'{column_name}_original': [x[2] for x in parsed_data]
    })
    
    return result_df

# Identify TXT columns that likely contain numeric lab results
txt_columns = [col for col in data.columns if 'ResultTXT' in col]
print(f"\nFound {len(txt_columns)} TXT columns to process:")
for col in txt_columns:
    print(f"  - {col}")

# Process each TXT column
processed_data = data.copy()
txt_conversion_summary = []

for col in txt_columns:
    print(f"\nProcessing {col}...")
    
    # Convert TXT to numeric
    converted_df = convert_txt_to_numeric(data, col)
    
    # Add converted columns to the dataset
    for new_col in converted_df.columns:
        processed_data[new_col] = converted_df[new_col]
    
    # Generate summary statistics for the conversion
    total_values = data[col].notna().sum()
    numeric_values = converted_df[f'{col}_numeric'].notna().sum()
    operator_counts = converted_df[f'{col}_operator'].value_counts().to_dict()
    
    summary = {
        'Column': col,
        'Total_Non_Null': total_values,
        'Converted_to_Numeric': numeric_values,
        'Conversion_Rate': (numeric_values/total_values*100) if total_values > 0 else 0,
        'Operators': operator_counts
    }
    txt_conversion_summary.append(summary)
    
    # Print conversion statistics
    print(f"  Converted {numeric_values}/{total_values} values ({summary['Conversion_Rate']:.1f}%)")
    print(f"  Operator distribution: {operator_counts}")

# Save conversion summary
conversion_df = pd.DataFrame(txt_conversion_summary)
conversion_df.to_csv(f'results/txt_conversion_summary_{timestamp}.csv', index=False)

# Enhanced statistics for TXT columns
print("\n=== ENHANCED STATISTICS FOR TXT COLUMNS ===")
txt_stats = []

for col in txt_columns:
    numeric_col = f'{col}_numeric'
    operator_col = f'{col}_operator'
    
    if numeric_col in processed_data.columns:
        # Get numeric statistics
        numeric_data = processed_data[numeric_col]
        
        # Calculate statistics considering operator context
        stats = {
            'Column': col,
            'Count': numeric_data.notna().sum(),
            'Mean': numeric_data.mean(),
            'Std': numeric_data.std(),
            'Min': numeric_data.min(),
            '25%': numeric_data.quantile(0.25),
            'Median': numeric_data.median(),
            '75%': numeric_data.quantile(0.75),
            'Max': numeric_data.max(),
            'Missing': numeric_data.isna().sum()
        }
        
        # Add operator-specific statistics
        for op in ['<', '>', '<=', '>=', '=']:
            op_mask = processed_data[operator_col] == op
            op_count = op_mask.sum()
            if op_count > 0:
                stats[f'Count_{op}'] = op_count
                stats[f'Mean_{op}'] = numeric_data[op_mask].mean()
        
        txt_stats.append(stats)

txt_stats_df = pd.DataFrame(txt_stats)
txt_stats_df.to_csv(f'results/txt_columns_statistics_{timestamp}.csv', index=False)

# Visualize distribution of values with operators for each TXT column
for col in txt_columns[:5]:  # Limit to first 5 for visualization
    numeric_col = f'{col}_numeric'
    operator_col = f'{col}_operator'
    
    if numeric_col in processed_data.columns:
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram of numeric values colored by operator
        operators = processed_data[operator_col].unique()
        operators = [op for op in operators if op is not None]
        
        for op in operators:
            mask = processed_data[operator_col] == op
            values = processed_data.loc[mask, numeric_col]
            if len(values) > 0:
                ax1.hist(values, alpha=0.7, label=f'{op} ({len(values)} values)', bins=20)
        
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Distribution of {col}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot by operator
        data_for_box = []
        labels_for_box = []
        for op in operators:
            mask = processed_data[operator_col] == op
            values = processed_data.loc[mask, numeric_col].dropna()
            if len(values) > 0:
                data_for_box.append(values)
                labels_for_box.append(f'{op}\n(n={len(values)})')
        
        if data_for_box:
            ax2.boxplot(data_for_box, labels=labels_for_box)
            ax2.set_ylabel('Value')
            ax2.set_title(f'Box Plot by Operator')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'figures/txt_column_distribution_{col}_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

# Create a comprehensive report for TXT columns
with open(f'results/txt_columns_analysis_report_{timestamp}.txt', 'w') as f:
    f.write("=== TXT COLUMNS ANALYSIS REPORT ===\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("CONVERSION SUMMARY:\n")
    f.write("-" * 80 + "\n")
    for summary in txt_conversion_summary:
        f.write(f"\n{summary['Column']}:\n")
        f.write(f"  Total non-null values: {summary['Total_Non_Null']}\n")
        f.write(f"  Successfully converted: {summary['Converted_to_Numeric']} ({summary['Conversion_Rate']:.1f}%)\n")
        f.write(f"  Operator distribution:\n")
        for op, count in summary['Operators'].items():
            f.write(f"    {op}: {count} values\n")
    
    f.write("\n\nSTATISTICAL INTERPRETATION NOTES:\n")
    f.write("-" * 80 + "\n")
    f.write("1. Values with '<' operator represent upper bounds (actual value is less than shown)\n")
    f.write("2. Values with '>' operator represent lower bounds (actual value is greater than shown)\n")
    f.write("3. Statistics calculated on numeric values should be interpreted with caution:\n")
    f.write("   - Mean may be biased if many '<' or '>' values exist\n")
    f.write("   - Consider using median for more robust central tendency\n")
    f.write("   - Min/Max values may represent detection limits rather than actual measurements\n")
    
    f.write("\n\nRECOMMENDATIONS:\n")
    f.write("-" * 80 + "\n")
    f.write("1. For statistical modeling, consider:\n")
    f.write("   - Using censored data methods for values with '<' or '>'\n")
    f.write("   - Creating binary indicators for 'below/above detection limit'\n")
    f.write("   - Imputing values at detection limit or limit/2 for simple analyses\n")
    f.write("2. For visualization, always indicate which values are censored\n")
    f.write("3. Report both the numeric value and operator in final results\n")

# Now run the original analysis with the enhanced numeric columns
print("\n=== CONTINUING WITH ORIGINAL ANALYSIS ON ENHANCED DATA ===")

# Update numeric columns to include converted TXT columns
numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
for col in txt_columns:
    numeric_col = f'{col}_numeric'
    if numeric_col in processed_data.columns:
        numeric_cols.append(numeric_col)

# Continue with the rest of your original analysis...
# (Include all the original analysis code here, but using processed_data instead of data)

# 0. Print all column names (including new numeric columns)
print("\n=== COLUMN NAMES (INCLUDING CONVERTED) ===")
columns = processed_data.columns.tolist()
for i, col in enumerate(columns):
    print(f"{i+1}. {col}")

# Save enhanced column names
with open(f'results/column_names_enhanced_{timestamp}.txt', 'w') as f:
    f.write("=== COLUMN NAMES (INCLUDING CONVERTED) ===\n")
    for i, col in enumerate(columns):
        f.write(f"{i+1}. {col}\n")

print("\n=== ENHANCED ANALYSIS COMPLETE ===")
print(f"Results saved in 'results/' directory")
print(f"Figures saved in 'figures/' directory")
print(f"Timestamp: {timestamp}")