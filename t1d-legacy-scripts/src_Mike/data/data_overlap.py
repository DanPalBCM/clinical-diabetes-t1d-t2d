import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import warnings
warnings.filterwarnings('ignore')

def load_datasets():
    """Load all three datasets"""
    print("Loading datasets...")
    
    # Load CPT T1D study (dataset 1) - has MRN
    try:
        df1 = pd.read_excel("/home/sagemaker-user/T2D/src_Mike/data/MRNs for Hypoglycemia (1).xlsx", sheet_name=1)
        print(f"Dataset 1 loaded: {df1.shape[0]} rows, {df1.shape[1]} columns")
        print(f"Dataset 1 columns: {df1.columns.tolist()}")
    except Exception as e:
        print(f"Error loading dataset 1: {e}")
        return None, None, None
    
    # Load TCH data (dataset 2) - has PATIENTID and PEDSNET_ID
    try:
        df2 = pd.read_csv("/home/sagemaker-user/T2D/data/CROSSWALK_PATINENTIDS.csv")
        print(f"Dataset 2 loaded: {df2.shape[0]} rows, {df2.shape[1]} columns")
        print(f"Dataset 2 columns: {df2.columns.tolist()}")
    except Exception as e:
        print(f"Error loading dataset 2: {e}")
        return None, None, None
    
    # Load crosswalk file for MRN to PATIENTID/PEDSNET_ID mapping
    try:
        crosswalk = pd.read_csv("/home/sagemaker-user/T2D/data_crosswalk_all/crosswalk_mrn.csv")
        print(f"Crosswalk loaded: {crosswalk.shape[0]} rows, {crosswalk.shape[1]} columns")
        print(f"Crosswalk columns: {crosswalk.columns.tolist()}")
    except Exception as e:
        print(f"Error loading crosswalk: {e}")
        return None, None, None
    
    return df1, df2, crosswalk

def find_mrn_column(df):
    """Find the MRN column in dataset 1"""
    possible_mrn_cols = ['MRN', 'mrn', 'PAT_MRN_ID', 'Patient_MRN', 'PatientMRN']
    
    for col in df.columns:
        if 'MRN' in col.upper():
            return col
    
    # If not found, return the first column as a fallback
    return df.columns[0]

def add_mrn_to_dataset2(df2, crosswalk):
    """Add MRN column to dataset 2 using crosswalk"""
    print("\nAdding MRN column to dataset 2...")
    
    # Ensure correct data types for merging
    df2['PATIENTID'] = df2['PATIENTID'].astype(str)
    crosswalk['TCH_SOURCE_ID'] = crosswalk['TCH_SOURCE_ID'].astype(str)
    crosswalk['PAT_MRN_ID'] = crosswalk['PAT_MRN_ID'].astype(str)
    
    # Merge dataset 2 with crosswalk to add MRN
    df2_with_mrn = df2.merge(
        crosswalk[['TCH_SOURCE_ID', 'PAT_MRN_ID']], 
        left_on='PATIENTID', 
        right_on='TCH_SOURCE_ID', 
        how='left'
    )
    
    # Rename PAT_MRN_ID to MRN for clarity
    df2_with_mrn['MRN'] = df2_with_mrn['PAT_MRN_ID']
    
    # Count how many MRNs were successfully added
    mrn_added = df2_with_mrn['MRN'].notna().sum()
    print(f"Successfully added MRN for {mrn_added} out of {len(df2_with_mrn)} records in dataset 2")
    
    return df2_with_mrn

def save_overlap_data(df1, df2_with_mrn, overlap_mrns, mrn_col):
    """Save overlapping patients data and generate summary statistics"""
    print("\n" + "="*60)
    print("SAVING OVERLAPPING PATIENTS DATA")
    print("="*60)
    
    # Prepare MRNs for matching
    df1[mrn_col] = df1[mrn_col].astype(str)
    df2_with_mrn['MRN_clean'] = df2_with_mrn['MRN'].apply(
        lambda x: str(int(x))[:-2] if isinstance(x, float) and not pd.isna(x) else str(x)[:-2] if str(x) != 'nan' else str(x)
    )
    
    # Filter dataset 1 for overlapping patients
    df1_overlap = df1[df1[mrn_col].isin(overlap_mrns)].copy()
    df1_overlap['Source'] = 'Dataset1_CPT_T1D'
    
    # Filter dataset 2 for overlapping patients
    df2_overlap = df2_with_mrn[df2_with_mrn['MRN_clean'].isin(overlap_mrns)].copy()
    df2_overlap['Source'] = 'Dataset2_TCH'
    
    # Merge the overlapping data from both datasets
    # First, rename MRN column in df1_overlap for consistency
    df1_overlap = df1_overlap.rename(columns={mrn_col: 'MRN'})
    
    # For df2_overlap, use the clean MRN
    df2_overlap['MRN'] = df2_overlap['MRN_clean']
    df2_overlap = df2_overlap.drop('MRN_clean', axis=1)
    
    # Merge on MRN to get combined data
    combined_overlap = pd.merge(
        df1_overlap,
        df2_overlap,
        on='MRN',
        how='outer',
        suffixes=('_DS1', '_DS2')
    )
    
    # Save the combined overlap data
    output_file = "T1D_mike_data.csv"
    combined_overlap.to_csv(output_file, index=False)
    print(f"Saved overlapping patients data to: {output_file}")
    print(f"Total rows in overlap file: {len(combined_overlap)}")
    print(f"Total unique patients: {combined_overlap['MRN'].nunique()}")
    
    return combined_overlap

def generate_summary_statistics(df):
    """Generate comprehensive summary statistics for all columns"""
    print("\n" + "="*60)
    print("GENERATING SUMMARY STATISTICS")
    print("="*60)
    
    summary_list = []
    
    for col in df.columns:
        col_summary = {
            'Column': col,
            'Data_Type': str(df[col].dtype),
            'Non_Null_Count': df[col].notna().sum(),
            'Null_Count': df[col].isna().sum(),
            'Null_Percentage': f"{(df[col].isna().sum() / len(df) * 100):.2f}%",
            'Unique_Values': df[col].nunique()
        }
        
        # Add statistics for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            col_summary.update({
                'Mean': df[col].mean(),
                'Median': df[col].median(),
                'Min': df[col].min(),
                'Max': df[col].max(),
                'Std_Dev': df[col].std(),
                'Q1': df[col].quantile(0.25),
                'Q3': df[col].quantile(0.75)
            })
        else:
            # For categorical columns, get the most common value
            if df[col].notna().any():
                mode_value = df[col].mode()[0] if not df[col].mode().empty else None
                col_summary.update({
                    'Most_Common_Value': mode_value,
                    'Most_Common_Count': (df[col] == mode_value).sum() if mode_value else 0,
                    'Mean': 'N/A',
                    'Median': 'N/A',
                    'Min': 'N/A',
                    'Max': 'N/A',
                    'Std_Dev': 'N/A',
                    'Q1': 'N/A',
                    'Q3': 'N/A'
                })
            else:
                col_summary.update({
                    'Most_Common_Value': 'N/A',
                    'Most_Common_Count': 0,
                    'Mean': 'N/A',
                    'Median': 'N/A',
                    'Min': 'N/A',
                    'Max': 'N/A',
                    'Std_Dev': 'N/A',
                    'Q1': 'N/A',
                    'Q3': 'N/A'
                })
        
        summary_list.append(col_summary)
    
    # Create DataFrame from summary list
    summary_df = pd.DataFrame(summary_list)
    
    # Reorder columns for better readability
    column_order = ['Column', 'Data_Type', 'Non_Null_Count', 'Null_Count', 'Null_Percentage', 
                   'Unique_Values', 'Mean', 'Median', 'Min', 'Max', 'Std_Dev', 'Q1', 'Q3',
                   'Most_Common_Value', 'Most_Common_Count']
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in summary_df.columns]
    summary_df = summary_df[column_order]
    
    # Save summary statistics
    output_file = "summary_stats_mike.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"Saved summary statistics to: {output_file}")
    print(f"Summary generated for {len(summary_df)} columns")
    
    # Display first few rows of summary
    print("\nSample of summary statistics (first 5 columns):")
    print(summary_df.head())
    
    return summary_df

def perform_venn_analysis(df1, df2_with_mrn):
    """Perform Venn diagram analysis for all patients"""
    
    # Find MRN column in dataset 1
    # mrn_col = find_mrn_column(df1)
    # print(f"\nUsing '{mrn_col}' as MRN column in dataset 1")
    mrn_col = "MRN"
    # Get unique MRNs from dataset 1
    df1[mrn_col] = df1[mrn_col].astype(str)
    mrns_dataset1 = set(df1[mrn_col].dropna().unique())
    mrns_dataset1.discard('nan')  # Remove 'nan' string if present
    print(f"Unique MRNs in dataset 1: {len(mrns_dataset1)}")
    # NOTE: do not print raw MRN values here (real patient identifiers) --
    # counts only, for a public repo.
    # Get unique MRNs from dataset 2 (now with MRN column added)
    df2_with_mrn['MRN'] = df2_with_mrn['MRN'].apply(lambda x: str(int(x)) if isinstance(x, float) and not pd.isna(x) else str(x))

    mrns_dataset2 = set(df2_with_mrn['MRN'].dropna().unique())
    mrns_dataset2.discard('nan')  # Remove 'nan' string if present
    mrns_dataset2 = {mrn[:-2] for mrn in mrns_dataset2}
    print(f"Unique MRNs in dataset 2 (after mapping): {len(mrns_dataset2)}")
    # NOTE: do not print raw MRN values here (real patient identifiers) --
    # counts only, for a public repo.
    # Calculate overlaps
    overlap = mrns_dataset1.intersection(mrns_dataset2)
    only_dataset1 = mrns_dataset1 - mrns_dataset2
    only_dataset2 = mrns_dataset2 - mrns_dataset1
    
    print("\n" + "="*60)
    print("VENN DIAGRAM ANALYSIS - ALL PATIENTS")
    print("="*60)
    print(f"1. Total unique patients in CPT T1D study (Dataset 1): {len(mrns_dataset1)}")
    print(f"2. Total unique patients in TCH data (Dataset 2): {len(mrns_dataset2)}")
    print(f"3. Patients in BOTH datasets (overlap): {len(overlap)}")
    print(f"4. Patients ONLY in Dataset 1 (CPT T1D): {len(only_dataset1)}")
    print(f"5. Patients ONLY in Dataset 2 (TCH): {len(only_dataset2)}")
    
    # NOTE: do not print raw MRN values here (real patient identifiers) --
    # counts only, for a public repo.
    
    # Create Venn diagram
    plt.figure(figsize=(10, 6))
    venn2(subsets=(len(only_dataset1), len(only_dataset2), len(overlap)),
          set_labels=('CPT T1D Study', 'TCH Data'))
    plt.title('Patient Overlap - All Patients')
    plt.show()
    
    return overlap, mrn_col

def perform_hypoglycemia_analysis(df1, df2_with_mrn, mrn_col):
    """Perform Venn diagram analysis for patients with hypoglycemia"""
    
    # Find hypoglycemia column
    hypo_col = None
    for col in df1.columns:
        if 'Sev Hypogly Event' in col:
            hypo_col = col
            break
    
    if hypo_col is None:
        print("\nWarning: Could not find 'Sev Hypogly Event' column")
        print("Available columns:", df1.columns.tolist())
        return
    
    print(f"\nUsing '{hypo_col}' as hypoglycemia column")
    
    # Filter dataset 1 for patients with hypoglycemia (Yes values)
    df1_hypo = df1[df1[hypo_col].astype(str).str.strip().str.lower() == 'yes']
    
    # Get unique MRNs with hypoglycemia from dataset 1
    df1_hypo[mrn_col] = df1_hypo[mrn_col].astype(str)
    mrns_hypo_dataset1 = set(df1_hypo[mrn_col].dropna().unique())
    mrns_hypo_dataset1.discard('nan')
    print(f"Unique MRNs with hypoglycemia (Yes) in dataset 1: {len(mrns_hypo_dataset1)}")
    
    # Get all MRNs from dataset 2
    mrns_dataset2 = set(df2_with_mrn['MRN'].dropna().unique())
    mrns_dataset2.discard('nan')
    mrns_dataset2 = {mrn[:-2] for mrn in mrns_dataset2}    
    # Find overlap between hypoglycemia patients in dataset 1 and all patients in dataset 2
    overlap_hypo = mrns_hypo_dataset1.intersection(mrns_dataset2)
    only_dataset1_hypo = mrns_hypo_dataset1 - mrns_dataset2
    
    print("\n" + "="*60)
    print("VENN DIAGRAM ANALYSIS - PATIENTS WITH HYPOGLYCEMIA")
    print("="*60)
    print(f"1. Patients with hypoglycemia (Yes) in CPT T1D study (Dataset 1): {len(mrns_hypo_dataset1)}")
    print(f"2. Hypoglycemia patients from Dataset 1 that are also in Dataset 2: {len(overlap_hypo)}")
    print(f"3. Hypoglycemia patients ONLY in Dataset 1: {len(only_dataset1_hypo)}")
    
    # Calculate percentage
    if len(mrns_hypo_dataset1) > 0:
        overlap_percentage = (len(overlap_hypo) / len(mrns_hypo_dataset1)) * 100
        print(f"4. Percentage of hypoglycemia patients in both datasets: {overlap_percentage:.1f}%")
    
    # Create Venn diagram for hypoglycemia patients
    if len(mrns_hypo_dataset1) > 0:
        plt.figure(figsize=(10, 6))
        # Show hypoglycemia patients from dataset 1 and their overlap with dataset 2
        venn2(subsets=(len(only_dataset1_hypo), 
                      len(mrns_dataset2) - len(overlap_hypo),  # Dataset 2 patients without hypoglycemia from dataset 1
                      len(overlap_hypo)),
              set_labels=('CPT T1D Hypoglycemia', 'TCH Data (All Patients)'))
        plt.title('Hypoglycemia Patients from Dataset 1 vs All Patients in Dataset 2')
        plt.show()
        
    # Also create a simple focused diagram
    if len(mrns_hypo_dataset1) > 0:
        plt.figure(figsize=(10, 6))
        sizes = [len(only_dataset1_hypo), len(overlap_hypo)]
        labels = [f'Only in Dataset 1\n({len(only_dataset1_hypo)})', 
                 f'In Both Datasets\n({len(overlap_hypo)})']
        colors = ['lightcoral', 'lightgreen']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title(f'Hypoglycemia Patients Distribution\n(Total: {len(mrns_hypo_dataset1)} patients)')
        plt.axis('equal')
        plt.show()

def main():
    """Main execution function"""
    print("T1D Dataset Venn Diagram Analysis")
    print("="*60)
    
    # Load datasets
    df1, df2, crosswalk = load_datasets()
    
    if df1 is None or df2 is None or crosswalk is None:
        print("Failed to load datasets. Please check file paths.")
        return
    
    # Step 1: Add MRN column to dataset 2 using crosswalk
    df2_with_mrn = add_mrn_to_dataset2(df2, crosswalk)
    
    # Step 2: Perform analysis for all patients
    overlap_mrns, mrn_col = perform_venn_analysis(df1, df2_with_mrn)
    
    # Step 3: Save overlapping patients data
    combined_overlap_data = save_overlap_data(df1, df2_with_mrn, overlap_mrns, mrn_col)
    
    # Step 4: Generate and save summary statistics
    summary_stats = generate_summary_statistics(combined_overlap_data)
    
    # Step 5: Perform analysis for patients with hypoglycemia
    perform_hypoglycemia_analysis(df1, df2_with_mrn, mrn_col)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("Output files created:")
    print("1. T1D_mike_data.csv - Contains all overlapping patients data")
    print("2. summary_stats_mike.csv - Contains summary statistics for all columns")
    print("="*60)

if __name__ == "__main__":
    main()