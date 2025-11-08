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
        df1 = pd.read_excel("/home/sagemaker-user/T2D/src_Mike/data/MRNs for Hypoglycemia (1).xlsx")
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

def perform_venn_analysis(df1, df2, crosswalk):
    """Perform Venn diagram analysis for all patients"""
    
    # Find MRN column in dataset 1
    mrn_col = find_mrn_column(df1)
    print(f"\nUsing '{mrn_col}' as MRN column in dataset 1")
    
    # Get unique MRNs from dataset 1
    mrns_dataset1 = set(df1[mrn_col].dropna().astype(str).unique())
    print(f"Unique MRNs in dataset 1: {len(mrns_dataset1)}")
    
    # Get unique patient IDs from dataset 2
    # Try both PATIENTID and TCH_SOURCE_ID (in case column names vary)
    patient_id_col = 'PATIENTID' if 'PATIENTID' in df2.columns else 'TCH_SOURCE_ID'
    patientids_dataset2 = set(df2[patient_id_col].dropna().astype(str).unique())
    print(f"Unique patient IDs in dataset 2: {len(patientids_dataset2)}")
    
    # Map MRNs to patient IDs using crosswalk
    # Create mapping dictionary
    crosswalk['PAT_MRN_ID'] = crosswalk['PAT_MRN_ID'].astype(str)
    crosswalk['TCH_SOURCE_ID'] = crosswalk['TCH_SOURCE_ID'].astype(str)
    
    mrn_to_patientid = dict(zip(crosswalk['PAT_MRN_ID'], crosswalk['TCH_SOURCE_ID']))
    
    # Find MRNs from dataset 1 that have corresponding patient IDs in dataset 2
    mrns_with_mapping = set()
    mapped_patientids = set()
    
    for mrn in mrns_dataset1:
        if mrn in mrn_to_patientid:
            patient_id = mrn_to_patientid[mrn]
            if patient_id in patientids_dataset2:
                mrns_with_mapping.add(mrn)
                mapped_patientids.add(patient_id)
    
    # Calculate overlaps
    overlap_count = len(mrns_with_mapping)
    only_dataset1 = len(mrns_dataset1) - overlap_count
    only_dataset2 = len(patientids_dataset2) - len(mapped_patientids)
    
    print("\n" + "="*60)
    print("VENN DIAGRAM ANALYSIS - ALL PATIENTS")
    print("="*60)
    print(f"1. Total unique patients in CPT T1D study (Dataset 1): {len(mrns_dataset1)}")
    print(f"2. Total unique patients in TCH data (Dataset 2): {len(patientids_dataset2)}")
    print(f"3. Patients in BOTH datasets (overlap): {overlap_count}")
    print(f"4. Patients ONLY in Dataset 1 (CPT T1D): {only_dataset1}")
    print(f"5. Patients ONLY in Dataset 2 (TCH): {only_dataset2}")
    
    # Create Venn diagram
    plt.figure(figsize=(10, 6))
    venn2(subsets=(only_dataset1, only_dataset2, overlap_count),
          set_labels=('CPT T1D Study', 'TCH Data'))
    plt.title('Patient Overlap - All Patients')
    plt.show()
    
    return mrns_with_mapping, mrn_col

def perform_hypoglycemia_analysis(df1, df2, crosswalk, mrn_col):
    """Perform Venn diagram analysis for patients with hypoglycemia"""
    
    # Find hypoglycemia column
    hypo_col = None
    for col in df1.columns:
        if 'Sev Hypogly Event' in col or 'hypogly' in col.lower():
            hypo_col = col
            break
    
    if hypo_col is None:
        print("\nWarning: Could not find 'Sev Hypogly Event' column")
        print("Available columns:", df1.columns.tolist())
        return
    
    print(f"\nUsing '{hypo_col}' as hypoglycemia column")
    
    # Filter dataset 1 for patients with hypoglycemia
    # Assuming any non-null, non-zero value indicates hypoglycemia event
    df1_hypo = df1[df1[hypo_col].notna() & (df1[hypo_col] != 0) & (df1[hypo_col] != '0')]
    
    # Get unique MRNs with hypoglycemia from dataset 1
    mrns_hypo_dataset1 = set(df1_hypo[mrn_col].dropna().astype(str).unique())
    print(f"Unique MRNs with hypoglycemia in dataset 1: {len(mrns_hypo_dataset1)}")
    
    # For dataset 2, we need to identify hypoglycemia patients
    # This might require additional logic depending on how hypoglycemia is recorded in dataset 2
    # For now, we'll assume we need to map from dataset 1
    
    # Get patient IDs from dataset 2
    patient_id_col = 'PATIENTID' if 'PATIENTID' in df2.columns else 'TCH_SOURCE_ID'
    patientids_dataset2 = set(df2[patient_id_col].dropna().astype(str).unique())
    
    # Map MRNs to patient IDs using crosswalk
    crosswalk['PAT_MRN_ID'] = crosswalk['PAT_MRN_ID'].astype(str)
    crosswalk['TCH_SOURCE_ID'] = crosswalk['TCH_SOURCE_ID'].astype(str)
    mrn_to_patientid = dict(zip(crosswalk['PAT_MRN_ID'], crosswalk['TCH_SOURCE_ID']))
    
    # Find hypoglycemia MRNs from dataset 1 that have corresponding patient IDs in dataset 2
    mrns_hypo_with_mapping = set()
    mapped_hypo_patientids = set()
    
    for mrn in mrns_hypo_dataset1:
        if mrn in mrn_to_patientid:
            patient_id = mrn_to_patientid[mrn]
            if patient_id in patientids_dataset2:
                mrns_hypo_with_mapping.add(mrn)
                mapped_hypo_patientids.add(patient_id)
    
    # Calculate overlaps for hypoglycemia patients
    overlap_hypo = len(mrns_hypo_with_mapping)
    only_dataset1_hypo = len(mrns_hypo_dataset1) - overlap_hypo
    
    print("\n" + "="*60)
    print("VENN DIAGRAM ANALYSIS - PATIENTS WITH HYPOGLYCEMIA")
    print("="*60)
    print(f"1. Patients with hypoglycemia in CPT T1D study (Dataset 1): {len(mrns_hypo_dataset1)}")
    print(f"2. Patients with hypoglycemia in BOTH datasets: {overlap_hypo}")
    print(f"3. Patients with hypoglycemia ONLY in Dataset 1: {only_dataset1_hypo}")
    
    # Note: We cannot determine patients with hypoglycemia ONLY in dataset 2 
    # without hypoglycemia information in dataset 2
    print("\nNote: Cannot determine hypoglycemia patients only in Dataset 2 without")
    print("      hypoglycemia event information in the TCH dataset")
    
    # Create Venn diagram for hypoglycemia patients
    if len(mrns_hypo_dataset1) > 0:
        plt.figure(figsize=(10, 6))
        # For this diagram, we only show dataset 1 hypoglycemia patients and their overlap
        venn2(subsets=(only_dataset1_hypo, 0, overlap_hypo),
              set_labels=('CPT T1D Hypoglycemia', 'TCH Data (mapped)'))
        plt.title('Patient Overlap - Hypoglycemia Patients from Dataset 1')
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
    
    # Perform analysis for all patients
    mrns_with_mapping, mrn_col = perform_venn_analysis(df1, df2, crosswalk)
    
    # Perform analysis for patients with hypoglycemia
    perform_hypoglycemia_analysis(df1, df2, crosswalk, mrn_col)
    
    print("\n" + "="*60)
    print("Analysis complete!")

if __name__ == "__main__":
    main()