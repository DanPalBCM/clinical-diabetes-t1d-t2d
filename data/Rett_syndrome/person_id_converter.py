import pandas as pd
import os

def process_datasets():
    """
    Process three CSV files to add person_id column based on PAT_MRN_ID crosswalk.
    """
    
    # File paths
    rett_file = '/home/sagemaker-user/T2D/data/Rett_syndrome/Rett_syndrome_subset.csv'
    melax_file = '/home/sagemaker-user/T2D/data/Melax_sleeping/MRN_to_PatientID_all_combinations.csv'
    crosswalk_file = '/home/sagemaker-user/T2D/data_crosswalk_all/crosswalk_mrn.csv'
    
    try:
        # Read the datasets
        print("Reading datasets...")
        rett_df = pd.read_csv(rett_file)
        melax_df = pd.read_csv(melax_file)
        crosswalk_df = pd.read_csv(crosswalk_file)
        
        print(f"Rett syndrome dataset shape: {rett_df.shape}")
        print(f"Melax dataset shape: {melax_df.shape}")
        print(f"Crosswalk dataset shape: {crosswalk_df.shape}")
        
        # Verify required columns exist
        required_cols_main = ['PAT_ID', 'PAT_MRN_ID']
        required_cols_crosswalk = ['PAT_MRN_ID', 'TCH_SOURCE_ID', 'PEDSNET_ID']
        
        for df_name, df, cols in [('Rett syndrome', rett_df, required_cols_main),
                                  ('Melax', melax_df, required_cols_main),
                                  ('Crosswalk', crosswalk_df, required_cols_crosswalk)]:
            missing_cols = [col for col in cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {df_name} dataset: {missing_cols}")
        
        # Create crosswalk mapping from PAT_MRN_ID to PEDSNET_ID
        print("Creating crosswalk mapping...")
        
        # Check for duplicates in crosswalk dataset
        duplicate_mrns = crosswalk_df[crosswalk_df.duplicated(subset=['PAT_MRN_ID'], keep=False)]
        if not duplicate_mrns.empty:
            print(f"Warning: Found {len(duplicate_mrns)} duplicate PAT_MRN_ID values in crosswalk dataset")
            print("Duplicate MRN IDs:")
            print(duplicate_mrns[['PAT_MRN_ID', 'PEDSNET_ID']].sort_values('PAT_MRN_ID'))
            
            # Remove duplicates, keeping the first occurrence
            print("Removing duplicates, keeping first occurrence of each PAT_MRN_ID...")
            crosswalk_df_clean = crosswalk_df.drop_duplicates(subset=['PAT_MRN_ID'], keep='first')
            print(f"Crosswalk dataset reduced from {len(crosswalk_df)} to {len(crosswalk_df_clean)} records")
        else:
            crosswalk_df_clean = crosswalk_df.copy()
            print("No duplicate PAT_MRN_ID values found in crosswalk dataset")
        
        crosswalk_mapping = dict(zip(crosswalk_df_clean['PAT_MRN_ID'], crosswalk_df_clean['PEDSNET_ID']))
        print(f"Created mapping for {len(crosswalk_mapping)} unique MRN IDs")
        
        # Add person_id column to Rett syndrome dataset
        print("Adding person_id to Rett syndrome dataset...")
        rett_df['person_id'] = rett_df['PAT_MRN_ID'].map(crosswalk_mapping)
        
        # Add person_id column to Melax dataset
        print("Adding person_id to Melax dataset...")
        melax_df['person_id'] = melax_df['PAT_MRN_ID'].map(crosswalk_mapping)
        
        # Check how many records have person_id
        rett_with_person_id = rett_df['person_id'].notna().sum()
        melax_with_person_id = melax_df['person_id'].notna().sum()
        
        print(f"Rett syndrome records with person_id: {rett_with_person_id}/{len(rett_df)}")
        print(f"Melax records with person_id: {melax_with_person_id}/{len(melax_df)}")
        
        # Drop records where person_id is null
        print("Removing records with null person_id...")
        rett_df_clean = rett_df.dropna(subset=['person_id']).copy()
        melax_df_clean = melax_df.dropna(subset=['person_id']).copy()
        
        print(f"Rett syndrome dataset after cleaning: {rett_df_clean.shape}")
        print(f"Melax dataset after cleaning: {melax_df_clean.shape}")
        
        # Define output file paths
        output_dir_rett = '/home/sagemaker-user/T2D/data/Rett_syndrome/'
        output_dir_melax = '/home/sagemaker-user/T2D/data/Melax_sleeping/'
        
        rett_output_file = os.path.join(output_dir_rett, 'Rett_syndrome_subset_person_id.csv')
        melax_output_file = os.path.join(output_dir_melax, 'Melax_person_id.csv')
        
        # Save the cleaned datasets
        print("Saving cleaned datasets...")
        rett_df_clean.to_csv(rett_output_file, index=False)
        melax_df_clean.to_csv(melax_output_file, index=False)
        
        print(f"Successfully saved:")
        print(f"- Rett syndrome dataset: {rett_output_file}")
        print(f"- Melax dataset: {melax_output_file}")
        
        # Display summary statistics
        print("\n=== SUMMARY ===")
        print(f"Original Rett syndrome records: {len(rett_df)}")
        print(f"Final Rett syndrome records: {len(rett_df_clean)}")
        print(f"Records removed: {len(rett_df) - len(rett_df_clean)}")
        
        print(f"\nOriginal Melax records: {len(melax_df)}")
        print(f"Final Melax records: {len(melax_df_clean)}")
        print(f"Records removed: {len(melax_df) - len(melax_df_clean)}")
        
        # Display sample of final datasets
        print("\n=== SAMPLE DATA ===")
        print("Rett syndrome dataset (first 5 rows):")
        print(rett_df_clean.head())
        
        print("\nMelax dataset (first 5 rows):")
        print(melax_df_clean.head())
        
        return rett_df_clean, melax_df_clean
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        return None, None
    except ValueError as e:
        print(f"Error: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # Run the processing
    result = process_datasets()
    if result is not None:
        rett_result, melax_result = result
        print("Processing completed successfully!")