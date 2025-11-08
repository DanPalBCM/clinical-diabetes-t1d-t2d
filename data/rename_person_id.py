import pandas as pd
from pathlib import Path

# Input file paths
files = [
    "/home/sagemaker-user/T2D/data/CROSSWALK_PATINENTIDS_08052025.csv",
    "/home/sagemaker-user/T2D/data/CROSSWALK_PATINENTIDS.csv"
]

for file in files:
    # Load CSV
    df = pd.read_csv(file)
    
    # Rename column
    if "PEDSNET_ID" in df.columns:
        df = df.rename(columns={"PEDSNET_ID": "person_id"})
    else:
        print(f"⚠️ Column 'PEDSNET_ID' not found in {file}, skipping...")
        continue
    
    # Create new filename
    new_file = Path(file).with_name(Path(file).stem + "_person_id.csv")
    
    # Save updated file
    df.to_csv(new_file, index=False)
    print(f"✅ Saved: {new_file}")
