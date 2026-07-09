import pandas as pd

# Read the CSV file
df = pd.read_csv('T1D_mike_data.csv')

# Select only the 'PEDSNET_ID' column and rename it to 'person_id'
df_person_id = df[['PEDSNET_ID']].rename(columns={'PEDSNET_ID': 'person_id'})

# Save the dataframe with only person_id to a new CSV file
df_person_id.to_csv('Mike_T1D_person_id.csv', index=False)

# List of specific columns to remove
columns_to_remove = [
    'MRN', 'Patient', 'Date of Dx', 'CGM Date', 'PATIENTID',
    'PEDSNET_ID', 'TCH_SOURCE_ID', 'PAT_MRN_ID', 'Source_DS2', 
    'Source_DS1', 'Last Pump Rx Date', 'Retinal Eye Exam Order Dt'
    'Last Endo OV', 'Last Endo Provider', 'Last Endo Dept',
    'Last CDE Enc', 'Last RD Enc', 'Last CDE Dept',
    'Last RD Dept', 'Last SW Enc', 'Last SW Dept', 'Last Psychology Enc Dept',
    'Lst Enc Nutrition', 'Last Canceled Dep', 'Last NoShow Dep', 'Celiac Screen Order Dt',
    'Last Lipid Panel', 'Last LDL Dt', 'Last Microalbumin Dt', 'Last Creatinine Dt',
    'Last BUN Dt', 'Last Ur Micro:Creat Dt', 'RDT ID', 'MyChart Status', 'Pt Comm Pref'
]

# Remove the specific columns (only if they exist in the dataframe)
columns_to_remove_existing = [col for col in columns_to_remove if col in df.columns]
df = df.drop(columns=columns_to_remove_existing)

# Remove columns that end with "Date"
date_columns = [col for col in df.columns if col.endswith('Date')]
df = df.drop(columns=date_columns)

# Print all remaining column names
print("Remaining columns in the cleaned dataframe:")
print("=" * 50)
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")
print("=" * 50)
print(f"\nTotal number of columns: {len(df.columns)}")
print(f"Total number of rows: {len(df)}")

# Save the cleaned dataframe to a new CSV file
df.to_csv('T1D_mike_data_cleaned.csv', index=False)
print("\nCleaned data saved to 'T1D_mike_data_cleaned.csv'")