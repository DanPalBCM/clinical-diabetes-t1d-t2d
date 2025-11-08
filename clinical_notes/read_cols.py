import pandas as pd

# Load the CSV file
file_path = '/home/sagemaker-user/T2D/clinical_notes/T2D_patients.csv'
df = pd.read_csv(file_path)

# Display all column names
print("Columns in the dataset:")
print(df.columns)

# Display 5 samples of each column
print("\n5 samples from each column:")
print(df.head(5))
