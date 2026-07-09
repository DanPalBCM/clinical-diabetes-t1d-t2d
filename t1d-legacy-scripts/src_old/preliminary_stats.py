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

# Create directories for results
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load the data
print("Loading data...")
data = pd.read_excel('../data/T2DPopulation_732025.xlsx')

# Create a timestamp for unique file naming
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 0. Print all column names
print("\n=== COLUMN NAMES ===")
columns = data.columns.tolist()
for i, col in enumerate(columns):
    print(f"{i+1}. {col}")

# Save column names to file
with open(f'results/column_names_{timestamp}.txt', 'w') as f:
    f.write("=== COLUMN NAMES ===\n")
    for i, col in enumerate(columns):
        f.write(f"{i+1}. {col}\n")

# 1. Empty values analysis
print("\n=== EMPTY VALUES ANALYSIS ===")
empty_values = pd.DataFrame({
    'Column': data.columns,
    'Missing_Count': data.isnull().sum(),
    'Missing_Percentage': (data.isnull().sum() / len(data)) * 100
})
empty_values = empty_values.sort_values('Missing_Percentage', ascending=False)

print("\nEmpty values per column:")
print(empty_values.to_string())

# Save empty values analysis
empty_values.to_csv(f'results/empty_values_analysis_{timestamp}.csv', index=False)

# Plot distribution of top 10 columns with missing values
top_10_empty = empty_values.head(10)
if len(top_10_empty) > 0:
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_10_empty)), top_10_empty['Missing_Percentage'])
    plt.xticks(range(len(top_10_empty)), top_10_empty['Column'], rotation=45, ha='right')
    plt.ylabel('Missing Values (%)')
    plt.title('Top 10 Columns with Missing Values')
    plt.tight_layout()
    plt.savefig(f'figures/missing_values_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Quality control checks
print("\n=== QUALITY CONTROL CHECKS ===")
qc_results = []

# Check for duplicate patients (assuming first column or 'ID' column contains patient IDs)
id_column = None
for col in data.columns:
    if 'id' in col.lower() or 'patient' in col.lower():
        id_column = col
        break
if id_column is None:
    id_column = data.columns[0]  # Use first column if no ID column found

duplicates = data[data.duplicated(subset=[id_column], keep=False)]
qc_results.append(f"Duplicate check on '{id_column}': {len(duplicates)} duplicate rows found")
print(f"Duplicate patients: {len(duplicates)} found")

if len(duplicates) > 0:
    duplicates.to_csv(f'results/duplicate_patients_{timestamp}.csv', index=False)

# Additional quality checks
# Check for negative values in columns that shouldn't have them
numeric_cols = data.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if any(keyword in col.lower() for keyword in ['age', 'weight', 'height', 'bmi', 'glucose', 'pressure']):
        negative_values = data[data[col] < 0]
        if len(negative_values) > 0:
            qc_results.append(f"Negative values in '{col}': {len(negative_values)} found")

# Check for outliers using IQR method
outlier_summary = {}
for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)]
    if len(outliers) > 0:
        outlier_summary[col] = len(outliers)

# Save quality control results
with open(f'results/quality_control_{timestamp}.txt', 'w') as f:
    f.write("=== QUALITY CONTROL RESULTS ===\n\n")
    for result in qc_results:
        f.write(result + '\n')
    f.write(f"\n=== OUTLIERS (IQR METHOD) ===\n")
    for col, count in outlier_summary.items():
        f.write(f"{col}: {count} outliers\n")

# 3. Spearman correlation plot
print("\n=== SPEARMAN CORRELATION ANALYSIS ===")
# Select only numeric columns for correlation
numeric_data = data.select_dtypes(include=[np.number])

# Calculate Spearman correlation
spearman_corr = numeric_data.corr(method='spearman')

# Save correlation matrix
spearman_corr.to_csv(f'results/spearman_correlation_matrix_{timestamp}.csv')

# Plot correlation heatmap
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
sns.heatmap(spearman_corr, mask=mask, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot=True if len(numeric_cols) < 20 else False, fmt='.2f')
plt.title('Spearman Correlation Matrix')
plt.tight_layout()
plt.savefig(f'figures/spearman_correlation_heatmap_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. UMAP and PCA
print("\n=== DIMENSIONALITY REDUCTION ===")
# Prepare data for dimensionality reduction
# Handle missing values
imputer = SimpleImputer(strategy='median')
numeric_data_imputed = pd.DataFrame(
    imputer.fit_transform(numeric_data),
    columns=numeric_data.columns,
    index=numeric_data.index
)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data_imputed)

# PCA
print("Performing PCA...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Plot PCA
plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA - First Two Components')
plt.grid(True, alpha=0.3)
plt.savefig(f'figures/pca_plot_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.close()

# UMAP
print("Performing UMAP...")
reducer = umap.UMAP(n_components=2, random_state=42)
umap_result = reducer.fit_transform(scaled_data)

# Plot UMAP
plt.figure(figsize=(10, 8))
plt.scatter(umap_result[:, 0], umap_result[:, 1], alpha=0.6)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP Projection')
plt.grid(True, alpha=0.3)
plt.savefig(f'figures/umap_plot_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.close()

# Save PCA and UMAP results
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_df.to_csv(f'results/pca_coordinates_{timestamp}.csv', index=False)

umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])
umap_df.to_csv(f'results/umap_coordinates_{timestamp}.csv', index=False)

# Save explained variance for PCA
with open(f'results/pca_explained_variance_{timestamp}.txt', 'w') as f:
    f.write("PCA Explained Variance Ratio:\n")
    for i, var in enumerate(pca.explained_variance_ratio_):
        f.write(f"PC{i+1}: {var:.4f} ({var*100:.2f}%)\n")

# Basic statistics for all variables
print("\n=== BASIC STATISTICS ===")
basic_stats = data.describe()
basic_stats.to_csv(f'results/basic_statistics_{timestamp}.csv')

# Additional statistics including non-numeric columns
full_stats = pd.DataFrame()
for col in data.columns:
    if data[col].dtype in ['int64', 'float64']:
        stats = {
            'Column': col,
            'Type': 'Numeric',
            'Count': data[col].count(),
            'Mean': data[col].mean(),
            'Std': data[col].std(),
            'Min': data[col].min(),
            '25%': data[col].quantile(0.25),
            'Median': data[col].median(),
            '75%': data[col].quantile(0.75),
            'Max': data[col].max(),
            'Missing': data[col].isnull().sum()
        }
    else:
        stats = {
            'Column': col,
            'Type': 'Categorical',
            'Count': data[col].count(),
            'Unique': data[col].nunique(),
            'Most_Common': data[col].mode()[0] if not data[col].mode().empty else 'N/A',
            'Missing': data[col].isnull().sum()
        }
    full_stats = pd.concat([full_stats, pd.DataFrame([stats])], ignore_index=True)

full_stats.to_csv(f'results/full_statistics_summary_{timestamp}.csv', index=False)

# Create a summary report
with open(f'results/analysis_summary_{timestamp}.txt', 'w') as f:
    f.write(f"=== T2D DATA ANALYSIS SUMMARY ===\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Dataset Shape: {data.shape[0]} rows × {data.shape[1]} columns\n")
    f.write(f"Numeric Columns: {len(numeric_cols)}\n")
    f.write(f"Non-numeric Columns: {len(data.columns) - len(numeric_cols)}\n")
    f.write(f"\nTop 5 columns with missing data:\n")
    for idx, row in empty_values.head(5).iterrows():
        f.write(f"  - {row['Column']}: {row['Missing_Percentage']:.2f}%\n")
    f.write(f"\nDuplicate Patients: {len(duplicates)}\n")
    f.write(f"Columns with outliers (IQR method): {len(outlier_summary)}\n")

print("\n=== ANALYSIS COMPLETE ===")
print(f"Results saved in 'results/' directory")
print(f"Figures saved in 'figures/' directory")
print(f"Timestamp: {timestamp}")