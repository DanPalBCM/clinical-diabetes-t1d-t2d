import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

def preprocess_measurements(value):
    """Convert measurement values to numeric, handling special cases."""
    if pd.isna(value):
        return np.nan
    
    # Convert to string for processing
    value_str = str(value).strip()
    
    # Handle empty strings
    if not value_str:
        return np.nan
    
    # Remove asterisks
    value_str = value_str.replace('*', '')
    
    # Handle "less than" values (e.g., "<5.0") - map to 0 (undetectable)
    if value_str.startswith('<'):
        return 0
    
    # Handle "greater than" values (e.g., ">100") - map to 1 (high)
    if value_str.startswith('>'):
        return 1
    
    # Try to convert to float
    try:
        return float(value_str)
    except:
        return np.nan

def categorize_insurance(insurance_str):
    """Categorize insurance into private, government, or self-pay."""
    if pd.isna(insurance_str):
        return 'Unknown'
    
    insurance_lower = str(insurance_str).lower()
    
    private_keywords = ['aetna', 'blue cross', 'cigna', 'humana', 'private', 
                       'united healthcare', 'commercial']
    government_keywords = ['government', 'medicaid', 'medicare', 'tricare', 
                          'chip', "texas children's health plan"]
    selfpay_keywords = ['international', 'self-pay', 'self pay', 'selfpay']
    
    for keyword in private_keywords:
        if keyword in insurance_lower:
            return 'Private'
    
    for keyword in government_keywords:
        if keyword in insurance_lower:
            return 'Government'
    
    for keyword in selfpay_keywords:
        if keyword in insurance_lower:
            return 'Self-Pay'
    
    return 'Other'

def calculate_p_value(group1, group2, is_categorical=False):
    """Calculate p-value for comparing two groups."""
    # Check if groups are empty
    if len(group1) == 0 or len(group2) == 0:
        return np.nan
    
    if is_categorical:
        # Chi-square test for categorical variables
        from scipy.stats import chi2_contingency
        try:
            # Create contingency table
            combined = pd.concat([group1, group2])
            contingency = pd.crosstab(
                pd.concat([pd.Series(['Hypoglycemia']*len(group1), index=group1.index),
                          pd.Series(['Control']*len(group2), index=group2.index)]),
                combined
            )
            
            # Check if contingency table has enough data
            if contingency.size == 0 or contingency.sum().sum() == 0:
                return np.nan
                
            chi2, p_value, _, _ = chi2_contingency(contingency)
            return p_value
        except ValueError as e:
            print(f"Chi-square test failed: {e}")
            return np.nan
    else:
        # T-test for continuous variables
        # Remove NaN values
        group1_clean = group1.dropna()
        group2_clean = group2.dropna()
        
        if len(group1_clean) < 2 or len(group2_clean) < 2:
            return np.nan
        
        try:
            _, p_value = stats.ttest_ind(group1_clean, group2_clean)
            return p_value
        except Exception as e:
            print(f"T-test failed: {e}")
            return np.nan


def create_descriptive_table(df):
    """Create descriptive statistics table when no comparison groups are available."""
    results = []
    
    # Age
    if 'Age' in df.columns:
        age_mean = df['Age'].mean()
        age_std = df['Age'].std()
        results.append({
            'Characteristic': 'Age (mean ± std)',
            'Entire Cohort': f"{age_mean:.1f} ± {age_std:.1f}",
            'Notes': f"Range: {df['Age'].min():.0f} - {df['Age'].max():.0f}"
        })
    
    # Gender
    if 'Gender' in df.columns:
        results.append({'Characteristic': 'Gender', 'Entire Cohort': '', 'Notes': ''})
        for gender in ['Male', 'Female']:
            count = (df['Gender'] == gender).sum()
            pct = (count / len(df)) * 100
            results.append({
                'Characteristic': f"  {gender}",
                'Entire Cohort': f"{count} ({pct:.1f}%)",
                'Notes': ''
            })
    
    return pd.DataFrame(results)


def process_omop_medications_conditions(df, hypoglycemia_group, control_group):
    """Process OMOP medication and condition variables (0/1 categorical)."""
    results = []
    
    omop_vars = [
        'OMOP_Insulins_onset', 'OMOP_Insulins_anytime', 
        'OMOP_Insulin_Pump_onset', 'OMOP_Insulin_Pump_anytime',
        'OMOP_CGM_Device_onset', 'OMOP_CGM_Device_anytime',
        'OMOP_Biguanide_onset', 'OMOP_Biguanide_anytime',
        'OMOP_GLP1_agonists_onset', 'OMOP_GLP1_agonists_anytime',
        'OMOP_SGLT2_inhibitors_onset', 'OMOP_SGLT2_inhibitors_anytime',
        'OMOP_ACE_Inhibitors_onset', 'OMOP_ACE_Inhibitors_anytime',
        'OMOP_Statins_onset', 'OMOP_Statins_anytime',
        'OMOP_Amylin_analogue_onset', 'OMOP_Amylin_analogue_anytime',
        'OMOP_DKA_onset', 'OMOP_DKA_anytime',
        'OMOP_Ketosis_onset', 'OMOP_Ketosis_anytime',
        'OMOP_Dyslipidemia_onset', 'OMOP_Dyslipidemia_anytime',
        'OMOP_Hypertension_onset', 'OMOP_Hypertension_anytime',
        'OMOP_Diabetic_Retinopathy_onset', 'OMOP_Diabetic_Retinopathy_anytime',
        'OMOP_Microalbuminuria_onset', 'OMOP_Microalbuminuria_anytime',
        'OMOP_Neuropathy_onset', 'OMOP_Neuropathy_anytime',
        'OMOP_Hypoglycemia_onset', 'OMOP_Hypoglycemia_anytime'
    ]
    
    # Add section header
    results.append({'Characteristic': 'OMOP Medications & Conditions', 
                   'Entire Cohort': '', 'Hypoglycemia Group': '', 
                   'Control Group': '', 'P-value': ''})
    
    for var in omop_vars:
        if var in df.columns:
            # Calculate counts for presence (1)
            all_count = (df[var] == 1).sum()
            all_pct = (all_count / len(df)) * 100
            hypo_count = (hypoglycemia_group[var] == 1).sum()
            hypo_pct = (hypo_count / len(hypoglycemia_group)) * 100 if len(hypoglycemia_group) > 0 else 0
            control_count = (control_group[var] == 1).sum()
            control_pct = (control_count / len(control_group)) * 100 if len(control_group) > 0 else 0
            
            # Calculate p-value
            p_val = calculate_p_value(hypoglycemia_group[var], control_group[var], True)
            
            # Clean variable name for display
            display_name = var.replace('OMOP_', '').replace('_', ' ')
            
            results.append({
                'Characteristic': f"  {display_name}",
                'Entire Cohort': f"{all_count} ({all_pct:.1f}%)",
                'Hypoglycemia Group': f"{hypo_count} ({hypo_pct:.1f}%)",
                'Control Group': f"{control_count} ({control_pct:.1f}%)",
                'P-value': f"{p_val:.4f}" if not pd.isna(p_val) else "N/A"
            })
    
    return pd.DataFrame(results)


def process_yes_no_categorical(df, hypoglycemia_group, control_group):
    """Process Yes/No/NaN categorical variables."""
    results = []
    
    yes_no_vars = ['CGM?', 'Pump?', 'PCOS', 'Seen by Mental Health',
                   'SW Enc Past 12m', 'Psych Enc Past 12m', 'RD Enc Past 12m',
                   'CDE Enc Past 12m', 'Pub/Self/Charity']
    
    # Add section header
    results.append({'Characteristic': 'Clinical Features', 
                   'Entire Cohort': '', 'Hypoglycemia Group': '', 
                   'Control Group': '', 'P-value': ''})
    
    for var in yes_no_vars:
        if var in df.columns:
            # Map NaN to 'Unknown' for this variable
            df[f'{var}_mapped'] = df[var].fillna('Unknown')
            hypoglycemia_group[f'{var}_mapped'] = hypoglycemia_group[var].fillna('Unknown')
            control_group[f'{var}_mapped'] = control_group[var].fillna('Unknown')
            
            # Get unique categories
            categories = df[f'{var}_mapped'].unique()
            
            # Add variable header
            results.append({'Characteristic': f'{var}', 'Entire Cohort': '', 
                          'Hypoglycemia Group': '', 'Control Group': '', 'P-value': ''})
            
            # Calculate p-value once for the variable
            p_val = calculate_p_value(hypoglycemia_group[f'{var}_mapped'], 
                                    control_group[f'{var}_mapped'], True)
            
            # Process each category
            for i, category in enumerate(['Yes', 'No', 'Unknown']):
                if category in categories:
                    all_count = (df[f'{var}_mapped'] == category).sum()
                    all_pct = (all_count / len(df)) * 100
                    hypo_count = (hypoglycemia_group[f'{var}_mapped'] == category).sum()
                    hypo_pct = (hypo_count / len(hypoglycemia_group)) * 100 if len(hypoglycemia_group) > 0 else 0
                    control_count = (control_group[f'{var}_mapped'] == category).sum()
                    control_pct = (control_count / len(control_group)) * 100 if len(control_group) > 0 else 0
                    
                    results.append({
                        'Characteristic': f"    {category}",
                        'Entire Cohort': f"{all_count} ({all_pct:.1f}%)",
                        'Hypoglycemia Group': f"{hypo_count} ({hypo_pct:.1f}%)",
                        'Control Group': f"{control_count} ({control_pct:.1f}%)",
                        'P-value': f"{p_val:.4f}" if i == 0 and not pd.isna(p_val) else ""
                    })
    
    return pd.DataFrame(results)


def create_characteristics_table(df):
    """Create the statistical characteristics table."""
    
    # Check if we have any hypoglycemia events
    unique_events = df['Sev Hypogly Event'].unique()
    print(f"Unique values in 'Sev Hypogly Event': {unique_events}")
    
    # Try to handle different possible encodings
    if df['Sev Hypogly Event'].dtype == 'object':
        # Convert string representations to numeric
        df['Sev Hypogly Event'] = df['Sev Hypogly Event'].astype(str).str.strip()
        df['Sev Hypogly Event'] = df['Sev Hypogly Event'].replace({'True': 1, 'False': 0, 'Yes': 1, 'No': 0})
        df['Sev Hypogly Event'] = pd.to_numeric(df['Sev Hypogly Event'], errors='coerce')
    
    # Separate groups
    hypoglycemia_group = df[df['Sev Hypogly Event'] == 1].copy()
    control_group = df[df['Sev Hypogly Event'] == 0].copy()
    
    print(f"Total cohort: {len(df)}")
    print(f"Hypoglycemia group: {len(hypoglycemia_group)}")
    print(f"Control group: {len(control_group)}")
    
    # Check if we have enough data for analysis
    if len(hypoglycemia_group) == 0:
        print("⚠️  WARNING: No patients with hypoglycemia events found!")
        print("This might indicate:")
        print("1. Data encoding issue (check if events are coded differently)")
        print("2. No hypoglycemia events in this dataset")
        print("3. Column name mismatch")
        return pd.DataFrame({'Message': ['No hypoglycemia events found - cannot create comparison table']})
    
    if len(control_group) == 0:
        print("⚠️  WARNING: No control patients found!")
        return pd.DataFrame({'Message': ['No control patients found - cannot create comparison table']})
    
    
    results = []
    
    # 1. Age (continuous)
    if 'Age' in df.columns:
        age_all_mean = df['Age'].mean()
        age_all_std = df['Age'].std()
        age_hypo_mean = hypoglycemia_group['Age'].mean()
        age_hypo_std = hypoglycemia_group['Age'].std()
        age_control_mean = control_group['Age'].mean()
        age_control_std = control_group['Age'].std()
        age_p = calculate_p_value(hypoglycemia_group['Age'], control_group['Age'], False)
        
        results.append({
            'Characteristic': 'Age (mean ± std)',
            'Entire Cohort': f"{age_all_mean:.1f} ± {age_all_std:.1f}",
            'Hypoglycemia Group': f"{age_hypo_mean:.1f} ± {age_hypo_std:.1f}",
            'Control Group': f"{age_control_mean:.1f} ± {age_control_std:.1f}",
            'P-value': f"{age_p:.4f}" if not pd.isna(age_p) else "N/A"
        })
    
    # 2. Gender (categorical)
    if 'Gender' in df.columns:
        results.append({'Characteristic': 'Gender', 'Entire Cohort': '', 
                       'Hypoglycemia Group': '', 'Control Group': '', 'P-value': ''})
        
        for gender in ['Male', 'Female']:
            all_count = (df['Gender'] == gender).sum()
            all_pct = (all_count / len(df)) * 100
            hypo_count = (hypoglycemia_group['Gender'] == gender).sum()
            hypo_pct = (hypo_count / len(hypoglycemia_group)) * 100 if len(hypoglycemia_group) > 0 else 0
            control_count = (control_group['Gender'] == gender).sum()
            control_pct = (control_count / len(control_group)) * 100 if len(control_group) > 0 else 0
            
            p_val = calculate_p_value(hypoglycemia_group['Gender'], control_group['Gender'], True) if gender == 'Male' else ''
            
            results.append({
                'Characteristic': f"  {gender}",
                'Entire Cohort': f"{all_count} ({all_pct:.1f}%)",
                'Hypoglycemia Group': f"{hypo_count} ({hypo_pct:.1f}%)",
                'Control Group': f"{control_count} ({control_pct:.1f}%)",
                'P-value': f"{p_val:.4f}" if p_val and not pd.isna(p_val) else ""
            })
    
    # 3. Ethnicity (categorical)
    if 'Ethnicity' in df.columns:
        results.append({'Characteristic': 'Ethnicity', 'Entire Cohort': '', 
                       'Hypoglycemia Group': '', 'Control Group': '', 'P-value': ''})
        
        ethnicities = ['Not Hispanic or Latino', 'Hispanic or Latino']
        for i, ethnicity in enumerate(ethnicities):
            all_count = df['Ethnicity'].str.contains(ethnicity.split(' or ')[0], case=False, na=False).sum()
            all_pct = (all_count / len(df)) * 100
            hypo_count = hypoglycemia_group['Ethnicity'].str.contains(ethnicity.split(' or ')[0], case=False, na=False).sum()
            hypo_pct = (hypo_count / len(hypoglycemia_group)) * 100 if len(hypoglycemia_group) > 0 else 0
            control_count = control_group['Ethnicity'].str.contains(ethnicity.split(' or ')[0], case=False, na=False).sum()
            control_pct = (control_count / len(control_group)) * 100 if len(control_group) > 0 else 0
            
            p_val = calculate_p_value(hypoglycemia_group['Ethnicity'], control_group['Ethnicity'], True) if i == 0 else ''
            
            results.append({
                'Characteristic': f"  {ethnicity}",
                'Entire Cohort': f"{all_count} ({all_pct:.1f}%)",
                'Hypoglycemia Group': f"{hypo_count} ({hypo_pct:.1f}%)",
                'Control Group': f"{control_count} ({control_pct:.1f}%)",
                'P-value': f"{p_val:.4f}" if p_val and not pd.isna(p_val) else ""
            })
    
    # 4. Race (categorical)
    if 'Race' in df.columns:
        results.append({'Characteristic': 'Race', 'Entire Cohort': '', 
                       'Hypoglycemia Group': '', 'Control Group': '', 'P-value': ''})
        
        # Define main race categories
        main_races = ['White', 'Black', 'Asian']
        
        for i, race in enumerate(main_races):
            all_count = df['Race'].str.contains(race, case=False, na=False).sum()
            all_pct = (all_count / len(df)) * 100
            hypo_count = hypoglycemia_group['Race'].str.contains(race, case=False, na=False).sum()
            hypo_pct = (hypo_count / len(hypoglycemia_group)) * 100 if len(hypoglycemia_group) > 0 else 0
            control_count = control_group['Race'].str.contains(race, case=False, na=False).sum()
            control_pct = (control_count / len(control_group)) * 100 if len(control_group) > 0 else 0
            
            p_val = calculate_p_value(hypoglycemia_group['Race'], control_group['Race'], True) if i == 0 else ''
            
            results.append({
                'Characteristic': f"  {race}",
                'Entire Cohort': f"{all_count} ({all_pct:.1f}%)",
                'Hypoglycemia Group': f"{hypo_count} ({hypo_pct:.1f}%)",
                'Control Group': f"{control_count} ({control_pct:.1f}%)",
                'P-value': f"{p_val:.4f}" if p_val and not pd.isna(p_val) else ""
            })
        
        # Other races
        other_mask = ~df['Race'].str.contains('|'.join(main_races), case=False, na=False)
        other_count = other_mask.sum()
        other_pct = (other_count / len(df)) * 100
        
        other_hypo = ~hypoglycemia_group['Race'].str.contains('|'.join(main_races), case=False, na=False)
        other_hypo_count = other_hypo.sum()
        other_hypo_pct = (other_hypo_count / len(hypoglycemia_group)) * 100 if len(hypoglycemia_group) > 0 else 0
        
        other_control = ~control_group['Race'].str.contains('|'.join(main_races), case=False, na=False)
        other_control_count = other_control.sum()
        other_control_pct = (other_control_count / len(control_group)) * 100 if len(control_group) > 0 else 0
        
        results.append({
            'Characteristic': f"  Other",
            'Entire Cohort': f"{other_count} ({other_pct:.1f}%)",
            'Hypoglycemia Group': f"{other_hypo_count} ({other_hypo_pct:.1f}%)",
            'Control Group': f"{other_control_count} ({other_control_pct:.1f}%)",
            'P-value': ""
        })
        
        # Other races
        other_mask = ~df['Race'].str.contains('|'.join(main_races), case=False, na=False)
        other_count = other_mask.sum()
        other_pct = (other_count / len(df)) * 100
        
        other_hypo = ~hypoglycemia_group['Race'].str.contains('|'.join(main_races), case=False, na=False)
        other_hypo_count = other_hypo.sum()
        other_hypo_pct = (other_hypo_count / len(hypoglycemia_group)) * 100 if len(hypoglycemia_group) > 0 else 0
        
        other_control = ~control_group['Race'].str.contains('|'.join(main_races), case=False, na=False)
        other_control_count = other_control.sum()
        other_control_pct = (other_control_count / len(control_group)) * 100 if len(control_group) > 0 else 0
        
        results.append({
            'Characteristic': f"  Other",
            'Entire Cohort': f"{other_count} ({other_pct:.1f}%)",
            'Hypoglycemia Group': f"{other_hypo_count} ({other_hypo_pct:.1f}%)",
            'Control Group': f"{other_control_count} ({other_control_pct:.1f}%)",
            'P-value': ""
        })
    
    # 5. Preferred Language
    if 'Preferred Language' in df.columns:
        results.append({'Characteristic': 'Preferred Language', 'Entire Cohort': '', 
                       'Hypoglycemia Group': '', 'Control Group': '', 'P-value': ''})
        
        languages = ['English', 'Spanish']
        for i, lang in enumerate(languages):
            all_count = df['Preferred Language'].str.contains(lang, case=False, na=False).sum()
            all_pct = (all_count / len(df)) * 100
            hypo_count = hypoglycemia_group['Preferred Language'].str.contains(lang, case=False, na=False).sum()
            hypo_pct = (hypo_count / len(hypoglycemia_group)) * 100 if len(hypoglycemia_group) > 0 else 0
            control_count = control_group['Preferred Language'].str.contains(lang, case=False, na=False).sum()
            control_pct = (control_count / len(control_group)) * 100 if len(control_group) > 0 else 0
            
            p_val = calculate_p_value(hypoglycemia_group['Preferred Language'], 
                                     control_group['Preferred Language'], True) if i == 0 else ''
            
            results.append({
                'Characteristic': f"  {lang}",
                'Entire Cohort': f"{all_count} ({all_pct:.1f}%)",
                'Hypoglycemia Group': f"{hypo_count} ({hypo_pct:.1f}%)",
                'Control Group': f"{control_count} ({control_pct:.1f}%)",
                'P-value': f"{p_val:.4f}" if p_val and not pd.isna(p_val) else ""
            })
        
        # Other languages
        other_mask = ~df['Preferred Language'].str.contains('English|Spanish', case=False, na=False)
        other_count = other_mask.sum()
        other_pct = (other_count / len(df)) * 100
        
        results.append({
            'Characteristic': f"  Other",
            'Entire Cohort': f"{other_count} ({other_pct:.1f}%)",
            'Hypoglycemia Group': f"{(~hypoglycemia_group['Preferred Language'].str.contains('English|Spanish', case=False, na=False)).sum()} ({((~hypoglycemia_group['Preferred Language'].str.contains('English|Spanish', case=False, na=False)).sum() / len(hypoglycemia_group)) * 100:.1f}%)" if len(hypoglycemia_group) > 0 else "0 (0.0%)",
            'Control Group': f"{(~control_group['Preferred Language'].str.contains('English|Spanish', case=False, na=False)).sum()} ({((~control_group['Preferred Language'].str.contains('English|Spanish', case=False, na=False)).sum() / len(control_group)) * 100:.1f}%)" if len(control_group) > 0 else "0 (0.0%)",
            'P-value': ""
        })
    
    # 6. Insurance (categorical - processed)
    insurance_col = None
    for col in df.columns:
        if 'insurance' in col.lower() or 'payor' in col.lower():
            insurance_col = col
            break
    
    if insurance_col:
        df['Insurance_Category'] = df[insurance_col].apply(categorize_insurance)
        hypoglycemia_group['Insurance_Category'] = hypoglycemia_group[insurance_col].apply(categorize_insurance)
        control_group['Insurance_Category'] = control_group[insurance_col].apply(categorize_insurance)
        
        results.append({'Characteristic': 'Insurance', 'Entire Cohort': '', 
                       'Hypoglycemia Group': '', 'Control Group': '', 'P-value': ''})
        
        for i, ins_type in enumerate(['Private', 'Government', 'Self-Pay']):
            all_count = (df['Insurance_Category'] == ins_type).sum()
            all_pct = (all_count / len(df)) * 100
            hypo_count = (hypoglycemia_group['Insurance_Category'] == ins_type).sum()
            hypo_pct = (hypo_count / len(hypoglycemia_group)) * 100 if len(hypoglycemia_group) > 0 else 0
            control_count = (control_group['Insurance_Category'] == ins_type).sum()
            control_pct = (control_count / len(control_group)) * 100 if len(control_group) > 0 else 0
            
            p_val = calculate_p_value(hypoglycemia_group['Insurance_Category'], 
                                     control_group['Insurance_Category'], True) if i == 0 else ''
            
            results.append({
                'Characteristic': f"  {ins_type}",
                'Entire Cohort': f"{all_count} ({all_pct:.1f}%)",
                'Hypoglycemia Group': f"{hypo_count} ({hypo_pct:.1f}%)",
                'Control Group': f"{control_count} ({control_pct:.1f}%)",
                'P-value': f"{p_val:.4f}" if p_val and not pd.isna(p_val) else ""
            })
    
    # 7. Diabetes Treatment Regimen
    treatment_col = None
    for col in df.columns:
        if 'treatment' in col.lower() or 'regimen' in col.lower():
            treatment_col = col
            break
    
    if treatment_col:
        results.append({'Characteristic': 'Diabetes Treatment Regimen', 'Entire Cohort': '', 
                       'Hypoglycemia Group': '', 'Control Group': '', 'P-value': ''})
        
        unique_treatments = df[treatment_col].value_counts().index[:5]  # Top 5 treatments
        
        for i, treatment in enumerate(unique_treatments):
            all_count = (df[treatment_col] == treatment).sum()
            all_pct = (all_count / len(df)) * 100
            hypo_count = (hypoglycemia_group[treatment_col] == treatment).sum()
            hypo_pct = (hypo_count / len(hypoglycemia_group)) * 100 if len(hypoglycemia_group) > 0 else 0
            control_count = (control_group[treatment_col] == treatment).sum()
            control_pct = (control_count / len(control_group)) * 100 if len(control_group) > 0 else 0
            
            p_val = calculate_p_value(hypoglycemia_group[treatment_col], 
                                     control_group[treatment_col], True) if i == 0 else ''
            
            results.append({
                'Characteristic': f"  {str(treatment)[:30]}",  # Truncate long names
                'Entire Cohort': f"{all_count} ({all_pct:.1f}%)",
                'Hypoglycemia Group': f"{hypo_count} ({hypo_pct:.1f}%)",
                'Control Group': f"{control_count} ({control_pct:.1f}%)",
                'P-value': f"{p_val:.4f}" if p_val and not pd.isna(p_val) else ""
            })
    
    return pd.DataFrame(results)

def process_measurements(df):
    """Process measurement columns and add to characteristics table."""
    measurement_cols = ['Last A1c', 'Last HbA1c', 'Last LDL', 'Last Microalbumin', 
                       'Last Creatinine', 'Last BUN', 'Last TSH', 'Last HDL', 
                       'Last Chol', 'Last Tg', 'Ur Cr', 'Last Ur Micro:Creat', 
                       'Last PHQ-2 Score', 'Last PHQ-9 Score', 'Last BMI']
    
    results = []
    
    # Separate groups
    hypoglycemia_group = df[df['Sev Hypogly Event'] == 1].copy()
    control_group = df[df['Sev Hypogly Event'] == 0].copy()
    
    for col in measurement_cols:
        if col in df.columns:
            # Preprocess the column
            df[f'{col}_processed'] = df[col].apply(preprocess_measurements)
            hypoglycemia_group[f'{col}_processed'] = hypoglycemia_group[col].apply(preprocess_measurements)
            control_group[f'{col}_processed'] = control_group[col].apply(preprocess_measurements)
            # Calculate statistics
            non_missing = df[f'{col}_processed'].notna().sum()
            missing_pct = (df[f'{col}_processed'].isna().sum() / len(df)) * 100
            
            # Skip if more than 50% missing
            if missing_pct > 50:
                print(f"Skipping {col}: {missing_pct:.1f}% missing values")
                continue
            
            # Calculate mean and std
            all_mean = df[f'{col}_processed'].mean()
            all_std = df[f'{col}_processed'].std()
            hypo_mean = hypoglycemia_group[f'{col}_processed'].mean()
            hypo_std = hypoglycemia_group[f'{col}_processed'].std()
            control_mean = control_group[f'{col}_processed'].mean()
            control_std = control_group[f'{col}_processed'].std()
            
            # Calculate p-value
            p_val = calculate_p_value(hypoglycemia_group[f'{col}_processed'], 
                                     control_group[f'{col}_processed'], False)
            
            results.append({
                'Characteristic': f'{col} (mean ± std)',
                'Entire Cohort': f"{all_mean:.2f} ± {all_std:.2f}" if not pd.isna(all_mean) else "N/A",
                'Hypoglycemia Group': f"{hypo_mean:.2f} ± {hypo_std:.2f}" if not pd.isna(hypo_mean) else "N/A",
                'Control Group': f"{control_mean:.2f} ± {control_std:.2f}" if not pd.isna(control_mean) else "N/A",
                'P-value': f"{p_val:.4f}" if not pd.isna(p_val) else "N/A"
            })
    
    return pd.DataFrame(results)


def main():
    # Load the data
    input_file = '/home/sagemaker-user/T2D/src_Mike/data/Mike_CPT_OMOP_data.csv'
    print(f"Loading data from: {input_file}")
    
    df = pd.read_csv(input_file)
    print(f"Original data shape: {df.shape}")
    print("\nExploring 'Sev Hypogly Event' column:")
    print(f"Unique values: {df['Sev Hypogly Event'].unique()}")
    print(f"Value counts:\n{df['Sev Hypogly Event'].value_counts()}")
    print(f"Data type: {df['Sev Hypogly Event'].dtype}")

    # Check if the column might have different encodings
    if df['Sev Hypogly Event'].dtype == 'object':
        print("String values found, checking for variations...")
        unique_str_values = df['Sev Hypogly Event'].astype(str).unique()
        print(f"String representations: {unique_str_values}")
    # Check for the key column
    if 'Sev Hypogly Event' not in df.columns:
        print("ERROR: 'Sev Hypogly Event' column not found in the dataset!")
        print("Available columns:", df.columns.tolist())
        return
    
    # Drop columns ending with 'Date'
    date_cols = [col for col in df.columns if col.endswith('Date')]
    print(f"\nDropping {len(date_cols)} date columns: {date_cols[:5]}...")  # Show first 5
    df_cleaned = df.drop(columns=date_cols)
    
    # Remove columns with >50% missing values (except Sev Hypogly Event and the new variables we want to keep)
    # Define variables to keep regardless of missing values
    keep_vars = ['Sev Hypogly Event', 'CGM?', 'Pump?', 'PCOS', 'Seen by Mental Health',
                 'SW Enc Past 12m', 'Psych Enc Past 12m', 'RD Enc Past 12m',
                 'CDE Enc Past 12m', 'Pub/Self/Charity']
    
    # Add OMOP variables to keep list
    omop_vars = [col for col in df_cleaned.columns if col.startswith('OMOP_')]
    keep_vars.extend(omop_vars)
    
    missing_pct = (df_cleaned.isna().sum() / len(df_cleaned)) * 100
    cols_to_drop = missing_pct[missing_pct > 50].index.tolist()
    cols_to_drop = [col for col in cols_to_drop if col not in keep_vars]
    
    print(f"\nDropping {len(cols_to_drop)} columns with >50% missing values")
    df_cleaned = df_cleaned.drop(columns=cols_to_drop)
    
    print(f"\nCleaned data shape: {df_cleaned.shape}")
    
    # Create output directory
    output_dir = 'cleaned_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save cleaned data
    cleaned_file = os.path.join(output_dir, 'T1D_OMOP_CPT_Mike_cleaned.csv')
    df_cleaned.to_csv(cleaned_file, index=False)
    print(f"\nCleaned data saved to: {cleaned_file}")
    
    # Create characteristics table
    print("\nCreating statistical characteristics table...")
    
    # Handle the Sev Hypogly Event encoding for analysis
    if df_cleaned['Sev Hypogly Event'].dtype == 'object':
        df_cleaned['Sev Hypogly Event'] = df_cleaned['Sev Hypogly Event'].astype(str).str.strip()
        df_cleaned['Sev Hypogly Event'] = df_cleaned['Sev Hypogly Event'].replace({'True': 1, 'False': 0, 'Yes': 1, 'No': 0})
        df_cleaned['Sev Hypogly Event'] = pd.to_numeric(df_cleaned['Sev Hypogly Event'], errors='coerce')
    
    # Separate groups for the new functions
    hypoglycemia_group = df_cleaned[df_cleaned['Sev Hypogly Event'] == 1].copy()
    control_group = df_cleaned[df_cleaned['Sev Hypogly Event'] == 0].copy()
    
    # Demographics and categorical variables
    demo_table = create_characteristics_table(df_cleaned)
    
    # Yes/No categorical variables
    yes_no_table = process_yes_no_categorical(df_cleaned, hypoglycemia_group, control_group)
    
    # OMOP medications and conditions
    omop_table = process_omop_medications_conditions(df_cleaned, hypoglycemia_group, control_group)
    
    # Measurement variables
    measurement_table = process_measurements(df_cleaned)
    
    # Combine all tables
    final_table = pd.concat([demo_table, yes_no_table, omop_table, measurement_table], ignore_index=True)
    
    # Save the table
    table_file = os.path.join(output_dir, 'statistical_characteristics_table.csv')
    final_table.to_csv(table_file, index=False)
    print(f"\nStatistical characteristics table saved to: {table_file}")
    
    # Display the table
    print("\n" + "="*100)
    print("STATISTICAL CHARACTERISTICS TABLE")
    print("="*100)
    print(final_table.to_string(index=False))
    
    # Summary statistics
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total patients: {len(df_cleaned)}")
    print(f"Hypoglycemia group (Sev Hypogly Event = 1): {(df_cleaned['Sev Hypogly Event'] == 1).sum()}")
    print(f"Control group (Sev Hypogly Event = 0): {(df_cleaned['Sev Hypogly Event'] == 0).sum()}")
    
    # Print statistical test information
    print("\n" + "="*50)
    print("STATISTICAL TESTS USED")
    print("="*50)
    print("For categorical variables we use Chi-square test for independence")
    print("For numerical variables we use Independent samples t-test")
    print("="*50)

if __name__ == "__main__":
    main()