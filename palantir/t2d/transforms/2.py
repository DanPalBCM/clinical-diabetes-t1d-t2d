"""
Palantir Foundry PySpark Transform 2: T2D Patient Diagnosis Conditions - FIXED
Purpose: Add diagnosis condition flags at multiple timepoints using OMOP condition_occurrence
"""

from pyspark.sql import functions as F
from pyspark.sql import Window
from transforms.api import transform_df, Input, Output


# Define condition codes as module-level constant
CONDITION_CODES = {
    'DKA': ['250.11', '250.13', '250.10', '250.12', 'E10.10', 'E10.11', 'E11.10', 'E11.11'],
    'Ketosis': ['276.2', '790.6', 'E87.2'],
    'Dyslipidemia': ['272.', 'E78.0', 'E78.1', 'E78.2', 'E78.3', 'E78.4', 'E78.5', 'E78.6'],
    'Hypertension': ['401.', '402.', '403.', '404.', '405.', 'I10.', 'I11.', 'I12.', 'I13.', 
                     'I15.', 'H35.03', 'I67.4'],
    'Diabetic_Retinopathy': ['362.01', '362.02', '362.03', '362.04', '362.05', '362.06',
                             'E08.35', 'E09.35', 'E10.35', 'E11.35', 'E13.35', 'E08.31', 'E08.37',
                             'E09.31', 'E09.37', 'E10.31', 'E10.37', 'E11.31', 'E11.37', 'E13.31',
                             'E13.37', 'E08.32', 'E09.32', 'E10.32', 'E11.32', 'E13.32', 'E08.33',
                             'E09.33', 'E10.33', 'E11.33', 'E13.33', 'E08.34', 'E09.34', 'E10.34',
                             'E11.34', 'E13.34'],
    'Microalbuminuria': ['791.0', 'R80.9'],
    'Neuropathy': ['250.61', '250.63', '250.60', '250.62', '357.2', 'E10.40', 'E10.41', 
                   'E10.42', 'E10.43', 'E10.44', 'E10.49', 'E11.40', 'E11.41', 'E11.42', 'E11.45'],
    'Hypoglycemia': ['250.3', '250.8', '251.0', '251.1', '251.2', '270.3', '775.0', '775.6', 
                     '962.39', 'E08.641', 'E08.649', 'E09.641', 'E09.649', 'E10.641', 'E10.649',
                     'E11.641', 'E11.649', 'E13.641', 'E13.649', 'E15', 'E16.0', 'E16.1', 'E16.2',
                     'T38.3X1A', 'T38.3X1D', 'T38.3X1S', 'T38.3X2A', 'T38.3X2D', 'T38.3X2S',
                     'T38.3X3A', 'T38.3X3D', 'T38.3X3S', 'T38.3X4A', 'T38.3X4D', 'T38.3X4S',
                     'T38.3X5A', 'T38.3X5D', 'T38.3X5S']
}


@transform_df(
    Output("ri.foundry.main.dataset.xxxxx"),
    patient_demographics=Input("ri.foundry.main.dataset.xxxxx"),
    condition_occurrence=Input("ri.foundry.main.dataset.xxxxx")
)
def compute(patient_demographics, condition_occurrence):
    """
    Add diagnosis condition indicators at diagnosis, 2yr, and 5yr timepoints
    """
    
    # ============================================================================
    # STEP 1: Create target dates from DiagnosisDate if they don't exist
    # ============================================================================
    patient_df = patient_demographics
    
    # ============================================================================
    # STEP 2: Prepare condition data - extract ICD codes SAFELY
    # ============================================================================
    # Get unique patient list for filtering - USE OMOP_ID to match PERSON_ID
    patient_list = patient_df.select(
        F.col("mrn").cast("string"),
        F.col("OMOP_ID").cast("string")
    ).distinct()
    
    # Process condition_occurrence table with SAFE extraction
    # Handle both pipe-separated and plain formats
    conditions_processed = condition_occurrence.select(
        F.col("PERSON_ID").cast("string").alias("OMOP_ID"),  # Match with OMOP_ID
        F.to_date(F.col("CONDITION_START_DATE")).alias("condition_date"),
        # SAFE extraction: if pipe exists, take second part, otherwise use whole value
        F.when(
            F.col("CONDITION_SOURCE_VALUE").contains("|"),
            F.trim(F.element_at(F.split(F.col("CONDITION_SOURCE_VALUE"), "\\|"), 2))
        ).otherwise(
            F.trim(F.col("CONDITION_SOURCE_VALUE"))
        ).alias("icd_code")
    ).filter(
        (F.col("OMOP_ID").isNotNull()) & 
        (F.col("icd_code").isNotNull()) &
        (F.col("icd_code") != "")
    ).join(
        patient_list,
        on="OMOP_ID",  # Join on OMOP_ID instead of mrn
        how="inner"
    )
    
    # ============================================================================
    # STEP 3: Create condition flags - FIXED MATCHING LOGIC
    # ============================================================================
    # For each condition, check BOTH exact match AND prefix match for ALL codes
    # This matches the Python code logic
    for condition_name, codes in CONDITION_CODES.items():
        # Build condition filter with BOTH exact and prefix matching
        condition_filter = None
        for code in codes:
            # Check both exact match AND prefix match for every code
            # This is what the Python code does!
            exact_match = (F.col("icd_code") == code)
            prefix_match = F.col("icd_code").startswith(code)
            cond = exact_match | prefix_match
            
            condition_filter = cond if condition_filter is None else (condition_filter | cond)
        
        # Add flag column
        conditions_processed = conditions_processed.withColumn(
            condition_name,
            F.when(condition_filter, 1).otherwise(0)
        )
    
    # ============================================================================
    # STEP 4: Process each timepoint efficiently
    # ============================================================================
    def get_conditions_at_timepoint(patient_df, condition_df, timepoint_name, target_date_col):
        """
        Extract condition indicators at specific timepoint
        """
        # Get patient dates
        patient_dates = patient_df.select(
            "mrn",
            F.to_date(F.col(target_date_col)).alias("target_date")
        ).filter(F.col("target_date").isNotNull())
        
        # Join conditions with patient target dates
        conditions_with_target = patient_dates.join(
            condition_df,
            on="mrn",
            how="inner"
        )
        
        # Filter within +/- 6 months (180 days) window
        conditions_filtered = conditions_with_target.filter(
            F.abs(F.datediff(F.col("condition_date"), F.col("target_date"))) <= 180
        )
        
        # Aggregate: take MAX of each condition flag per patient
        condition_cols = list(CONDITION_CODES.keys())
        agg_exprs = [F.max(F.col(cond)).alias(f"{cond}_{timepoint_name}") 
                     for cond in condition_cols]
        
        conditions_agg = conditions_filtered.groupBy("mrn").agg(*agg_exprs)
        
        # Join back to all patients and fill nulls with 0
        result_df = patient_df.select("mrn").join(
            conditions_agg,
            on="mrn",
            how="left"
        )
        
        # Fill nulls with 0 (no condition present)
        fill_cols = [f"{cond}_{timepoint_name}" for cond in condition_cols]
        result_df = result_df.fillna(0, subset=fill_cols)
        
        return result_df
    
    # Process all three timepoints
    conditions_diagnosis = get_conditions_at_timepoint(
        patient_df, 
        conditions_processed, 
        "diagnosis", 
        "target_date_diagnosis"
    )
    
    conditions_2yr = get_conditions_at_timepoint(
        patient_df, 
        conditions_processed, 
        "2yr", 
        "target_date_2yr"
    )
    
    conditions_5yr = get_conditions_at_timepoint(
        patient_df, 
        conditions_processed, 
        "5yr", 
        "target_date_5yr"
    )
    
    # ============================================================================
    # STEP 5: Join everything together
    # ============================================================================
    final_df = patient_df.join(
        conditions_diagnosis, on="mrn", how="left"
    ).join(
        conditions_2yr, on="mrn", how="left"
    ).join(
        conditions_5yr, on="mrn", how="left"
    )
    
    return final_df.orderBy("mrn")