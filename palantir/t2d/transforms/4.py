"""
Palantir Foundry PySpark Transform: T2D Patient Medication Exposure
Purpose: Add medication class flags at multiple timepoints using OMOP drug_exposure
"""

from pyspark.sql import functions as F
from pyspark.sql import Window
from transforms.api import transform_df, Input, Output

# Define medication classes as module-level constant
DRUG_CLASSES = {
    'Insulins': [
        'insulin aspart', 'insulin degludec', 'insulin detemir', 
        'insulin glargine', 'insulin glulisine', 'insulin human', 
        'insulin regular', 'insulin nph', 'insulin isophane', 
        'insulin lispro', 'insulin lispro protamine', 
        'inhaled human insulin', 'technosphere insulin'
    ],
    'Biguanide': ['metformin'],
    'GLP1_agonists': [
        'albiglutide', 'dulaglutide', 'exenatide', 
        'liraglutide', 'lixisenatide', 'semaglutide', 'tirzepatide'
    ],
    'DPP4_inhibitors': [
        'alogliptin', 'anagliptin', 'evogliptin', 'gemigliptin', 
        'linagliptin', 'saxagliptin', 'sitagliptin', 
        'teneligliptin', 'vildagliptin'
    ],
    'SGLT2_inhibitors': [
        'canagliflozin', 'dapagliflozin', 'empagliflozin', 
        'ertugliflozin', 'ipragliflozin', 'luseogliflozin', 
        'remogliflozin', 'sotagliflozin', 'tofogliflozin'
    ],
    'Sulfonylureas': [
        'acetohexamide', 'chlorpropamide', 'glimepiride', 
        'glipizide', 'glyburide', 'glibenclamide', 
        'tolazamide', 'tolbutamide'
    ],
    'Meglitinides': ['nateglinide', 'repaglinide'],
    'Thiazolidinediones': ['lobeglitazone', 'pioglitazone', 'rosiglitazone'],
    'Alpha_glucosidase_inhibitors': ['acarbose', 'miglitol', 'voglibose'],
    'Amylin_analogue': ['pramlintide']
}

@transform_df(
    Output("ri.foundry.main.dataset.xxxxx"),
    patient_demographics=Input("ri.foundry.main.dataset.xxxxx"),
    drug_exposure=Input("ri.foundry.main.dataset.xxxxx")
)
def compute(patient_demographics, drug_exposure):
    """
    Add medication class indicators at diagnosis, 2yr, and 5yr timepoints
    """
    
    # ============================================================================
    # STEP 1: Prepare patient data
    # ============================================================================
    patient_df = patient_demographics
    
    # ============================================================================
    # STEP 2: Prepare medication data
    # ============================================================================
    # Get unique patient list for filtering - USE OMOP_ID to match PERSON_ID
    patient_list = patient_df.select(
        F.col("mrn").cast("string"),
        F.col("OMOP_ID").cast("string")
    ).distinct()
    
    # Process drug_exposure table
    # Convert drug name to lowercase for case-insensitive matching
    medications_processed = drug_exposure.select(
        F.col("PERSON_ID").cast("string").alias("OMOP_ID"),  # Match with OMOP_ID
        F.to_date(F.col("DRUG_EXPOSURE_START_DATETIME")).alias("medication_date"),
        F.lower(F.trim(F.col("DRUG_SOURCE_VALUE"))).alias("drug_name")
    ).filter(
        (F.col("OMOP_ID").isNotNull()) &
        (F.col("drug_name").isNotNull()) &
        (F.col("drug_name") != "")
    ).join(
        patient_list,
        on="OMOP_ID",  # Join on OMOP_ID instead of mrn
        how="inner"
    )
    
    # ============================================================================
    # STEP 3: Create medication class flags
    # ============================================================================
    # For each medication class, check if drug name contains any of the medications
    for class_name, medications in DRUG_CLASSES.items():
        # Build condition filter - check if drug_name contains any medication
        medication_filter = None
        for med in medications:
            # Case-insensitive substring match (contains)
            cond = F.col("drug_name").contains(med.lower())
            medication_filter = cond if medication_filter is None else (medication_filter | cond)
        
        # Add flag column
        medications_processed = medications_processed.withColumn(
            class_name,
            F.when(medication_filter, 1).otherwise(0)
        )
    
    # ============================================================================
    # STEP 4: Process each timepoint efficiently
    # ============================================================================
    def get_medications_at_timepoint(patient_df, medication_df, timepoint_name, target_date_col):
        """
        Extract medication indicators at specific timepoint
        """
        # Get patient dates
        patient_dates = patient_df.select(
            "mrn",
            F.to_date(F.col(target_date_col)).alias("target_date")
        ).filter(F.col("target_date").isNotNull())
        
        # Join medications with patient target dates
        medications_with_target = patient_dates.join(
            medication_df,
            on="mrn",
            how="inner"
        )
        
        # Filter within +/- 6 months (180 days) window
        medications_filtered = medications_with_target.filter(
            F.abs(F.datediff(F.col("medication_date"), F.col("target_date"))) <= 180
        )
        
        # Aggregate: take MAX of each medication class flag per patient
        medication_classes = list(DRUG_CLASSES.keys())
        agg_exprs = [F.max(F.col(med_class)).alias(f"{med_class}_{timepoint_name}") 
                     for med_class in medication_classes]
        
        medications_agg = medications_filtered.groupBy("mrn").agg(*agg_exprs)
        
        # Join back to all patients and fill nulls with 0
        result_df = patient_df.select("mrn").join(
            medications_agg,
            on="mrn",
            how="left"
        )
        
        # Fill nulls with 0 (no medication present)
        fill_cols = [f"{med_class}_{timepoint_name}" for med_class in medication_classes]
        result_df = result_df.fillna(0, subset=fill_cols)
        
        return result_df
    
    # Process all three timepoints
    medications_diagnosis = get_medications_at_timepoint(
        patient_df, medications_processed, "diagnosis", "target_date_diagnosis"
    )
    
    medications_2yr = get_medications_at_timepoint(
        patient_df, medications_processed, "2yr", "target_date_2yr"
    )
    
    medications_5yr = get_medications_at_timepoint(
        patient_df, medications_processed, "5yr", "target_date_5yr"
    )
    
    # ============================================================================
    # STEP 5: Join everything together
    # ============================================================================
    final_df = patient_df.join(
        medications_diagnosis, on="mrn", how="left"
    ).join(
        medications_2yr, on="mrn", how="left"
    ).join(
        medications_5yr, on="mrn", how="left"
    )
    
    return final_df.orderBy("mrn")