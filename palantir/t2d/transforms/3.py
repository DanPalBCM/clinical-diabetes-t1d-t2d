"""
Palantir Foundry PySpark Transform 3: T2D Patient A1C Measurements
Purpose: Add A1C measurements at multiple timepoints to complete dataset
"""

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from transforms.api import transform_df, Input, Output


@transform_df(
    Output("ri.foundry.main.dataset.xxxxx"),
    patient_with_conditions=Input("ri.foundry.main.dataset.xxxxx"),
    a1c_measurements=Input("ri.foundry.main.dataset.xxxxx")
)
def compute(patient_with_conditions, a1c_measurements):
    """
    Add A1C measurements at diagnosis, 2yr, and 5yr timepoints
    """
    
    # ============================================================================
    # STEP 1: Prepare A1C data ONCE - clean and filter
    # ============================================================================
    a1c_clean = a1c_measurements.select(
        F.col("mrn").cast("string").alias("mrn"),
        F.col("order_value"),
        F.col("reference_unit"),
        F.col("result_time").alias("a1c_date")
    ).filter(
        # Only keep measurements in % units
        (F.col("reference_unit") == "%") | 
        (F.col("reference_unit") == "% of total Hgb")
    )
    
    # Extract numeric values, handle "<14" cases
    a1c_clean = a1c_clean.withColumn(
        "a1c_numeric",
        F.when(
            F.col("order_value").startswith("<"),
            F.regexp_extract(F.col("order_value"), r"<(\d+\.?\d*)", 1).cast("double")
        ).otherwise(
            F.regexp_extract(F.col("order_value"), r"(\d+\.?\d*)", 1).cast("double")
        )
    )
    
    # Filter valid measurements (> 0 and <= 101)
    a1c_clean = a1c_clean.filter(
        (F.col("a1c_numeric") > 0) & 
        (F.col("a1c_numeric") <= 101)
    ).select(
        "mrn",
        "a1c_date",
        "a1c_numeric"
    )
    
    # ============================================================================
    # STEP 2: Process each timepoint efficiently
    # ============================================================================
    def get_a1c_at_timepoint(patient_df, a1c_df, timepoint_name, target_date_col):
        """
        Extract A1C measurement closest to timepoint within +/- 6 months window
        Uses pre-calculated target dates from patient_df
        """
        # Join A1C with patient target dates
        a1c_with_target = patient_df.select(
            "mrn",
            F.col(target_date_col).alias("target_date")
        ).join(
            a1c_df,
            on="mrn",
            how="left"
        )
        
        # Calculate days difference and filter within +/- 6 months (180 days)
        a1c_filtered = a1c_with_target.withColumn(
            "days_diff",
            F.abs(F.datediff(F.col("a1c_date"), F.col("target_date")))
        ).filter(
            F.col("days_diff") <= 180
        )
        
        # Get closest measurement for each patient using window function
        window_spec = Window.partitionBy("mrn").orderBy("days_diff")
        
        a1c_closest = a1c_filtered.withColumn(
            "rank",
            F.row_number().over(window_spec)
        ).filter(
            F.col("rank") == 1
        ).select(
            "mrn",
            F.col("a1c_numeric").alias(f"a1c_{timepoint_name}")
        )
        
        return a1c_closest
    
    # Process all three timepoints
    a1c_diagnosis = get_a1c_at_timepoint(
        patient_with_conditions, 
        a1c_clean, 
        "diagnosis", 
        "target_date_diagnosis"
    )
    
    a1c_2yr = get_a1c_at_timepoint(
        patient_with_conditions, 
        a1c_clean, 
        "2yr", 
        "target_date_2yr"
    )
    
    a1c_5yr = get_a1c_at_timepoint(
        patient_with_conditions, 
        a1c_clean, 
        "5yr", 
        "target_date_5yr"
    )
    
    # ============================================================================
    # STEP 3: Join A1C measurements to patient data
    # ============================================================================
    final_df = patient_with_conditions.join(
        a1c_diagnosis, on="mrn", how="left"
    ).join(
        a1c_2yr, on="mrn", how="left"
    ).join(
        a1c_5yr, on="mrn", how="left"
    )
    
    # ============================================================================
    # STEP 4: Order by MRN (preserving all input columns + new A1C columns)
    # ============================================================================
    # Simply order by mrn - all columns from patient_with_conditions are preserved
    # plus the three new A1C columns we added
    final_df = final_df.orderBy("mrn")
    
    return final_df