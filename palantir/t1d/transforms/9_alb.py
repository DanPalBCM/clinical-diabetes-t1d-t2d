from pyspark.sql import functions as F
from pyspark.sql.window import Window
from transforms.api import Input, Output, transform
from datetime import timedelta

@transform(
    t2d_input=Input("ri.foundry.main.dataset.xxxxx"),
    labs=Input("ri.foundry.main.dataset.xxxxx"),
    final_df=Output("ri.foundry.main.dataset.xxxxx")
)
def compute(t2d_input, labs, final_df):
    """
    Extract microalbumin/creatinine ratio measurements at diagnosis, 2 years, and 5 years.
    Includes consecutive measurements within time windows for accuracy confirmation.
    Searches for multiple name variations and calculates ratio from components if needed.
    """
    
    # Load dataframes
    t2d_df = t2d_input.dataframe()
    labs_df = labs.dataframe()
    
    # Define ratio name patterns (common variations found in EHR systems)
    ratio_patterns = [
        "MICROALB/CREAT URN POCT",
        "MICROALBUMIN/CREATININE",
        "ALBUMIN/CREATININE",
        "MICROALBUMIN CREATININE RATIO",
        "ALBUMIN CREATININE RATIO",
        "URINE ALBUMIN/CREATININE",
        "ACR",  # Common abbreviation
        "UACR",  # Urine Albumin-Creatinine Ratio
        "MA/CR",  # Microalbumin/Creatinine abbreviation
        "ALB/CREAT"
    ]
    
    # Create a pattern matching condition
    ratio_condition = None
    for pattern in ratio_patterns:
        condition = F.upper(F.col("clarity_component_name")).contains(pattern)
        ratio_condition = condition if ratio_condition is None else (ratio_condition | condition)
    
    # Filter for ratio measurements
    microalb_creat_ratio_df = labs_df.filter(ratio_condition).select(
        F.col("patient_id").alias("lab_patient_id"),
        F.col("result_date"),
        F.col("clarity_component_name"),
        F.coalesce(F.col("order_num_value"), F.col("order_value")).alias("ratio_value")
    )
    
    # Define microalbumin name patterns
    microalbumin_patterns = [
        "MICROALBUMIN",
        "ALBUMIN.*URINE",
        "URINE.*ALBUMIN",
        "MICROALB",
        "MA URINE"
    ]
    
    # Define creatinine name patterns
    creatinine_patterns = [
        "CREATININE.*URINE",
        "URINE.*CREATININE",
        "URINE CREAT",
        "UCR",
        "CR URINE"
    ]
    
    # Create pattern matching for microalbumin
    microalbumin_condition = None
    for pattern in microalbumin_patterns:
        condition = F.upper(F.col("clarity_component_name")).rlike(pattern)
        microalbumin_condition = condition if microalbumin_condition is None else (microalbumin_condition | condition)
    
    # Create pattern matching for creatinine (excluding ratio tests)
    creatinine_condition = None
    for pattern in creatinine_patterns:
        condition = F.upper(F.col("clarity_component_name")).rlike(pattern)
        creatinine_condition = condition if creatinine_condition is None else (creatinine_condition | condition)
    
    # Filter for individual component measurements
    microalbumin_df = labs_df.filter(
        microalbumin_condition & ~F.upper(F.col("clarity_component_name")).contains("CREAT")
    ).select(
        F.col("patient_id").alias("lab_patient_id"),
        F.col("result_date").alias("microalbumin_date"),
        F.coalesce(F.col("order_num_value"), F.col("order_value")).alias("microalbumin_value")
    )
    
    creatinine_df = labs_df.filter(
        creatinine_condition & ~F.upper(F.col("clarity_component_name")).contains("MICROALB") & ~F.upper(F.col("clarity_component_name")).contains("ALBUMIN")
    ).select(
        F.col("patient_id").alias("lab_patient_id"),
        F.col("result_date").alias("creatinine_date"),
        F.coalesce(F.col("order_num_value"), F.col("order_value")).alias("creatinine_value")
    )
    
    # Define time points and windows (in days)
    time_points = [
        {"name": "at_diagnosis", "years": 0, "window_days": 180},
        {"name": "at_2_years", "years": 2, "window_days": 180},
        {"name": "at_5_years", "years": 5, "window_days": 180}
    ]
    
    # Process each time point
    for tp in time_points:
        suffix = tp["name"]
        years_offset = tp["years"]
        window_days = tp["window_days"]
        
        # Create target date column (diagnosis date + years offset)
        t2d_df = t2d_df.withColumn(
            f"target_date_{suffix}",
            F.expr(f"date_add(DiagnosisDate, {years_offset * 365})")
        )
        
        # ===== PRIMARY APPROACH: Use pre-calculated ratio =====
        joined_ratio_df = t2d_df.alias("t2d").join(
            microalb_creat_ratio_df.alias("lab"),
            (F.col("t2d.PatientID") == F.col("lab.lab_patient_id")) &
            (F.col("lab.result_date").between(
                F.expr(f"date_sub(t2d.target_date_{suffix}, {window_days})"),
                F.expr(f"date_add(t2d.target_date_{suffix}, {window_days})")
            )),
            "left"
        )
        
        # Rank ratio measurements by proximity to target date
        window_spec_ratio = Window.partitionBy("t2d.PatientID", f"t2d.target_date_{suffix}").orderBy(
            F.abs(F.datediff(F.col("lab.result_date"), F.col(f"t2d.target_date_{suffix}"))),
            F.col("lab.result_date").desc()
        )
        
        joined_ratio_df = joined_ratio_df.withColumn("rank_ratio", F.row_number().over(window_spec_ratio))
        
        # Get primary ratio measurement
        primary_ratio = joined_ratio_df.filter(F.col("rank_ratio") == 1).select(
            F.col("t2d.PatientID"),
            F.col("lab.ratio_value").alias(f"ratio_from_direct_{suffix}"),
            F.col("lab.result_date").alias(f"ratio_date_{suffix}")
        )
        
        # Get consecutive ratio measurement
        consecutive_ratio = joined_ratio_df.filter(F.col("rank_ratio") == 2).select(
            F.col("t2d.PatientID"),
            F.col("lab.ratio_value").alias(f"consecutive_ratio_from_direct_{suffix}")
        )
        
        # ===== FALLBACK APPROACH: Calculate from components =====
        # Join microalbumin
        joined_microalbumin_df = t2d_df.alias("t2d").join(
            microalbumin_df.alias("ma"),
            (F.col("t2d.PatientID") == F.col("ma.lab_patient_id")) &
            (F.col("ma.microalbumin_date").between(
                F.expr(f"date_sub(t2d.target_date_{suffix}, {window_days})"),
                F.expr(f"date_add(t2d.target_date_{suffix}, {window_days})")
            )),
            "left"
        )
        
        window_spec_ma = Window.partitionBy("t2d.PatientID", f"t2d.target_date_{suffix}").orderBy(
            F.abs(F.datediff(F.col("ma.microalbumin_date"), F.col(f"t2d.target_date_{suffix}"))),
            F.col("ma.microalbumin_date").desc()
        )
        
        joined_microalbumin_df = joined_microalbumin_df.withColumn("rank_ma", F.row_number().over(window_spec_ma))
        
        primary_microalbumin = joined_microalbumin_df.filter(F.col("rank_ma") == 1).select(
            F.col("t2d.PatientID"),
            F.col("ma.microalbumin_value").alias(f"microalbumin_{suffix}"),
            F.col("ma.microalbumin_date").alias(f"microalbumin_date_{suffix}")
        )
        
        # Join creatinine
        joined_creatinine_df = t2d_df.alias("t2d").join(
            creatinine_df.alias("cr"),
            (F.col("t2d.PatientID") == F.col("cr.lab_patient_id")) &
            (F.col("cr.creatinine_date").between(
                F.expr(f"date_sub(t2d.target_date_{suffix}, {window_days})"),
                F.expr(f"date_add(t2d.target_date_{suffix}, {window_days})")
            )),
            "left"
        )
        
        window_spec_cr = Window.partitionBy("t2d.PatientID", f"t2d.target_date_{suffix}").orderBy(
            F.abs(F.datediff(F.col("cr.creatinine_date"), F.col(f"t2d.target_date_{suffix}"))),
            F.col("cr.creatinine_date").desc()
        )
        
        joined_creatinine_df = joined_creatinine_df.withColumn("rank_cr", F.row_number().over(window_spec_cr))
        
        primary_creatinine = joined_creatinine_df.filter(F.col("rank_cr") == 1).select(
            F.col("t2d.PatientID"),
            F.col("cr.creatinine_value").alias(f"creatinine_{suffix}"),
            F.col("cr.creatinine_date").alias(f"creatinine_date_{suffix}")
        )
        
        # Join all measurements back to main dataframe
        t2d_df = t2d_df.join(primary_ratio, "PatientID", "left")
        t2d_df = t2d_df.join(consecutive_ratio, "PatientID", "left")
        t2d_df = t2d_df.join(primary_microalbumin, "PatientID", "left")
        t2d_df = t2d_df.join(primary_creatinine, "PatientID", "left")
        
        # Calculate ratio from components when direct ratio is not available
        t2d_df = t2d_df.withColumn(
            f"ratio_from_components_{suffix}",
            F.when(
                (F.col(f"microalbumin_{suffix}").isNotNull()) & 
                (F.col(f"creatinine_{suffix}").isNotNull()) &
                (F.col(f"creatinine_{suffix}") != 0),
                F.col(f"microalbumin_{suffix}") / F.col(f"creatinine_{suffix}")
            ).otherwise(None)
        )
        
        # Create final ratio column - prefer direct ratio, fall back to calculated
        t2d_df = t2d_df.withColumn(
            f"microalbumin_creatinine_ratio_{suffix}",
            F.coalesce(
                F.col(f"ratio_from_direct_{suffix}"),
                F.col(f"ratio_from_components_{suffix}")
            )
        )
        
        # Create final date column
        t2d_df = t2d_df.withColumn(
            f"microalbumin_creatinine_ratio_date_{suffix}",
            F.coalesce(
                F.col(f"ratio_date_{suffix}"),
                F.col(f"microalbumin_date_{suffix}"),
                F.col(f"creatinine_date_{suffix}")
            )
        )
        
        # Create consecutive ratio (prefer direct, fallback not applicable for consecutive)
        t2d_df = t2d_df.withColumn(
            f"consecutive_microalbumin_creatinine_ratio_{suffix}",
            F.col(f"consecutive_ratio_from_direct_{suffix}")
        )
        
        # Add flag for data source
        t2d_df = t2d_df.withColumn(
            f"ratio_source_{suffix}",
            F.when(F.col(f"ratio_from_direct_{suffix}").isNotNull(), "direct")
             .when(F.col(f"ratio_from_components_{suffix}").isNotNull(), "calculated")
             .otherwise("not_available")
        )
        
        # Drop temporary columns
        temp_cols = [
            f"target_date_{suffix}",
            f"ratio_from_direct_{suffix}",
            f"ratio_from_components_{suffix}",
            f"ratio_date_{suffix}",
            f"consecutive_ratio_from_direct_{suffix}",
            f"microalbumin_{suffix}",
            f"microalbumin_date_{suffix}",
            f"creatinine_{suffix}",
            f"creatinine_date_{suffix}"
        ]
        t2d_df = t2d_df.drop(*temp_cols)
    
    # Remove old microalbumin and creatinine columns
    columns_to_keep = [
        col for col in t2d_df.columns 
        if not (col.startswith("urine_microalbumin_") or col.startswith("urine_creatinine"))
    ]
    
    t2d_df = t2d_df.select(*columns_to_keep)
    
    # Add Diabetes Duration column (in years)
    t2d_df = t2d_df.withColumn(
        "Diabetes_Duration",
        F.round(F.datediff(F.lit("2025-04-30"), F.col("DiagnosisDate")) / 365.25, 2)
    )
    
    # ===== ADD MICROALB/CREAT RATIO FLAG =====
    # Get the last 2 ratio measurements for each patient (ignoring time windows)
    # to determine if patient has consistently elevated ratios (both > 30)
    
    # Get all ratio measurements for all patients, ranked by date (most recent first)
    all_ratios_window = Window.partitionBy("lab_patient_id").orderBy(F.col("result_date").desc())
    
    microalb_creat_ratio_ranked = microalb_creat_ratio_df.withColumn(
        "ratio_rank",
        F.row_number().over(all_ratios_window)
    ).filter(
        F.col("ratio_rank").isin([1, 2])  # Get only the last 2 measurements
    )
    
    # Get the most recent measurement
    most_recent_ratio = microalb_creat_ratio_ranked.filter(F.col("ratio_rank") == 1).select(
        F.col("lab_patient_id").alias("patient_id_recent"),
        F.col("ratio_value").alias("most_recent_ratio")
    )
    
    # Get the second most recent measurement
    second_recent_ratio = microalb_creat_ratio_ranked.filter(F.col("ratio_rank") == 2).select(
        F.col("lab_patient_id").alias("patient_id_second"),
        F.col("ratio_value").alias("second_recent_ratio")
    )
    
    # Join both measurements to main dataframe
    t2d_df = t2d_df.join(
        most_recent_ratio,
        F.col("PatientID") == F.col("patient_id_recent"),
        "left"
    ).join(
        second_recent_ratio,
        F.col("PatientID") == F.col("patient_id_second"),
        "left"
    )
    
    # Create the flag: 1 if both measurements > 30, otherwise 0
    t2d_df = t2d_df.withColumn(
        "microalb_creat_ratio_FLAG",
        F.when(
            (F.col("most_recent_ratio").isNotNull()) & 
            (F.col("second_recent_ratio").isNotNull()) &
            (F.col("most_recent_ratio") > 30) & 
            (F.col("second_recent_ratio") > 30),
            1
        ).otherwise(0)
    )
    
    # Drop temporary columns
    t2d_df = t2d_df.drop("patient_id_recent", "patient_id_second", "most_recent_ratio", "second_recent_ratio")
    
    # Write output
    final_df.write_dataframe(t2d_df)