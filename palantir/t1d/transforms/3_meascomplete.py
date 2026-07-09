from pyspark.sql import functions as F
from pyspark.sql import Window
from transforms.api import transform, Input, Output


@transform(
    output_dataset=Output("ri.foundry.main.dataset.xxxxx"),
    measurement_df=Input("ri.foundry.main.dataset.xxxxx"), # Output from step 1
    original_df=Input("ri.foundry.main.dataset.xxxxx")
)
def compute(measurement_df, original_df, output_dataset):
    # Step 1: Load the dataframes
    measurements = measurement_df.dataframe()
    original = original_df.dataframe()
    
    # Step 2: Ensure person_id columns are compatible and select necessary columns
    # Rename person_id to PERSON_ID to match measurement_df
    original = original.withColumn("PERSON_ID", F.col("person_id").cast("long"))
    original = original.drop("person_id")  # Drop the original lowercase column
    
    # Convert DiagnosisDate to timestamp for date arithmetic
    original = original.withColumn(
        "DiagnosisDateTime",
        F.to_timestamp(F.col("DiagnosisDate"))
    )
    
    # Step 3: Prepare measurements dataframe
    measurements = measurements.select(
        F.col("PERSON_ID").cast("long").alias("PERSON_ID"),
        "MEASUREMENT_DATETIME",
        "measurement_type",
        "VALUE_SOURCE_VALUE",
        "UNIT_SOURCE_VALUE"
    )
    
    # Step 4: Join measurements with original to get diagnosis date
    measurements_with_diagnosis = measurements.join(
        original.select("PERSON_ID", "DiagnosisDateTime"),
        on="PERSON_ID",
        how="inner"
    )
    
    # Step 5: Calculate time difference between measurement and diagnosis
    measurements_with_diagnosis = measurements_with_diagnosis.withColumn(
        "days_from_diagnosis",
        F.datediff(F.col("MEASUREMENT_DATETIME"), F.col("DiagnosisDateTime"))
    )
    
    # Step 6: Filter valid measurements for specific types and standardize units
    
    # Helper function: Remove < or > prefix and check if the remaining value is numeric
    def get_cleaned_value(value_col):
        """Remove < or > prefix and whitespace from value"""
        return F.regexp_replace(F.trim(value_col), "^[<>]\\s*", "")
    
    # Create validation column
    measurements_with_diagnosis = measurements_with_diagnosis.withColumn(
        "cleaned_value",
        get_cleaned_value(F.col("VALUE_SOURCE_VALUE"))
    )
    
    # SPECIAL PROCESSING FOR BICARBONATE: Split into BICARBONATE and PCO2
    measurements_with_diagnosis = measurements_with_diagnosis.withColumn(
        "measurement_type",
        F.when(
            (F.col("measurement_type") == "BICARBONATE") &
            (F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["mmhg", "mm hg"]) | 
             (F.trim(F.col("UNIT_SOURCE_VALUE")) == "MMHG")),
            F.lit("PCO2")  # Rename to PCO2 if unit is MMHG
        ).otherwise(F.col("measurement_type"))
    )
    
    # SPECIAL PROCESSING FOR URINE_KETONE: Binarize values
    measurements_with_diagnosis = measurements_with_diagnosis.withColumn(
        "VALUE_SOURCE_VALUE",
        F.when(
            F.col("measurement_type") == "URINE_KETONE",
            F.when(
                F.lower(F.trim(F.col("VALUE_SOURCE_VALUE"))).isin([
                    "negative", "neg", "none", "0"
                ]),
                F.lit("negative")
            ).otherwise(F.lit("non-negative"))  # trace, 1+, 2+, 3+, 4+, etc.
        ).otherwise(F.col("VALUE_SOURCE_VALUE"))
    )
    
    # Update cleaned_value after VALUE_SOURCE_VALUE transformation
    measurements_with_diagnosis = measurements_with_diagnosis.withColumn(
        "cleaned_value",
        get_cleaned_value(F.col("VALUE_SOURCE_VALUE"))
    )
    
    # UNIT CONVERSIONS AND STANDARDIZATION
    # Convert values to standard units where needed
    measurements_with_diagnosis = measurements_with_diagnosis.withColumn(
        "standardized_value",
        F.when(
            # HDL Cholesterol: convert nmol/L to mg/dL (divide by 0.0259)
            (F.col("measurement_type") == "HDL_CHOLESTEROL") &
            (F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))) == "nmol/l"),
            (F.col("cleaned_value").cast("double") / 0.0259).cast("string")
        ).when(
            # LDL Cholesterol: convert nmol/L to mg/dL (divide by 0.0259)
            (F.col("measurement_type") == "LDL_CHOLESTEROL") &
            (F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))) == "nmol/l"),
            (F.col("cleaned_value").cast("double") / 0.0259).cast("string")
        ).when(
            # Total Cholesterol: convert from non-standard units
            (F.col("measurement_type") == "TOTAL_CHOLESTEROL") &
            (F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))) == "mg/dl"),
            F.col("VALUE_SOURCE_VALUE")  # Already in correct unit
        ).when(
            # Serum C-Peptide: convert pg/mL to ng/mL (divide by 1000)
            (F.col("measurement_type") == "SERUM_C_PEPTIDE") &
            (F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))) == "pg/ml"),
            (F.col("cleaned_value").cast("double") / 1000.0).cast("string")
        ).otherwise(
            F.col("VALUE_SOURCE_VALUE")  # Keep original value for other cases
        )
    )
    
    # Update cleaned_value to use standardized_value
    measurements_with_diagnosis = measurements_with_diagnosis.withColumn(
        "cleaned_value",
        get_cleaned_value(F.col("standardized_value"))
    )
    
    # Standardize units after conversion
    measurements_with_diagnosis = measurements_with_diagnosis.withColumn(
        "standardized_unit",
        F.when(
            F.col("measurement_type") == "HBA1C",
            F.lit("%")
        ).when(
            F.col("measurement_type") == "GLUCOSE",
            F.lit("mg/dL")
        ).when(
            F.col("measurement_type").isin(["HDL_CHOLESTEROL", "LDL_CHOLESTEROL", "TOTAL_CHOLESTEROL"]),
            F.lit("mg/dL")
        ).when(
            F.col("measurement_type") == "TRIGLYCERIDES",
            F.lit("mg/dL")
        ).when(
            F.col("measurement_type").isin(["AST", "ALT"]),
            F.lit("U/L")
        ).when(
            F.col("measurement_type") == "BUN",
            F.lit("mg/dL")
        ).when(
            F.col("measurement_type") == "SERUM_CREATININE",
            F.lit("mg/dL")
        ).when(
            F.col("measurement_type") == "SERUM_CYSTATIN_C",
            F.lit("mg/L")
        ).when(
            F.col("measurement_type") == "URINE_CREATININE",
            F.lit("mg/dL")
        ).when(
            F.col("measurement_type") == "URINE_MICROALBUMIN",
            F.lit("mg/dL")
        ).when(
            F.col("measurement_type") == "BICARBONATE",
            F.lit("mmol/L")
        ).when(
            F.col("measurement_type") == "PCO2",
            F.lit("mmHg")
        ).when(
            F.col("measurement_type") == "SERUM_C_PEPTIDE",
            F.lit("ng/mL")
        ).when(
            F.col("measurement_type") == "EGFR",
            F.lit("mL/min/1.73m2")
        ).when(
            F.col("measurement_type").isin(["GAD65_ANTIBODY", "ICA512_ANTIBODY", "INSULIN_ANTIBODY", "ZNT8_ANTIBODY"]),
            F.lit("U/mL")
        ).when(
            F.col("measurement_type") == "HEIGHT",
            F.lit("[in_us]")
        ).when(
            F.col("measurement_type") == "WEIGHT",
            F.lit("lb")
        ).otherwise(
            F.col("UNIT_SOURCE_VALUE")  # Keep original for others
        )
    )
    
    measurements_with_diagnosis = measurements_with_diagnosis.withColumn(
        "is_valid_measurement",
        F.when(
            F.col("measurement_type") == "HBA1C",
            # A1C should be numeric (with optional < or >) and within reasonable range (3-20)
            # Only keep % units
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (F.col("cleaned_value").cast("double") >= 3.0) &
            (F.col("cleaned_value").cast("double") <= 20.0) &
            (F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["%", "% of total hgb"]) |
             (F.trim(F.col("UNIT_SOURCE_VALUE")) == "%"))
        ).when(
            F.col("measurement_type") == "GLUCOSE",
            # Glucose should be numeric (with optional < or >) and within reasonable range (20-600 mg/dl)
            # Only keep values with mg/dl or MG/DL units
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (F.col("cleaned_value").cast("double") >= 20.0) &
            (F.col("cleaned_value").cast("double") <= 600.0) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["mg/dl", "mg / dl", "mg/ dl", "mg /dl"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "MG/DL")
            )
        ).when(
            F.col("measurement_type").isin(["GAD65_ANTIBODY", "ICA512_ANTIBODY", "INSULIN_ANTIBODY", "ZNT8_ANTIBODY"]),
            # Antibody measurements: numeric values with optional < or >, only U/mL or IU/mL
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["u/ml", "iu/ml", "uu/ml"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")).isin(["U/mL", "IU/mL", "uU/mL"]))
            )
        ).when(
            F.col("measurement_type").isin(["AST", "ALT"]),
            # AST and ALT: only keep measurements with U/L units
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["u/l"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "U/L")
            )
        ).when(
            F.col("measurement_type") == "BUN",
            # BUN: only keep measurements with mg/dL or MG/DL units
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["mg/dl", "mg / dl", "mg/ dl", "mg /dl"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "MG/DL")
            )
        ).when(
            F.col("measurement_type") == "SERUM_CREATININE",
            # Serum creatinine: only keep measurements with MG/DL or mg/dL units
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["mg/dl", "mg / dl", "mg/ dl", "mg /dl"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "MG/DL")
            )
        ).when(
            F.col("measurement_type") == "SERUM_CYSTATIN_C",
            # Cystatin C: only keep measurements with MG/L or mg/L units
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["mg/l", "mg / l", "mg/ l", "mg /l"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "MG/L")
            )
        ).when(
            F.col("measurement_type") == "URINE_CREATININE",
            # Urine creatinine: keep only MG/DL and mg/dL (concentration units)
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["mg/dl", "mg / dl", "mg/ dl", "mg /dl"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "MG/DL")
            )
        ).when(
            F.col("measurement_type") == "URINE_MICROALBUMIN",
            # Urine microalbumin: keep only MG/DL and mg/dL (concentration units)
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (F.col("cleaned_value").cast("double") >= 0.0) &
            (F.col("cleaned_value").cast("double") <= 10000.0) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["mg/dl", "mg / dl", "mg/ dl", "mg /dl"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "MG/DL")
            )
        ).when(
            F.col("measurement_type") == "URINE_KETONE",
            # Urine ketone: keep binarized values (already processed above)
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".")
        ).when(
            F.col("measurement_type") == "BLOOD_PH",
            # Blood pH: only numeric values (no < or > allowed for pH)
            # Note: NG/ML seems incorrect for pH, but keeping validation
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            # Ensure no < or > prefix for pH
            (F.regexp_replace(F.trim(F.col("VALUE_SOURCE_VALUE")), "^[<>]\\s*", "") == F.trim(F.col("VALUE_SOURCE_VALUE"))) &
            (F.col("VALUE_SOURCE_VALUE").cast("double").isNotNull()) &
            (F.col("VALUE_SOURCE_VALUE").cast("double") >= 6.0) &
            (F.col("VALUE_SOURCE_VALUE").cast("double") <= 8.0)
        ).when(
            F.col("measurement_type") == "BICARBONATE",
            # Bicarbonate: only MMOL/L or mmol/L units (PCO2 split out above)
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["mmol/l", "mmol / l"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "MMOL/L")
            )
        ).when(
            F.col("measurement_type") == "PCO2",
            # PCO2: only MMHG or mmHg units (split from BICARBONATE above)
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["mmhg", "mm hg"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "MMHG")
            )
        ).when(
            F.col("measurement_type") == "HEIGHT",
            # Height should be numeric and valid
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (F.col("cleaned_value").cast("double") > 0)
        ).when(
            F.col("measurement_type") == "WEIGHT",
            # Weight should be numeric, valid, and have a unit (we'll drop null units later)
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (F.col("cleaned_value").cast("double") > 0) &
            (F.col("UNIT_SOURCE_VALUE").isNotNull())
        ).when(
            F.col("measurement_type") == "HDL_CHOLESTEROL",
            # HDL: keep mg/dL and nmol/L (will convert nmol/L)
            (F.trim(F.col("standardized_value")) != "") &
            (F.col("standardized_value") != ".") &
            (F.col("standardized_value").isNotNull()) &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["mg/dl", "nmol/l"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "MG/DL")
            )
        ).when(
            F.col("measurement_type") == "LDL_CHOLESTEROL",
            # LDL: keep mg/dL and nmol/L (will convert nmol/L), exclude other units
            (F.trim(F.col("standardized_value")) != "") &
            (F.col("standardized_value") != ".") &
            (F.col("standardized_value").isNotNull()) &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["mg/dl", "mg/dl (calc)", "nmol/l"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "MG/DL")
            )
        ).when(
            F.col("measurement_type") == "TOTAL_CHOLESTEROL",
            # Total cholesterol: keep mg/dL variants only
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["mg/dl", "mg/dl"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "MG/DL")
            )
        ).when(
            F.col("measurement_type") == "TRIGLYCERIDES",
            # Triglycerides: keep mg/dL variants only
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["mg/dl", "mg/dl (calc)"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "MG/DL")
            )
        ).when(
            F.col("measurement_type") == "EGFR",
            # eGFR: only keep mL/min/1.73m2 units
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))) == "ml/min/1.73m2")
        ).when(
            F.col("measurement_type") == "SERUM_C_PEPTIDE",
            # Serum C-peptide: keep ng/mL and pg/mL (will convert pg/mL)
            (F.trim(F.col("standardized_value")) != "") &
            (F.col("standardized_value") != ".") &
            (F.col("standardized_value").isNotNull()) &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["ng/ml", "pg/ml"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")).isin(["NG/ML", "ng/mL", "pg/mL"]))
            )
        ).otherwise(True)  # For other measurement types, keep all values
    )
    
    # Filter out invalid measurements
    measurements_filtered = measurements_with_diagnosis.filter(F.col("is_valid_measurement") == True)
    
    # Use standardized_value as the final value
    measurements_filtered = measurements_filtered.withColumn(
        "VALUE_SOURCE_VALUE",
        F.col("standardized_value")
    )
    
    # Use standardized_unit as the final unit
    measurements_filtered = measurements_filtered.withColumn(
        "UNIT_SOURCE_VALUE",
        F.col("standardized_unit")
    )
    
    # Drop the helper columns
    measurements_filtered = measurements_filtered.drop("cleaned_value", "is_valid_measurement", "standardized_value", "standardized_unit")
    
    # Step 7: Define time windows
    # At diagnosis: +/- 6 months (±180 days)
    # 2 years after: +/- 6 months around 2-year mark (730 ± 180 days)
    # 5 years after: +/- 6 months around 5-year mark (1825 ± 180 days)

    # Create flags for each time window
    measurements_with_windows = measurements_filtered.withColumn(
        "at_diagnosis",
        F.when(
            (F.col("days_from_diagnosis") >= -180) & (F.col("days_from_diagnosis") <= 180),
            F.abs(F.col("days_from_diagnosis"))
        ).otherwise(None)
    ).withColumn(
        "at_2_years",
        F.when(
            (F.col("days_from_diagnosis") >= 730 - 180) & (F.col("days_from_diagnosis") <= 730 + 180),
            F.abs(F.col("days_from_diagnosis") - 730)  # Distance from 2-year mark
        ).otherwise(None)
    ).withColumn(
        "at_5_years",
        F.when(
            (F.col("days_from_diagnosis") >= 1825 - 180) & (F.col("days_from_diagnosis") <= 1825 + 180),
            F.abs(F.col("days_from_diagnosis") - 1825)  # Distance from 5-year mark
        ).otherwise(None)
    )
    
    # Add numeric value column for outlier detection
    measurements_with_windows = measurements_with_windows.withColumn(
        "numeric_value",
        F.col("VALUE_SOURCE_VALUE").cast("double")
    )
    
    # NEW: Outlier detection and filtering
    # We'll apply outlier detection for numeric measurements within each time window
    # Using IQR method with a generous threshold (3.0 * IQR) to only remove extreme outliers
    
    # Define measurement types that should have outlier detection applied
    # Skip categorical measurements like URINE_KETONE
    numeric_measurement_types = [
        "HBA1C", "GLUCOSE", "HEIGHT", "WEIGHT", "TOTAL_CHOLESTEROL", "HDL_CHOLESTEROL",
        "LDL_CHOLESTEROL", "TRIGLYCERIDES", "SYSTOLIC_BLOOD_PRESSURE", "DIASTOLIC_BLOOD_PRESSURE",
        "SERUM_C_PEPTIDE", "ALT", "AST", "BUN", "SERUM_CREATININE", "EGFR",
        "SERUM_CYSTATIN_C", "URINE_MICROALBUMIN", "URINE_CREATININE", "GAD65_ANTIBODY",
        "ICA512_ANTIBODY", "INSULIN_ANTIBODY", "ZNT8_ANTIBODY", "BLOOD_PH", "BICARBONATE", "PCO2"
    ]
    
    # Process each time window separately for outlier detection
    def filter_outliers_in_window(df, window_col, window_name):
        """
        Filter outliers within a specific time window using IQR method.
        Only removes values that are extremely far from the IQR range (3.0 * IQR).
        """
        # Filter to measurements in this time window
        df_window = df.filter(F.col(window_col).isNotNull())
        
        # Calculate statistics per person and measurement type within the window
        window_stats = Window.partitionBy("PERSON_ID", "measurement_type")
        
        # Calculate Q1, Q3, median, and count for measurements in window
        df_window = df_window.withColumn(
            "q1",
            F.expr("percentile_approx(numeric_value, 0.25)").over(window_stats)
        ).withColumn(
            "q3",
            F.expr("percentile_approx(numeric_value, 0.75)").over(window_stats)
        ).withColumn(
            "median_val",
            F.expr("percentile_approx(numeric_value, 0.5)").over(window_stats)
        ).withColumn(
            "count_in_window",
            F.count("numeric_value").over(window_stats)
        )
        
        # Calculate IQR and outlier bounds
        df_window = df_window.withColumn(
            "iqr",
            F.col("q3") - F.col("q1")
        ).withColumn(
            "lower_bound",
            F.col("q1") - (3.0 * F.col("iqr"))  # 3.0 * IQR for very conservative filtering
        ).withColumn(
            "upper_bound",
            F.col("q3") + (3.0 * F.col("iqr"))
        )
        
        # Mark outliers - only for numeric measurement types and when we have multiple measurements
        df_window = df_window.withColumn(
            f"is_outlier_{window_name}",
            F.when(
                F.col("measurement_type").isin(numeric_measurement_types) &
                (F.col("count_in_window") >= 2) &  # Need at least 2 measurements to detect outliers
                (F.col("iqr") > 0) &  # IQR must be positive (measurements vary)
                (
                    (F.col("numeric_value") < F.col("lower_bound")) |
                    (F.col("numeric_value") > F.col("upper_bound"))
                ),
                True
            ).otherwise(False)
        )
        
        # Return non-outlier measurements
        return df_window.filter(F.col(f"is_outlier_{window_name}") == False).drop(
            "q1", "q3", "median_val", "count_in_window", "iqr", 
            "lower_bound", "upper_bound", f"is_outlier_{window_name}"
        )
    
    # Apply outlier filtering for each time window
    measurements_diagnosis_clean = filter_outliers_in_window(
        measurements_with_windows,
        "at_diagnosis",
        "diagnosis"
    )
    
    measurements_2years_clean = filter_outliers_in_window(
        measurements_with_windows,
        "at_2_years",
        "2years"
    )
    
    measurements_5years_clean = filter_outliers_in_window(
        measurements_with_windows,
        "at_5_years",
        "5years"
    )
    
    # Step 8: Get the measurement closest to each time point for each measurement type
    # For at_diagnosis: closest to diagnosis date (minimum absolute days)
    window_diagnosis = Window.partitionBy("PERSON_ID", "measurement_type").orderBy(
        F.col("at_diagnosis").asc_nulls_last()
    )
    
    # For 2 years: closest to 2-year mark
    window_2_years = Window.partitionBy("PERSON_ID", "measurement_type").orderBy(
        F.col("at_2_years").asc_nulls_last()
    )
    
    # For 5 years: closest to 5-year mark
    window_5_years = Window.partitionBy("PERSON_ID", "measurement_type").orderBy(
        F.col("at_5_years").asc_nulls_last()
    )
    
    # Get measurements at diagnosis (after outlier filtering)
    measurements_at_diagnosis = (
        measurements_diagnosis_clean
        .withColumn("rank", F.row_number().over(window_diagnosis))
        .filter(F.col("rank") == 1)
        .select(
            "PERSON_ID",
            "measurement_type",
            F.col("VALUE_SOURCE_VALUE").alias("value_at_diagnosis"),
            F.col("UNIT_SOURCE_VALUE").alias("unit_at_diagnosis"),
            F.col("MEASUREMENT_DATETIME").alias("datetime_at_diagnosis")
        )
    )
    
    # Get measurements at 2 years (after outlier filtering)
    measurements_at_2_years = (
        measurements_2years_clean
        .withColumn("rank", F.row_number().over(window_2_years))
        .filter(F.col("rank") == 1)
        .select(
            "PERSON_ID",
            "measurement_type",
            F.col("VALUE_SOURCE_VALUE").alias("value_at_2_years"),
            F.col("UNIT_SOURCE_VALUE").alias("unit_at_2_years"),
            F.col("MEASUREMENT_DATETIME").alias("datetime_at_2_years")
        )
    )
    
    # Get measurements at 5 years (after outlier filtering)
    measurements_at_5_years = (
        measurements_5years_clean
        .withColumn("rank", F.row_number().over(window_5_years))
        .filter(F.col("rank") == 1)
        .select(
            "PERSON_ID",
            "measurement_type",
            F.col("VALUE_SOURCE_VALUE").alias("value_at_5_years"),
            F.col("UNIT_SOURCE_VALUE").alias("unit_at_5_years"),
            F.col("MEASUREMENT_DATETIME").alias("datetime_at_5_years")
        )
    )
    
    # Step 9: Pivot to create columns for each measurement type at each time point
    # Get list of unique measurement types 
    # REMOVED: BMI, BMI_PERCENTILE, WAIST_CIRCUMFERENCE, URINE_C_PEPTIDE, URINE_MICROALBUMIN_CREATININE_RATIO
    # ADDED: PCO2 as separate measurement
    measurement_types = [
        "HBA1C",
        "GLUCOSE",
        "HEIGHT",
        "WEIGHT",
        "TOTAL_CHOLESTEROL",
        "HDL_CHOLESTEROL",
        "LDL_CHOLESTEROL",
        "TRIGLYCERIDES",
        "SYSTOLIC_BLOOD_PRESSURE",
        "DIASTOLIC_BLOOD_PRESSURE",
        "SERUM_C_PEPTIDE",
        "ALT",
        "AST",
        "BUN",
        "SERUM_CREATININE",
        "EGFR",
        "SERUM_CYSTATIN_C",
        "URINE_MICROALBUMIN",
        "URINE_CREATININE",
        "GAD65_ANTIBODY",
        "ICA512_ANTIBODY",
        "INSULIN_ANTIBODY",
        "ZNT8_ANTIBODY",
        "URINE_KETONE",
        "BLOOD_PH",
        "BICARBONATE",
        "PCO2"
    ]
    
    # Combine all measurements per person/type
    all_measurements = measurements_at_diagnosis.join(
        measurements_at_2_years,
        on=["PERSON_ID", "measurement_type"],
        how="outer"
    ).join(
        measurements_at_5_years,
        on=["PERSON_ID", "measurement_type"],
        how="outer"
    )
    
    # Step 10: Pivot the data to create columns
    # For each measurement type, create columns: [type]_at_diagnosis, [type]_at_2_years, [type]_at_5_years
    pivoted_measurements = all_measurements
    
    # Create aggregation expressions for pivot
    for time_point in ["diagnosis", "2_years", "5_years"]:
        for meas_type in measurement_types:
            col_name = f"{meas_type.lower()}_at_{time_point}"
            unit_col_name = f"{meas_type.lower()}_unit_at_{time_point}"
            datetime_col_name = f"{meas_type.lower()}_datetime_at_{time_point}"
            
            pivoted_measurements = pivoted_measurements.withColumn(
                col_name,
                F.when(F.col("measurement_type") == meas_type, F.col(f"value_at_{time_point}"))
            ).withColumn(
                unit_col_name,
                F.when(F.col("measurement_type") == meas_type, F.col(f"unit_at_{time_point}"))
            ).withColumn(
                datetime_col_name,
                F.when(F.col("measurement_type") == meas_type, F.col(f"datetime_at_{time_point}"))
            )
    
    # Group by PERSON_ID to consolidate rows
    agg_expressions = []
    for time_point in ["diagnosis", "2_years", "5_years"]:
        for meas_type in measurement_types:
            col_name = f"{meas_type.lower()}_at_{time_point}"
            unit_col_name = f"{meas_type.lower()}_unit_at_{time_point}"
            datetime_col_name = f"{meas_type.lower()}_datetime_at_{time_point}"
            
            agg_expressions.extend([
                F.max(col_name).alias(col_name),
                F.max(unit_col_name).alias(unit_col_name),
                F.max(datetime_col_name).alias(datetime_col_name)
            ])
    
    final_measurements = pivoted_measurements.groupBy("PERSON_ID").agg(*agg_expressions)
    
    # Step 11: Calculate BMI
    # Height is already in inches, weight is already in lb
    # BMI = (weight_lb / height_in^2) * 703
    
    for time_point in ["diagnosis", "2_years", "5_years"]:
        height_col = f"height_at_{time_point}"
        weight_col = f"weight_at_{time_point}"
        bmi_col = f"bmi_at_{time_point}"
        
        # Calculate BMI: BMI = (weight_lb / height_in^2) * 703
        final_measurements = final_measurements.withColumn(
            bmi_col,
            F.when(
                F.col(height_col).isNotNull() & 
                F.col(weight_col).isNotNull() & 
                (F.col(height_col).cast("double") > 0),
                (F.col(weight_col).cast("double") / (F.col(height_col).cast("double") * F.col(height_col).cast("double"))) * 703.0
            ).otherwise(None)
        )
        
        # Add unit and datetime columns for BMI
        final_measurements = final_measurements.withColumn(
            f"bmi_unit_at_{time_point}",
            F.when(F.col(bmi_col).isNotNull(), F.lit("kg/m^2")).otherwise(None)
        )
        # Use the weight datetime as the BMI datetime (since both height and weight are needed)
        final_measurements = final_measurements.withColumn(
            f"bmi_datetime_at_{time_point}",
            F.col(f"weight_datetime_at_{time_point}")
        )
    
    # Step 12: Calculate Urine Microalbumin/Creatinine Ratio
    # Both urine_microalbumin and urine_creatinine are in mg/dL
    # Ratio = (microalbumin_mg/dL / creatinine_mg/dL) * 1000 = mg/g
    # This converts from concentration ratio to the standard clinical unit
    
    for time_point in ["diagnosis", "2_years", "5_years"]:
        microalbumin_col = f"urine_microalbumin_at_{time_point}"
        creatinine_col = f"urine_creatinine_at_{time_point}"
        ratio_col = f"urine_microalbumin_creatinine_ratio_at_{time_point}"
        
        # Calculate ratio: (microalbumin / creatinine) * 1000 to get mg/g
        final_measurements = final_measurements.withColumn(
            ratio_col,
            F.when(
                F.col(microalbumin_col).isNotNull() & 
                F.col(creatinine_col).isNotNull() & 
                (F.col(creatinine_col).cast("double") > 0),
                (F.col(microalbumin_col).cast("double") / F.col(creatinine_col).cast("double")) * 1000.0
            ).otherwise(None)
        )
        
        # Add unit and datetime columns for ratio
        final_measurements = final_measurements.withColumn(
            f"urine_microalbumin_creatinine_ratio_unit_at_{time_point}",
            F.when(F.col(ratio_col).isNotNull(), F.lit("mg/g")).otherwise(None)
        )
        # Use the microalbumin datetime as the ratio datetime
        final_measurements = final_measurements.withColumn(
            f"urine_microalbumin_creatinine_ratio_datetime_at_{time_point}",
            F.col(f"urine_microalbumin_datetime_at_{time_point}")
        )
    
    # Step 13: Join back to original dataframe
    result = original.join(
        final_measurements,
        on="PERSON_ID",
        how="left"
    )
    
    # Write the result
    output_dataset.write_dataframe(result)