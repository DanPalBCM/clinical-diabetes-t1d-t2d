from pyspark.sql import functions as F
from pyspark.sql import Window
from transforms.api import transform, Input, Output


@transform(
    output_dataset=Output("ri.foundry.main.dataset.xxxxx"),
    measurement_df=Input("ri.foundry.main.dataset.xxxxx"),
    original_df=Input("ri.foundry.main.dataset.xxxxx")
)
def compute(measurement_df, original_df, output_dataset):
    # Step 1: Load the dataframes
    measurements = measurement_df.dataframe()
    original = original_df.dataframe()
    
    # Step 2: Ensure person_id columns are compatible and select necessary columns
    original = original.withColumn("PERSON_ID", F.col("OMOP_ID").cast("long"))
    original = original.drop("person_id")
    
    # Convert DiagnosisDate to timestamp for date arithmetic
    original = original.withColumn(
        "DiagnosisDateTime",
        F.to_timestamp(F.col("date_of_diagnosis"))
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

    def get_cleaned_value(value_col):
        return F.regexp_replace(F.trim(value_col), "^[<>]\\s*", "")
    
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
            F.lit("PCO2")
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
            ).otherwise(F.lit("non-negative"))
        ).otherwise(F.col("VALUE_SOURCE_VALUE"))
    )
    
    measurements_with_diagnosis = measurements_with_diagnosis.withColumn(
        "cleaned_value",
        get_cleaned_value(F.col("VALUE_SOURCE_VALUE"))
    )
    
    # UNIT CONVERSIONS AND STANDARDIZATION
    measurements_with_diagnosis = measurements_with_diagnosis.withColumn(
        "standardized_value",
        F.when(
            (F.col("measurement_type") == "HDL_CHOLESTEROL") &
            (F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))) == "nmol/l"),
            (F.col("cleaned_value").cast("double") / 0.0259).cast("string")
        ).when(
            (F.col("measurement_type") == "LDL_CHOLESTEROL") &
            (F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))) == "nmol/l"),
            (F.col("cleaned_value").cast("double") / 0.0259).cast("string")
        ).when(
            (F.col("measurement_type") == "TOTAL_CHOLESTEROL") &
            (F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))) == "mg/dl"),
            F.col("VALUE_SOURCE_VALUE")
        ).when(
            (F.col("measurement_type") == "SERUM_C_PEPTIDE") &
            (F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))) == "pg/ml"),
            (F.col("cleaned_value").cast("double") / 1000.0).cast("string")
        ).otherwise(
            F.col("VALUE_SOURCE_VALUE")
        )
    )
    
    measurements_with_diagnosis = measurements_with_diagnosis.withColumn(
        "cleaned_value",
        get_cleaned_value(F.col("standardized_value"))
    )
    
    # Standardize units
    measurements_with_diagnosis = measurements_with_diagnosis.withColumn(
        "standardized_unit",
        F.when(F.col("measurement_type") == "HBA1C", F.lit("%"))
        .when(F.col("measurement_type") == "GLUCOSE", F.lit("mg/dL"))
        .when(F.col("measurement_type").isin(["HDL_CHOLESTEROL", "LDL_CHOLESTEROL", "TOTAL_CHOLESTEROL"]), F.lit("mg/dL"))
        .when(F.col("measurement_type") == "TRIGLYCERIDES", F.lit("mg/dL"))
        .when(F.col("measurement_type").isin(["AST", "ALT"]), F.lit("U/L"))
        .when(F.col("measurement_type") == "BUN", F.lit("mg/dL"))
        .when(F.col("measurement_type") == "SERUM_CREATININE", F.lit("mg/dL"))
        .when(F.col("measurement_type") == "SERUM_CYSTATIN_C", F.lit("mg/L"))
        .when(F.col("measurement_type") == "URINE_CREATININE", F.lit("mg/dL"))
        .when(F.col("measurement_type") == "URINE_MICROALBUMIN", F.lit("mg/dL"))
        .when(F.col("measurement_type") == "BICARBONATE", F.lit("mmol/L"))
        .when(F.col("measurement_type") == "PCO2", F.lit("mmHg"))
        .when(F.col("measurement_type") == "SERUM_C_PEPTIDE", F.lit("ng/mL"))
        .when(F.col("measurement_type") == "EGFR", F.lit("mL/min/1.73m2"))
        .when(F.col("measurement_type").isin(["GAD65_ANTIBODY", "ICA512_ANTIBODY", "INSULIN_ANTIBODY", "ZNT8_ANTIBODY"]), F.lit("U/mL"))
        .when(F.col("measurement_type") == "HEIGHT", F.lit("[in_us]"))
        .when(F.col("measurement_type") == "WEIGHT", F.lit("oz_av"))
        # BP units will be assigned after splitting below
        .otherwise(F.col("UNIT_SOURCE_VALUE"))
    )
    
    measurements_with_diagnosis = measurements_with_diagnosis.withColumn(
        "is_valid_measurement",
        F.when(
            F.col("measurement_type") == "HBA1C",
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (F.col("cleaned_value").cast("double") >= 3.0) &
            (F.col("cleaned_value").cast("double") <= 20.0) &
            (F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["%", "% of total hgb"]) |
             (F.trim(F.col("UNIT_SOURCE_VALUE")) == "%"))
        ).when(
            F.col("measurement_type") == "GLUCOSE",
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
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["u/ml", "iu/ml", "uu/ml"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")).isin(["U/mL", "IU/mL", "uU/mL"]))
            )
        ).when(
            F.col("measurement_type").isin(["AST", "ALT"]),
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["u/l"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "U/L")
            )
        ).when(
            F.col("measurement_type") == "BUN",
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["mg/dl", "mg / dl", "mg/ dl", "mg /dl"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "MG/DL")
            )
        ).when(
            F.col("measurement_type") == "SERUM_CREATININE",
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["mg/dl", "mg / dl", "mg/ dl", "mg /dl"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "MG/DL")
            )
        ).when(
            F.col("measurement_type") == "SERUM_CYSTATIN_C",
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["mg/l", "mg / l", "mg/ l", "mg /l"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "MG/L")
            )
        ).when(
            F.col("measurement_type") == "URINE_CREATININE",
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["mg/dl", "mg / dl", "mg/ dl", "mg /dl"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "MG/DL")
            )
        ).when(
            F.col("measurement_type") == "URINE_MICROALBUMIN",
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
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".")
        ).when(
            F.col("measurement_type") == "BLOOD_PH",
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.regexp_replace(F.trim(F.col("VALUE_SOURCE_VALUE")), "^[<>]\\s*", "") == F.trim(F.col("VALUE_SOURCE_VALUE"))) &
            (F.col("VALUE_SOURCE_VALUE").cast("double").isNotNull()) &
            (F.col("VALUE_SOURCE_VALUE").cast("double") >= 6.0) &
            (F.col("VALUE_SOURCE_VALUE").cast("double") <= 8.0)
        ).when(
            F.col("measurement_type") == "BICARBONATE",
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["mmol/l", "mmol / l"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "MMOL/L")
            )
        ).when(
            F.col("measurement_type") == "PCO2",
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["mmhg", "mm hg"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "MMHG")
            )
        ).when(
            F.col("measurement_type") == "HEIGHT",
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (F.col("cleaned_value").cast("double") > 0)
        ).when(
            F.col("measurement_type") == "WEIGHT",
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (F.col("cleaned_value").cast("double") > 0) &
            (F.col("UNIT_SOURCE_VALUE").isNotNull())
        ).when(
            F.col("measurement_type") == "HDL_CHOLESTEROL",
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
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["mg/dl"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "MG/DL")
            )
        ).when(
            F.col("measurement_type") == "TRIGLYCERIDES",
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["mg/dl", "mg/dl (calc)"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")) == "MG/DL")
            )
        ).when(
            F.col("measurement_type") == "EGFR",
            (F.trim(F.col("VALUE_SOURCE_VALUE")) != "") &
            (F.col("VALUE_SOURCE_VALUE") != ".") &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))) == "ml/min/1.73m2")
        ).when(
            F.col("measurement_type") == "SERUM_C_PEPTIDE",
            (F.trim(F.col("standardized_value")) != "") &
            (F.col("standardized_value") != ".") &
            (F.col("standardized_value").isNotNull()) &
            (F.col("cleaned_value").cast("double").isNotNull()) &
            (
                F.lower(F.trim(F.col("UNIT_SOURCE_VALUE"))).isin(["ng/ml", "pg/ml"]) |
                (F.trim(F.col("UNIT_SOURCE_VALUE")).isin(["NG/ML", "ng/mL", "pg/mL"]))
            )
        ).otherwise(True)
    )
    
    # Filter out invalid measurements
    measurements_filtered = measurements_with_diagnosis.filter(F.col("is_valid_measurement") == True)
    
    # Use standardized_value and unit as final values
    measurements_filtered = measurements_filtered.withColumn(
        "VALUE_SOURCE_VALUE", F.col("standardized_value")
    ).withColumn(
        "UNIT_SOURCE_VALUE", F.col("standardized_unit")
    ).drop("cleaned_value", "is_valid_measurement", "standardized_value", "standardized_unit")

    # ================================================================
    # BLOOD PRESSURE SPECIAL HANDLING
    # Split combined "120/70" format into separate systolic/diastolic rows
    # Handles measurement_type == "BLOOD_PRESSURE" with "SBP/DBP" values
    # ================================================================

    # Pattern: optional spaces around the slash, e.g. "120/70" or "120 / 70"
    bp_pattern = r"^\s*\d+(\.\d+)?\s*/\s*\d+(\.\d+)?\s*$"

    # Separate combined BP rows from all others
    bp_combined = measurements_filtered.filter(
        F.col("VALUE_SOURCE_VALUE").rlike(bp_pattern)
    )
    non_bp = measurements_filtered.filter(
        ~F.col("VALUE_SOURCE_VALUE").rlike(bp_pattern)
    )

    # Create SYSTOLIC rows — take everything before "/"
    systolic_rows = bp_combined.withColumn(
        "measurement_type", F.lit("SYSTOLIC_BLOOD_PRESSURE")
    ).withColumn(
        "VALUE_SOURCE_VALUE",
        F.trim(F.split(F.col("VALUE_SOURCE_VALUE"), "/")[0])
    ).withColumn(
        "UNIT_SOURCE_VALUE", F.lit("mmHg")
    )

    # Create DIASTOLIC rows — take everything after "/"
    diastolic_rows = bp_combined.withColumn(
        "measurement_type", F.lit("DIASTOLIC_BLOOD_PRESSURE")
    ).withColumn(
        "VALUE_SOURCE_VALUE",
        F.trim(F.split(F.col("VALUE_SOURCE_VALUE"), "/")[1])
    ).withColumn(
        "UNIT_SOURCE_VALUE", F.lit("mmHg")
    )

    # Recombine: non-BP rows + split BP rows
    measurements_filtered = non_bp.union(systolic_rows).union(diastolic_rows)

    # ================================================================
    # END OF BLOOD PRESSURE SPLITTING
    # ================================================================

    # Step 7: Define time windows
    measurements_with_windows = measurements_filtered.withColumn(
        "at_diagnosis",
        F.when(
            (F.col("days_from_diagnosis") >= -180) & (F.col("days_from_diagnosis") <= 180),
            F.abs(F.col("days_from_diagnosis"))
        ).otherwise(None)
    ).withColumn(
        "at_2_years",
        F.when(
            (F.col("days_from_diagnosis") >= 550) & (F.col("days_from_diagnosis") <= 910),
            F.abs(F.col("days_from_diagnosis") - 730)
        ).otherwise(None)
    ).withColumn(
        "at_5_years",
        F.when(
            (F.col("days_from_diagnosis") >= 1645) & (F.col("days_from_diagnosis") <= 2005),
            F.abs(F.col("days_from_diagnosis") - 1825)
        ).otherwise(None)
    )
    
    # Add numeric value column for outlier detection
    measurements_with_windows = measurements_with_windows.withColumn(
        "numeric_value",
        F.col("VALUE_SOURCE_VALUE").cast("double")
    )
    
    numeric_measurement_types = [
        "HBA1C", "GLUCOSE", "HEIGHT", "WEIGHT", "TOTAL_CHOLESTEROL", "HDL_CHOLESTEROL",
        "LDL_CHOLESTEROL", "TRIGLYCERIDES", "SYSTOLIC_BLOOD_PRESSURE", "DIASTOLIC_BLOOD_PRESSURE",
        "SERUM_C_PEPTIDE", "ALT", "AST", "BUN", "SERUM_CREATININE", "EGFR",
        "SERUM_CYSTATIN_C", "URINE_MICROALBUMIN", "URINE_CREATININE", "GAD65_ANTIBODY",
        "ICA512_ANTIBODY", "INSULIN_ANTIBODY", "ZNT8_ANTIBODY", "BLOOD_PH", "BICARBONATE", "PCO2"
    ]
    
    def filter_outliers_in_window(df, window_col, window_name):
        df_window = df.filter(F.col(window_col).isNotNull())
        window_stats = Window.partitionBy("PERSON_ID", "measurement_type")
        df_window = df_window.withColumn(
            "q1", F.expr("percentile_approx(numeric_value, 0.25)").over(window_stats)
        ).withColumn(
            "q3", F.expr("percentile_approx(numeric_value, 0.75)").over(window_stats)
        ).withColumn(
            "median_val", F.expr("percentile_approx(numeric_value, 0.5)").over(window_stats)
        ).withColumn(
            "count_in_window", F.count("numeric_value").over(window_stats)
        )
        df_window = df_window.withColumn(
            "iqr", F.col("q3") - F.col("q1")
        ).withColumn(
            "lower_bound", F.col("q1") - (3.0 * F.col("iqr"))
        ).withColumn(
            "upper_bound", F.col("q3") + (3.0 * F.col("iqr"))
        )
        df_window = df_window.withColumn(
            f"is_outlier_{window_name}",
            F.when(
                F.col("measurement_type").isin(numeric_measurement_types) &
                (F.col("count_in_window") >= 2) &
                (F.col("iqr") > 0) &
                (
                    (F.col("numeric_value") < F.col("lower_bound")) |
                    (F.col("numeric_value") > F.col("upper_bound"))
                ),
                True
            ).otherwise(False)
        )
        return df_window.filter(F.col(f"is_outlier_{window_name}") == False).drop(
            "q1", "q3", "median_val", "count_in_window", "iqr",
            "lower_bound", "upper_bound", f"is_outlier_{window_name}"
        )
    
    measurements_diagnosis_clean = filter_outliers_in_window(measurements_with_windows, "at_diagnosis", "diagnosis")
    measurements_2years_clean    = filter_outliers_in_window(measurements_with_windows, "at_2_years",   "2years")
    measurements_5years_clean    = filter_outliers_in_window(measurements_with_windows, "at_5_years",   "5years")

    # ================================================================
    # BLOOD PRESSURE: 3-MEASUREMENT AVERAGING
    #
    # For each time window and each BP type (SYSTOLIC / DIASTOLIC):
    #   - Count how many valid readings the person has in the window
    #   - If >= 3: rank by proximity to target date, take the 3 closest,
    #              average them, and set the flag to True
    #   - If 1 or 2: take the single closest and set the flag to False
    #
    # Returns one row per (PERSON_ID, measurement_type) with:
    #   value_at_{time_point}          — averaged or single numeric value (string)
    #   unit_at_{time_point}           — "mmHg"
    #   datetime_at_{time_point}       — datetime of the closest reading used
    #   at_least_3_bp_{time_point}     — True / False flag
    # ================================================================

    BP_TYPES = ["SYSTOLIC_BLOOD_PRESSURE", "DIASTOLIC_BLOOD_PRESSURE"]

    def process_bp_for_window(df_clean, window_col, time_point_label):
        """
        df_clean     : outlier-filtered dataframe for this window
                       (already filtered so window_col IS NOT NULL)
        window_col   : "at_diagnosis" | "at_2_years" | "at_5_years"
                       holds distance-to-target (smaller = closer)
        time_point_label : "diagnosis" | "2_years" | "5_years"
                           used for output column names

        Returns a dataframe with one row per (PERSON_ID, measurement_type)
        containing the averaged BP value and the boolean flag.
        """
        flag_col = f"at_least_3_bp_{time_point_label}"

        # Keep only BP rows that are in this window
        bp_in_window = df_clean.filter(
            F.col("measurement_type").isin(BP_TYPES) &
            F.col(window_col).isNotNull() &
            F.col("numeric_value").isNotNull()
        )

        # --- Count available readings per person × type ---
        count_win = Window.partitionBy("PERSON_ID", "measurement_type")
        bp_counted = bp_in_window.withColumn(
            "bp_total_count",
            F.count("numeric_value").over(count_win)
        ).withColumn(
            flag_col,
            F.col("bp_total_count") >= 3
        )

        # --- Rank readings by distance to target (ascending = closest first) ---
        rank_win = Window.partitionBy("PERSON_ID", "measurement_type").orderBy(
            F.col(window_col).asc_nulls_last()
        )
        bp_ranked = bp_counted.withColumn("bp_rank", F.row_number().over(rank_win))

        # --- Keep top-3 if flag True, else top-1 ---
        bp_selected = bp_ranked.filter(
            ((F.col(flag_col) == True)  & (F.col("bp_rank") <= 3)) |
            ((F.col(flag_col) == False) & (F.col("bp_rank") == 1))
        )

        # --- Average numeric value per person × type over selected rows ---
        avg_win = Window.partitionBy("PERSON_ID", "measurement_type")
        bp_averaged = bp_selected.withColumn(
            "avg_bp_value",
            F.avg("numeric_value").over(avg_win)
        )

        # --- Take one representative row (the closest reading) for metadata ---
        final_rank_win = Window.partitionBy("PERSON_ID", "measurement_type").orderBy(
            F.col(window_col).asc_nulls_last()
        )
        bp_final = (
            bp_averaged
            .withColumn("final_rank", F.row_number().over(final_rank_win))
            .filter(F.col("final_rank") == 1)
            .select(
                "PERSON_ID",
                "measurement_type",
                F.col("avg_bp_value").cast("string").alias(f"value_at_{time_point_label}"),
                F.col("UNIT_SOURCE_VALUE").alias(f"unit_at_{time_point_label}"),
                F.col("MEASUREMENT_DATETIME").alias(f"datetime_at_{time_point_label}"),
                F.col(flag_col)           # True / False
            )
        )

        return bp_final

    # Generate BP summary tables for each time window
    bp_at_diagnosis = process_bp_for_window(measurements_diagnosis_clean, "at_diagnosis", "diagnosis")
    bp_at_2_years   = process_bp_for_window(measurements_2years_clean,    "at_2_years",   "2_years")
    bp_at_5_years   = process_bp_for_window(measurements_5years_clean,    "at_5_years",   "5_years")

    # ================================================================
    # END OF BLOOD PRESSURE 3-MEASUREMENT AVERAGING
    # ================================================================

    # Step 8: Get the measurement closest to each time point (NON-BP measurements)
    window_diagnosis = Window.partitionBy("PERSON_ID", "measurement_type").orderBy(
        F.col("at_diagnosis").asc_nulls_last()
    )
    window_2_years = Window.partitionBy("PERSON_ID", "measurement_type").orderBy(
        F.col("at_2_years").asc_nulls_last()
    )
    window_5_years = Window.partitionBy("PERSON_ID", "measurement_type").orderBy(
        F.col("at_5_years").asc_nulls_last()
    )

    def get_non_bp_measurements(df_clean, window_spec, window_col, time_point_label):
        """Selects the single closest measurement for all non-BP types."""
        return (
            df_clean
            .filter(
                ~F.col("measurement_type").isin(BP_TYPES) &
                F.col(window_col).isNotNull()
            )
            .withColumn("rank", F.row_number().over(window_spec))
            .filter(F.col("rank") == 1)
            .select(
                "PERSON_ID",
                "measurement_type",
                F.col("VALUE_SOURCE_VALUE").alias(f"value_at_{time_point_label}"),
                F.col("UNIT_SOURCE_VALUE").alias(f"unit_at_{time_point_label}"),
                F.col("MEASUREMENT_DATETIME").alias(f"datetime_at_{time_point_label}")
            )
        )

    non_bp_at_diagnosis = get_non_bp_measurements(measurements_diagnosis_clean, window_diagnosis, "at_diagnosis", "diagnosis")
    non_bp_at_2_years   = get_non_bp_measurements(measurements_2years_clean,    window_2_years,   "at_2_years",   "2_years")
    non_bp_at_5_years   = get_non_bp_measurements(measurements_5years_clean,    window_5_years,   "at_5_years",   "5_years")

    # Merge non-BP and BP results for each time point
    # BP has an extra flag column; fill it with null for non-BP rows
    measurements_at_diagnosis = (
        non_bp_at_diagnosis
        .withColumn("at_least_3_bp_diagnosis", F.lit(None).cast("boolean"))
        .union(bp_at_diagnosis.select(
            "PERSON_ID", "measurement_type",
            "value_at_diagnosis", "unit_at_diagnosis", "datetime_at_diagnosis",
            "at_least_3_bp_diagnosis"
        ))
    )

    measurements_at_2_years = (
        non_bp_at_2_years
        .withColumn("at_least_3_bp_2_years", F.lit(None).cast("boolean"))
        .union(bp_at_2_years.select(
            "PERSON_ID", "measurement_type",
            "value_at_2_years", "unit_at_2_years", "datetime_at_2_years",
            "at_least_3_bp_2_years"
        ))
    )

    measurements_at_5_years = (
        non_bp_at_5_years
        .withColumn("at_least_3_bp_5_years", F.lit(None).cast("boolean"))
        .union(bp_at_5_years.select(
            "PERSON_ID", "measurement_type",
            "value_at_5_years", "unit_at_5_years", "datetime_at_5_years",
            "at_least_3_bp_5_years"
        ))
    )

    # Step 9: Pivot to create columns for each measurement type at each time point
    measurement_types = [
        "HBA1C", "GLUCOSE", "HEIGHT", "WEIGHT", "TOTAL_CHOLESTEROL", "HDL_CHOLESTEROL",
        "LDL_CHOLESTEROL", "TRIGLYCERIDES", "SYSTOLIC_BLOOD_PRESSURE", "DIASTOLIC_BLOOD_PRESSURE",
        "SERUM_C_PEPTIDE", "ALT", "AST", "BUN", "SERUM_CREATININE", "EGFR",
        "SERUM_CYSTATIN_C", "URINE_MICROALBUMIN", "URINE_CREATININE", "GAD65_ANTIBODY",
        "ICA512_ANTIBODY", "INSULIN_ANTIBODY", "ZNT8_ANTIBODY", "URINE_KETONE",
        "BLOOD_PH", "BICARBONATE", "PCO2"
    ]

    # Combine all time points into one long table
    all_measurements = (
        measurements_at_diagnosis
        .join(measurements_at_2_years, on=["PERSON_ID", "measurement_type"], how="outer")
        .join(measurements_at_5_years, on=["PERSON_ID", "measurement_type"], how="outer")
    )

    # Step 10: Pivot — create one column per (measurement_type × time_point)
    pivoted_measurements = all_measurements

    for time_point in ["diagnosis", "2_years", "5_years"]:
        for meas_type in measurement_types:
            col_name      = f"{meas_type.lower()}_at_{time_point}"
            unit_col_name = f"{meas_type.lower()}_unit_at_{time_point}"
            dt_col_name   = f"{meas_type.lower()}_datetime_at_{time_point}"

            pivoted_measurements = (
                pivoted_measurements
                .withColumn(col_name,      F.when(F.col("measurement_type") == meas_type, F.col(f"value_at_{time_point}")))
                .withColumn(unit_col_name, F.when(F.col("measurement_type") == meas_type, F.col(f"unit_at_{time_point}")))
                .withColumn(dt_col_name,   F.when(F.col("measurement_type") == meas_type, F.col(f"datetime_at_{time_point}")))
            )

    # Add BP flag columns (only meaningful for BP rows; will collapse to True/False after groupBy)
    for time_point in ["diagnosis", "2_years", "5_years"]:
        for bp_type in ["systolic_blood_pressure", "diastolic_blood_pressure"]:
            flag_col = f"at_least_3_bp_{time_point}"
            pivoted_measurements = pivoted_measurements.withColumn(
                f"{bp_type}_at_least_3_bp_{time_point}",
                F.when(
                    F.col("measurement_type") == bp_type.upper(),
                    F.col(flag_col)
                )
            )

    # Group by PERSON_ID to consolidate rows
    agg_expressions = []
    for time_point in ["diagnosis", "2_years", "5_years"]:
        for meas_type in measurement_types:
            col_name      = f"{meas_type.lower()}_at_{time_point}"
            unit_col_name = f"{meas_type.lower()}_unit_at_{time_point}"
            dt_col_name   = f"{meas_type.lower()}_datetime_at_{time_point}"
            agg_expressions.extend([
                F.max(col_name).alias(col_name),
                F.max(unit_col_name).alias(unit_col_name),
                F.max(dt_col_name).alias(dt_col_name)
            ])

    # Include BP flag aggregations
    # We use max() — True > False > null, so if any row says True it wins
    for time_point in ["diagnosis", "2_years", "5_years"]:
        for bp_type in ["systolic_blood_pressure", "diastolic_blood_pressure"]:
            agg_expressions.append(
                F.max(f"{bp_type}_at_least_3_bp_{time_point}").alias(
                    f"{bp_type}_at_least_3_bp_{time_point}"
                )
            )

    final_measurements = pivoted_measurements.groupBy("PERSON_ID").agg(*agg_expressions)

    # Step 11: Calculate BMI
    for time_point in ["diagnosis", "2_years", "5_years"]:
        height_col = f"height_at_{time_point}"
        weight_col = f"weight_at_{time_point}"
        bmi_col    = f"bmi_at_{time_point}"
        final_measurements = final_measurements.withColumn(
            bmi_col,
            F.when(
                F.col(height_col).isNotNull() &
                F.col(weight_col).isNotNull() &
                (F.col(height_col).cast("double") > 0),
                (F.col(weight_col).cast("double") / (F.col(height_col).cast("double") * F.col(height_col).cast("double"))) * 703.0
            ).otherwise(None)
        ).withColumn(
            f"bmi_unit_at_{time_point}",
            F.when(F.col(bmi_col).isNotNull(), F.lit("kg/m^2")).otherwise(None)
        ).withColumn(
            f"bmi_datetime_at_{time_point}",
            F.col(f"weight_datetime_at_{time_point}")
        )

    # Step 12: Calculate Urine Microalbumin/Creatinine Ratio
    for time_point in ["diagnosis", "2_years", "5_years"]:
        microalbumin_col = f"urine_microalbumin_at_{time_point}"
        creatinine_col   = f"urine_creatinine_at_{time_point}"
        ratio_col        = f"urine_microalbumin_creatinine_ratio_at_{time_point}"
        final_measurements = final_measurements.withColumn(
            ratio_col,
            F.when(
                F.col(microalbumin_col).isNotNull() &
                F.col(creatinine_col).isNotNull() &
                (F.col(creatinine_col).cast("double") > 0),
                (F.col(microalbumin_col).cast("double") / F.col(creatinine_col).cast("double")) * 1000.0
            ).otherwise(None)
        ).withColumn(
            f"urine_microalbumin_creatinine_ratio_unit_at_{time_point}",
            F.when(F.col(ratio_col).isNotNull(), F.lit("mg/g")).otherwise(None)
        ).withColumn(
            f"urine_microalbumin_creatinine_ratio_datetime_at_{time_point}",
            F.col(f"urine_microalbumin_datetime_at_{time_point}")
        )

    # Step 13: Join back to original dataframe
    result = original.join(final_measurements, on="PERSON_ID", how="left")

    # Step 14: Debug column
    result = result.withColumn(
        "_debug_weight_difference_date",
        F.datediff(F.col("weight_datetime_at_diagnosis"), F.col("DiagnosisDateTime"))
    )

    output_dataset.write_dataframe(result)