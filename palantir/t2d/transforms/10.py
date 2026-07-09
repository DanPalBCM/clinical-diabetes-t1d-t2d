"""
Palantir Foundry PySpark Transform: Blood Pressure Percentile Calculation
Purpose: Calculate BP percentiles for T2D patients at diagnosis, 2 years, and 5 years
         NOTE: Percentiles only calculated for patients aged 13 or younger at each time point.
               Age matching is EXACT. Height percentile matching uses closest available value.

COLUMN STRUCTURE (confirmed from data):
  - Systolic BP: systolic_blood_pressure_at_diagnosis / _at_2_years / _at_5_years  (numeric)
  - Diastolic BP: diastolic_blood_pressure_at_diagnosis / _at_2_years / _at_5_years (numeric, SEPARATE columns)
  - SBP and DBP are NOT stored as a combined "130/80" string — they are individual numeric columns.
"""

from pyspark.sql import functions as F
from pyspark.sql import Window
from transforms.api import transform_df, Input, Output


@transform_df(
    Output("ri.foundry.main.dataset.xxxxx"),
    patient_data=Input("ri.foundry.main.dataset.xxxxx"),
    bp_reference=Input("ri.foundry.main.dataset.xxxxx")
)
def compute(patient_data, bp_reference):
    """
    Calculate blood pressure percentiles at three time points.
    Only patients aged <= 13 at each time point receive a percentile.
    Age matching is exact. Height percentile uses closest available reference value.
    """

    # ============================================================================
    # STEP 1: Parse blood pressure values
    # SBP and DBP are stored in SEPARATE numeric columns — cast directly, no splitting.
    # ============================================================================

    df = patient_data

    # SBP — cast each time point's dedicated systolic column
    df = df.withColumn(
        "sbp_at_diagnosis_parsed",
        F.col("systolic_blood_pressure_at_diagnosis").cast("integer")
    ).withColumn(
        "sbp_at_2_years_parsed",
        F.col("systolic_blood_pressure_at_2_years").cast("integer")
    ).withColumn(
        "sbp_at_5_years_parsed",
        F.col("systolic_blood_pressure_at_5_years").cast("integer")
    )

    # DBP — cast each time point's dedicated diastolic column (NOT split from systolic)
    df = df.withColumn(
        "dbp_at_diagnosis_parsed",
        F.col("diastolic_blood_pressure_at_diagnosis").cast("integer")
    ).withColumn(
        "dbp_at_2_years_parsed",
        F.col("diastolic_blood_pressure_at_2_years").cast("integer")
    ).withColumn(
        "dbp_at_5_years_parsed",
        F.col("diastolic_blood_pressure_at_5_years").cast("integer")
    )

    # ============================================================================
    # STEP 2: Calculate age at each time point
    # ============================================================================

    df = df.withColumn(
        "age_at_diagnosis_blood_pressure_calculation",
        F.floor(
            F.months_between(
                F.col("systolic_blood_pressure_datetime_at_diagnosis"),
                F.col("date_of_birth")
            ) / 12
        ).cast("integer")
    )

    df = df.withColumn(
        "age_at_2_years",
        F.floor(
            F.months_between(
                F.col("diastolic_blood_pressure_datetime_at_2_years"),
                F.col("date_of_birth")
            ) / 12
        ).cast("integer")
    )

    df = df.withColumn(
        "age_at_5_years",
        F.floor(
            F.months_between(
                F.col("diastolic_blood_pressure_datetime_at_5_years"),
                F.col("date_of_birth")
            ) / 12
        ).cast("integer")
    )

    # ============================================================================
    # STEP 3: Keep original height percentile (no rounding — matched in helper)
    # ============================================================================

    df = df.withColumn(
        "height_percentile_for_lookup",
        F.col("height_percentile_at_diagnosis")
    )

    # ============================================================================
    # STEP 4: Prepare reference table
    # ============================================================================

    bp_ref = bp_reference.select(
        F.lower(F.col("sex")).alias("sex_ref"),
        F.col("age_years").alias("age_ref"),
        F.col("bp_reference").alias("bp_percentile_ref"),
        F.col("height_percentile").alias("height_percentile_ref"),
        F.col("sbp_mmHg").alias("sbp_threshold"),
        F.col("dbp_mmHg").alias("dbp_threshold")
    )

    # ============================================================================
    # STEP 5: Helper function — exact age match, closest height percentile
    # ============================================================================

    def calculate_single_bp_percentile(df, age_col, bp_col, bp_type, output_col_name):
        """
        Calculate a single BP percentile (SBP or DBP).

        Rules:
        - Patients OLDER THAN 13 at this time point receive NULL (not eligible).
        - Age match must be EXACT (age_col == age_ref). No fuzzy age matching.
        - Height percentile uses the CLOSEST available reference value (5,10,25,50,75,90,95).
        - If the patient's BP is below ALL thresholds for their exact age, they receive NULL.
        - All patients are preserved in the output (LEFT JOIN semantics).
        """

        AGE_CUTOFF = 13

        # Assign a stable row ID and lowercase sex
        patient_for_join = df.withColumn(
            "sex_lower", F.lower(F.col("sex"))
        ).withColumn(
            "patient_row_id", F.monotonically_increasing_id()
        )

        # -----------------------------------------------------------------------
        # Join: LEFT join so all patients are kept.
        # Filter reference rows to EXACT age match only.
        # Patients over 13 will still join but we null out results at the end.
        # -----------------------------------------------------------------------
        joined = patient_for_join.alias("p").join(
            bp_ref.alias("r"),
            (F.col("p.sex_lower") == F.col("r.sex_ref")) &
            (F.col(f"p.{age_col}") == F.col("r.age_ref")),   # <-- EXACT AGE MATCH
            "left"
        )

        # Distance metric: only on height percentile (age is already exact)
        joined = joined.withColumn(
            "height_percentile_distance",
            F.abs(F.col("p.height_percentile_for_lookup") - F.col("r.height_percentile_ref"))
        ).withColumn(
            "meets_threshold",
            F.when(F.col(f"p.{bp_col}") >= F.col(f"r.{bp_type}"), 1).otherwise(0)
        )

        # -----------------------------------------------------------------------
        # For each patient + bp_percentile_ref combo, keep the closest height row
        # -----------------------------------------------------------------------
        window_closest = Window.partitionBy(
            "p.patient_row_id",
            "r.bp_percentile_ref"
        ).orderBy(
            F.col("height_percentile_distance").asc()
        )

        joined = joined.withColumn(
            "rank_closest",
            F.row_number().over(window_closest)
        ).filter(
            F.col("rank_closest") == 1
        )

        # -----------------------------------------------------------------------
        # Among remaining rows, pick the highest percentile where threshold is met
        # -----------------------------------------------------------------------
        window_percentile = Window.partitionBy("p.patient_row_id").orderBy(
            F.when(F.col("meets_threshold") == 1, F.col("r.bp_percentile_ref")).desc()
        )

        result = joined.withColumn(
            "row_num",
            F.row_number().over(window_percentile)
        ).filter(
            F.col("row_num") == 1
        )

        # -----------------------------------------------------------------------
        # Apply age cutoff: patients OLDER THAN 13 get NULL regardless
        # -----------------------------------------------------------------------
        result = result.withColumn(
            output_col_name,
            F.when(
                F.col(f"p.{age_col}") > AGE_CUTOFF,
                F.lit(None)                              # over 13 → not eligible
            ).when(
                F.col("meets_threshold") == 1,
                F.col("r.bp_percentile_ref")             # threshold met → assign percentile
            ).otherwise(
                F.lit(None)                              # below all thresholds → NULL
            )
        ).withColumn(
            f"{output_col_name}_matched_age",
            F.when(
                (F.col(f"p.{age_col}") <= AGE_CUTOFF) & (F.col("meets_threshold") == 1),
                F.col("r.age_ref")
            ).otherwise(F.lit(None))
        ).withColumn(
            f"{output_col_name}_matched_height_pct",
            F.when(
                (F.col(f"p.{age_col}") <= AGE_CUTOFF) & (F.col("meets_threshold") == 1),
                F.col("r.height_percentile_ref")
            ).otherwise(F.lit(None))
        )

        # Select original patient columns + new output columns
        patient_cols = [f"p.{c}" for c in df.columns]
        result = result.select(
            *patient_cols,
            output_col_name,
            f"{output_col_name}_matched_age",
            f"{output_col_name}_matched_height_pct"
        ).drop("patient_row_id", "sex_lower")

        return result

    # ============================================================================
    # STEP 6a: Debugging column for SBP at diagnosis
    # ============================================================================

    df = df.withColumn(
        "reason_for_empty_SBP_percentile_debugging",
        F.when(F.col("age_at_diagnosis_blood_pressure_calculation") > 13,
               F.concat(
                   F.lit("Patient is older than 13 at diagnosis (age: "),
                   F.col("age_at_diagnosis_blood_pressure_calculation").cast("string"),
                   F.lit(") — BP percentile not applicable")
               ))
        .when(F.col("systolic_blood_pressure_at_diagnosis").isNull(),
              "Original SBP value is null")
        .when(F.col("sbp_at_diagnosis_parsed").isNull(),
              F.concat(
                  F.lit("SBP parsing failed — original value: "),
                  F.coalesce(F.col("systolic_blood_pressure_at_diagnosis").cast("string"), F.lit("NULL"))
              ))
        .when(F.col("systolic_blood_pressure_datetime_at_diagnosis").isNull(),
              "SBP datetime is null")
        .when(F.col("date_of_birth").isNull(),
              "Date of birth is null")
        .when(F.col("age_at_diagnosis_blood_pressure_calculation").isNull(),
              "Age calculation failed")
        .when(F.col("sex").isNull(),
              "Sex is null")
        .when(F.col("height_percentile_at_diagnosis").isNull(),
              "Height percentile is null")
        .otherwise("All inputs present — checking reference table match")
    )

    # ============================================================================
    # STEP 6b: Debugging column for DBP at diagnosis
    # ============================================================================

    df = df.withColumn(
        "reason_for_empty_DBP_percentile_debugging",
        F.when(F.col("age_at_diagnosis_blood_pressure_calculation") > 13,
               F.concat(
                   F.lit("Patient is older than 13 at diagnosis (age: "),
                   F.col("age_at_diagnosis_blood_pressure_calculation").cast("string"),
                   F.lit(") — BP percentile not applicable")
               ))
        .when(F.col("diastolic_blood_pressure_at_diagnosis").isNull(),
              "Original DBP value is null")
        .when(F.col("dbp_at_diagnosis_parsed").isNull(),
              F.concat(
                  F.lit("DBP parsing failed — original value: "),
                  F.coalesce(F.col("diastolic_blood_pressure_at_diagnosis").cast("string"), F.lit("NULL"))
              ))
        .when(F.col("systolic_blood_pressure_datetime_at_diagnosis").isNull(),
              "BP datetime is null")
        .when(F.col("date_of_birth").isNull(),
              "Date of birth is null")
        .when(F.col("age_at_diagnosis_blood_pressure_calculation").isNull(),
              "Age calculation failed")
        .when(F.col("sex").isNull(),
              "Sex is null")
        .when(F.col("height_percentile_at_diagnosis").isNull(),
              "Height percentile is null")
        .otherwise("All inputs present — checking reference table match")
    )

    # ============================================================================
    # STEP 7: Calculate all percentiles
    # ============================================================================

    # --- Diagnosis ---
    df = calculate_single_bp_percentile(
        df, "age_at_diagnosis_blood_pressure_calculation", "sbp_at_diagnosis_parsed", "sbp_threshold", "sbp_percentile_at_diagnosis"
    )

    # Update SBP debugging message after calculation
    df = df.withColumn(
        "reason_for_empty_SBP_percentile_debugging",
        F.when(
            F.col("sbp_percentile_at_diagnosis").isNotNull(),
            F.concat(
                F.lit("Success — matched to Age: "),
                F.col("sbp_percentile_at_diagnosis_matched_age").cast("string"),
                F.lit(", Height %ile: "),
                F.col("sbp_percentile_at_diagnosis_matched_height_pct").cast("string"),
                F.lit(", Percentile: "),
                F.col("sbp_percentile_at_diagnosis")
            )
        ).otherwise(
            F.when(
                F.col("reason_for_empty_SBP_percentile_debugging") == "All inputs present — checking reference table match",
                F.concat(
                    F.lit("BP below all thresholds — Sex: "),
                    F.coalesce(F.col("sex"), F.lit("NULL")),
                    F.lit(", Age: "),
                    F.coalesce(F.col("age_at_diagnosis_blood_pressure_calculation").cast("string"), F.lit("NULL")),
                    F.lit(", Height %ile: "),
                    F.coalesce(F.col("height_percentile_for_lookup").cast("string"), F.lit("NULL")),
                    F.lit(", SBP: "),
                    F.coalesce(F.col("sbp_at_diagnosis_parsed").cast("string"), F.lit("NULL"))
                )
            ).otherwise(F.col("reason_for_empty_SBP_percentile_debugging"))
        )
    ).drop("sbp_percentile_at_diagnosis_matched_age", "sbp_percentile_at_diagnosis_matched_height_pct")

    df = calculate_single_bp_percentile(
        df, "age_at_diagnosis_blood_pressure_calculation", "dbp_at_diagnosis_parsed", "dbp_threshold", "dbp_percentile_at_diagnosis"
    )

    # Update DBP debugging message after calculation
    df = df.withColumn(
        "reason_for_empty_DBP_percentile_debugging",
        F.when(
            F.col("dbp_percentile_at_diagnosis").isNotNull(),
            F.concat(
                F.lit("Success — matched to Age: "),
                F.col("dbp_percentile_at_diagnosis_matched_age").cast("string"),
                F.lit(", Height %ile: "),
                F.col("dbp_percentile_at_diagnosis_matched_height_pct").cast("string"),
                F.lit(", Percentile: "),
                F.col("dbp_percentile_at_diagnosis")
            )
        ).otherwise(
            F.when(
                F.col("reason_for_empty_DBP_percentile_debugging") == "All inputs present — checking reference table match",
                F.concat(
                    F.lit("BP below all thresholds — Sex: "),
                    F.coalesce(F.col("sex"), F.lit("NULL")),
                    F.lit(", Age: "),
                    F.coalesce(F.col("age_at_diagnosis_blood_pressure_calculation").cast("string"), F.lit("NULL")),
                    F.lit(", Height %ile: "),
                    F.coalesce(F.col("height_percentile_for_lookup").cast("string"), F.lit("NULL")),
                    F.lit(", DBP: "),
                    F.coalesce(F.col("dbp_at_diagnosis_parsed").cast("string"), F.lit("NULL"))
                )
            ).otherwise(F.col("reason_for_empty_DBP_percentile_debugging"))
        )
    ).drop("dbp_percentile_at_diagnosis_matched_age", "dbp_percentile_at_diagnosis_matched_height_pct")

    # --- 2 Years ---
    df = calculate_single_bp_percentile(
        df, "age_at_2_years", "sbp_at_2_years_parsed", "sbp_threshold", "sbp_percentile_at_2_years"
    ).drop("sbp_percentile_at_2_years_matched_age", "sbp_percentile_at_2_years_matched_height_pct")

    df = calculate_single_bp_percentile(
        df, "age_at_2_years", "dbp_at_2_years_parsed", "dbp_threshold", "dbp_percentile_at_2_years"
    ).drop("dbp_percentile_at_2_years_matched_age", "dbp_percentile_at_2_years_matched_height_pct")

    # --- 5 Years ---
    df = calculate_single_bp_percentile(
        df, "age_at_5_years", "sbp_at_5_years_parsed", "sbp_threshold", "sbp_percentile_at_5_years"
    ).drop("sbp_percentile_at_5_years_matched_age", "sbp_percentile_at_5_years_matched_height_pct")

    df = calculate_single_bp_percentile(
        df, "age_at_5_years", "dbp_at_5_years_parsed", "dbp_threshold", "dbp_percentile_at_5_years"
    ).drop("dbp_percentile_at_5_years_matched_age", "dbp_percentile_at_5_years_matched_height_pct")

    # ============================================================================
    # STEP 8: Return final dataset
    # ============================================================================

    return df.orderBy("mrn")