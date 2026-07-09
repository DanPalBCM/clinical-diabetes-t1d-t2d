from pyspark.sql import functions as F
from pyspark.sql import Window
from transforms.api import transform, Input, Output
from pyspark.sql.types import DoubleType, BooleanType, StringType
import math
from pyspark.sql.functions import pandas_udf
import pandas as pd


# ============================================================================
# Pandas UDFs for vectorized z-score and percentile calculations
# ============================================================================
@pandas_udf(DoubleType())
def calculate_z_score_pandas(value: pd.Series, L: pd.Series, M: pd.Series, S: pd.Series) -> pd.Series:
    """Vectorized z-score calculation using LMS method"""
    import numpy as np
    
    # Handle null values
    mask = value.notna() & M.notna() & L.notna() & S.notna() & (M != 0) & (S != 0)
    result = pd.Series([None] * len(value), dtype=float)
    
    if mask.any():
        # Convert to numpy arrays to avoid index alignment issues
        valid_values = value[mask].values
        valid_L = L[mask].values
        valid_M = M[mask].values
        valid_S = S[mask].values
        
        # Initialize z_scores array
        z_scores = np.full(len(valid_values), np.nan)
        
        # When L != 0
        l_nonzero = valid_L != 0
        if l_nonzero.any():
            z_scores[l_nonzero] = ((valid_values[l_nonzero] / valid_M[l_nonzero]) ** valid_L[l_nonzero] - 1) / (valid_L[l_nonzero] * valid_S[l_nonzero])
        
        # When L = 0
        l_zero = valid_L == 0
        if l_zero.any():
            z_scores[l_zero] = np.log(valid_values[l_zero] / valid_M[l_zero]) / valid_S[l_zero]
        
        result[mask] = z_scores
    
    return result


@pandas_udf(DoubleType())
def z_to_percentile_pandas(z: pd.Series) -> pd.Series:
    """Vectorized percentile conversion using normal CDF"""
    from scipy.special import erf
    import numpy as np
    
    result = pd.Series([None] * len(z), dtype=float)
    mask = z.notna()
    
    if mask.any():
        valid_z = z[mask].values
        percentiles = 0.5 * (1 + erf(valid_z / np.sqrt(2))) * 100
        result[mask] = percentiles
    
    return result


# ============================================================================
# Main function to add growth percentiles
# ============================================================================
def add_growth_percentiles(df, time_point, measurement, cdc_table, who_table):
    """
    Add percentiles and z-scores for height, weight, or BMI at a specific time point
    
    Args:
        df: Input dataframe
        time_point: 'diagnosis', '2_years', or '5_years'
        measurement: 'height_cm', 'weight_kg', or 'bmi'
        cdc_table: CDC reference table
        who_table: WHO reference table
    """
    value_col = f"{measurement}_at_{time_point}"
    age_col = f"{measurement}_age_months_at_{time_point}"
    percentile_col = f"{measurement}_percentile_at_{time_point}"
    zscore_col = f"{measurement}_zscore_at_{time_point}"
    
    # Early return if required columns don't exist
    required_cols = [value_col, age_col, "sex_numeric"]
    if not all(col in df.columns for col in required_cols):
        return df.withColumn(percentile_col, F.lit(None).cast(DoubleType())) \
                 .withColumn(zscore_col, F.lit(None).cast(DoubleType()))
    
    # Round age to nearest 0.5 months for WHO and nearest month for CDC to match reference tables
    df = df.withColumn(
        f"_rounded_age_{time_point}_{measurement}",
        F.col(age_col)
    )
    
    # Add flag for WHO vs CDC (WHO for < 24 months, CDC for >= 24 months)
    df = df.withColumn(
        f"_use_who_{time_point}_{measurement}", 
        (F.col(age_col) >= 0) & (F.col(age_col) < 24)
    )
    
    df = df.withColumn(
        f"_use_cdc_{time_point}_{measurement}", 
        (F.col(age_col) >= 24) & (F.col(age_col) <= 240)
    )
    
    # Add a unique row identifier to prevent issues with window functions
    df = df.withColumn(f"_row_id_{time_point}_{measurement}", F.monotonically_increasing_id())
    
    # Prepare WHO table with unique column names
    who_prepared = who_table.select(
        F.col("age_months").alias(f"who_age_{time_point}_{measurement}"),
        F.col("sex").alias(f"who_sex_{time_point}_{measurement}"),
        F.col("L").alias(f"who_L_{time_point}_{measurement}"),
        F.col("M").alias(f"who_M_{time_point}_{measurement}"),
        F.col("S").alias(f"who_S_{time_point}_{measurement}")
    )
    
    # Prepare CDC table with unique column names
    cdc_prepared = cdc_table.select(
        F.col("age_months").alias(f"cdc_age_{time_point}_{measurement}"),
        F.col("sex").alias(f"cdc_sex_{time_point}_{measurement}"),
        F.col("L").alias(f"cdc_L_{time_point}_{measurement}"),
        F.col("M").alias(f"cdc_M_{time_point}_{measurement}"),
        F.col("S").alias(f"cdc_S_{time_point}_{measurement}")
    )
    
    # ========================================================================
    # Join with reference tables: match on sex, then find nearest age
    # ========================================================================
    
    # Join with WHO table - match on sex only for children < 24 months
    df_with_who = df.join(
        who_prepared,
        (F.col("sex_numeric") == F.col(f"who_sex_{time_point}_{measurement}")) &
        (F.col(f"_use_who_{time_point}_{measurement}")),
        how="left"
    )
    
    # Calculate distance to reference age and rank by distance
    df_with_who = df_with_who.withColumn(
        f"_who_age_dist_{time_point}_{measurement}",
        F.abs(F.col(f"_rounded_age_{time_point}_{measurement}") - F.col(f"who_age_{time_point}_{measurement}"))
    )
    
    # Window to find the closest WHO age for each patient
    window_who = Window.partitionBy(f"_row_id_{time_point}_{measurement}").orderBy(
        F.col(f"_who_age_dist_{time_point}_{measurement}")
    )
    
    df_with_who = df_with_who.withColumn("_who_rank", F.row_number().over(window_who))
    
    # Keep only the closest match for WHO
    df_with_who = df_with_who.filter((F.col("_who_rank") == 1) | (F.col("_who_rank").isNull()))
    df_with_who = df_with_who.drop("_who_rank", f"_who_age_dist_{time_point}_{measurement}")
    
    # Join with CDC table - match on sex only for children >= 24 months
    df_with_both = df_with_who.join(
        cdc_prepared,
        (F.col("sex_numeric") == F.col(f"cdc_sex_{time_point}_{measurement}")) &
        (F.col(f"_use_cdc_{time_point}_{measurement}")),
        how="left"
    )
    
    # Calculate distance to reference age and rank by distance
    df_with_both = df_with_both.withColumn(
        f"_cdc_age_dist_{time_point}_{measurement}",
        F.abs(F.col(f"_rounded_age_{time_point}_{measurement}") - F.col(f"cdc_age_{time_point}_{measurement}"))
    )
    
    # Window to find the closest CDC age for each patient
    window_cdc = Window.partitionBy(f"_row_id_{time_point}_{measurement}").orderBy(
        F.col(f"_cdc_age_dist_{time_point}_{measurement}")
    )
    
    df_with_both = df_with_both.withColumn("_cdc_rank", F.row_number().over(window_cdc))
    
    # Keep only the closest match for CDC
    df_with_both = df_with_both.filter((F.col("_cdc_rank") == 1) | (F.col("_cdc_rank").isNull()))
    df_with_both = df_with_both.drop("_cdc_rank", f"_cdc_age_dist_{time_point}_{measurement}")
    
    # ========================================================================
    # DEBUG COLUMNS - Only for weight at diagnosis
    # ========================================================================
    if measurement == "weight_kg" and time_point == "diagnosis":
        # a. Age in months used to compute z-score
        df_with_both = df_with_both.withColumn(
            "_debug_age_months_used",
            F.col(age_col)
        )
        
        # b. Weight value in original and converted units
        # Original weight (in ounces from input)
        df_with_both = df_with_both.withColumn(
            "_debug_weight_original_oz",
            F.col("weight_at_diagnosis")  # This is the original value before conversion
        )
        
        # Converted weight (in kg used for calculation)
        df_with_both = df_with_both.withColumn(
            "_debug_weight_converted_kg",
            F.col(value_col)
        )
        
        # c. Gender used to calculate z-score
        df_with_both = df_with_both.withColumn(
            "_debug_gender_numeric",
            F.col("sex_numeric")
        )
        
        df_with_both = df_with_both.withColumn(
            "_debug_gender_text",
            F.when(F.col("sex_numeric") == 1, F.lit("MALE"))
             .when(F.col("sex_numeric") == 2, F.lit("FEMALE"))
             .otherwise(F.lit("UNKNOWN"))
        )
        
        # d. Reference table used (WHO or CDC)
        df_with_both = df_with_both.withColumn(
            "_debug_reference_table",
            F.when(F.col(f"_use_who_{time_point}_{measurement}"), F.lit("WHO"))
             .when(F.col(f"_use_cdc_{time_point}_{measurement}"), F.lit("CDC"))
             .otherwise(F.lit("NONE"))
        )
        
        # e. LMS values used
        df_with_both = df_with_both.withColumn(
            "_debug_L_value",
            F.when(
                F.col(f"_use_who_{time_point}_{measurement}"),
                F.col(f"who_L_{time_point}_{measurement}")
            ).when(
                F.col(f"_use_cdc_{time_point}_{measurement}"),
                F.col(f"cdc_L_{time_point}_{measurement}")
            ).otherwise(None)
        )
        
        df_with_both = df_with_both.withColumn(
            "_debug_M_value",
            F.when(
                F.col(f"_use_who_{time_point}_{measurement}"),
                F.col(f"who_M_{time_point}_{measurement}")
            ).when(
                F.col(f"_use_cdc_{time_point}_{measurement}"),
                F.col(f"cdc_M_{time_point}_{measurement}")
            ).otherwise(None)
        )
        
        df_with_both = df_with_both.withColumn(
            "_debug_S_value",
            F.when(
                F.col(f"_use_who_{time_point}_{measurement}"),
                F.col(f"who_S_{time_point}_{measurement}")
            ).when(
                F.col(f"_use_cdc_{time_point}_{measurement}"),
                F.col(f"cdc_S_{time_point}_{measurement}")
            ).otherwise(None)
        )
        
        # f. Matched reference age
        df_with_both = df_with_both.withColumn(
            "_debug_matched_reference_age",
            F.when(
                F.col(f"_use_who_{time_point}_{measurement}"),
                F.col(f"who_age_{time_point}_{measurement}")
            ).when(
                F.col(f"_use_cdc_{time_point}_{measurement}"),
                F.col(f"cdc_age_{time_point}_{measurement}")
            ).otherwise(None)
        )
    
    # Calculate z-scores using appropriate LMS values
    df_with_both = df_with_both.withColumn(
        zscore_col,
        F.when(
            F.col(f"_use_who_{time_point}_{measurement}") & 
            F.col(f"who_L_{time_point}_{measurement}").isNotNull(),
            calculate_z_score_pandas(
                F.col(value_col).cast("double"),
                F.col(f"who_L_{time_point}_{measurement}").cast("double"),
                F.col(f"who_M_{time_point}_{measurement}").cast("double"),
                F.col(f"who_S_{time_point}_{measurement}").cast("double")
            )
        ).when(
            F.col(f"_use_cdc_{time_point}_{measurement}") & 
            F.col(f"cdc_L_{time_point}_{measurement}").isNotNull(),
            calculate_z_score_pandas(
                F.col(value_col).cast("double"),
                F.col(f"cdc_L_{time_point}_{measurement}").cast("double"),
                F.col(f"cdc_M_{time_point}_{measurement}").cast("double"),
                F.col(f"cdc_S_{time_point}_{measurement}").cast("double")
            )
        ).otherwise(None)
    )
    
    # Add debug column for z-score (only for weight at diagnosis)
    if measurement == "weight_kg" and time_point == "diagnosis":
        df_with_both = df_with_both.withColumn(
            "_debug_zscore_calculated",
            F.col(zscore_col)
        )
    
    # Calculate percentiles from z-scores
    df_with_both = df_with_both.withColumn(
        percentile_col,
        z_to_percentile_pandas(F.col(zscore_col))
    )
    
    # Clean up temporary columns
    cols_to_drop = [
        f"_row_id_{time_point}_{measurement}",
        f"_rounded_age_{time_point}_{measurement}",
        f"_use_who_{time_point}_{measurement}",
        f"_use_cdc_{time_point}_{measurement}",
        f"who_age_{time_point}_{measurement}", f"who_sex_{time_point}_{measurement}", 
        f"who_L_{time_point}_{measurement}", f"who_M_{time_point}_{measurement}", f"who_S_{time_point}_{measurement}",
        f"cdc_age_{time_point}_{measurement}", f"cdc_sex_{time_point}_{measurement}", 
        f"cdc_L_{time_point}_{measurement}", f"cdc_M_{time_point}_{measurement}", f"cdc_S_{time_point}_{measurement}"
    ]
    
    for col in cols_to_drop:
        if col in df_with_both.columns:
            df_with_both = df_with_both.drop(col)
    
    return df_with_both


@transform(
    output_dataset=Output("ri.foundry.main.dataset.xxxxx"),
    input_measurements=Input("ri.foundry.main.dataset.xxxxx"),
    cdc_height=Input("ri.foundry.main.dataset.xxxxx"),
    cdc_weight=Input("ri.foundry.main.dataset.xxxxx"),
    cdc_bmi=Input("ri.foundry.main.dataset.xxxxx"),
    who_bmi_girls=Input("ri.foundry.main.dataset.xxxxx"),
    who_bmi_boys=Input("ri.foundry.main.dataset.xxxxx"),
    who_height_boys=Input("ri.foundry.main.dataset.xxxxx"),
    who_height_girls=Input("ri.foundry.main.dataset.xxxxx"),
    who_weight_boys=Input("ri.foundry.main.dataset.xxxxx"),
    who_weight_girls=Input("ri.foundry.main.dataset.xxxxx")
)

def compute(input_measurements, output_dataset,
            cdc_height, cdc_weight, cdc_bmi,
            who_bmi_girls, who_bmi_boys,
            who_height_boys, who_height_girls,
            who_weight_boys, who_weight_girls):
    
    # Load input dataframe
    df = input_measurements.dataframe()
    
    # ========================================================================
    # Convert sex to numeric codes (1=Male, 2=Female) - ONCE for entire dataframe
    # ========================================================================
    df = df.withColumn(
        "sex_numeric",
        F.when(F.upper(F.col("sex")) == "MALE", 1)
         .when(F.upper(F.col("sex")) == "FEMALE", 2)
         .when(F.upper(F.col("sex")) == "M", 1)
         .when(F.upper(F.col("sex")) == "F", 2)
         .otherwise(None)
    )
    
    # ========================================================================
    # Prepare CDC reference tables
    # ========================================================================
    cdc_height_df = (cdc_height.dataframe()
                     .withColumnRenamed("Agemos", "age_months")
                     .withColumnRenamed("Sex", "sex")
                     .select("age_months", "sex", "L", "M", "S")
                     .cache())
    
    cdc_weight_df = (cdc_weight.dataframe()
                     .withColumnRenamed("Agemos", "age_months")
                     .withColumnRenamed("Sex", "sex")
                     .select("age_months", "sex", "L", "M", "S")
                     .cache())
    
    cdc_bmi_df = (cdc_bmi.dataframe()
                  .withColumnRenamed("Agemos", "age_months")
                  .withColumnRenamed("Sex", "sex")
                  .select("age_months", "sex", "L", "M", "S")
                  .cache())
    
    # ========================================================================
    # Prepare WHO reference tables (combine boys and girls)
    # ========================================================================
    who_height_df = (
        who_height_boys.dataframe()
        .withColumnRenamed("Month", "age_months")
        .withColumn("sex", F.lit(1))
        .select("age_months", "sex", "L", "M", "S")
        .union(
            who_height_girls.dataframe()
            .withColumnRenamed("Month", "age_months")
            .withColumn("sex", F.lit(2))
            .select("age_months", "sex", "L", "M", "S")
        )
        .cache()
    )
    
    who_weight_df = (
        who_weight_boys.dataframe()
        .withColumnRenamed("Month", "age_months")
        .withColumn("sex", F.lit(1))
        .select("age_months", "sex", "L", "M", "S")
        .union(
            who_weight_girls.dataframe()
            .withColumnRenamed("Month", "age_months")
            .withColumn("sex", F.lit(2))
            .select("age_months", "sex", "L", "M", "S")
        )
        .cache()
    )
    
    who_bmi_df = (
        who_bmi_boys.dataframe()
        .withColumnRenamed("Month", "age_months")
        .withColumn("sex", F.lit(1))
        .select("age_months", "sex", "L", "M", "S")
        .union(
            who_bmi_girls.dataframe()
            .withColumnRenamed("Month", "age_months")
            .withColumn("sex", F.lit(2))
            .select("age_months", "sex", "L", "M", "S")
        )
        .cache()
    )
    
    # ========================================================================
    # Process each time point: diagnosis, 2_years, 5_years
    # ========================================================================
    for time_point in ["diagnosis", "2_years", "5_years"]:
        
        # Calculate age in months for HEIGHT measurement at this time point
        # months_between(end, start) calculates: end - start in months
        # So measurement_date - birth_date gives age at measurement
        if f"height_datetime_at_{time_point}" in df.columns:
            df = df.withColumn(
                f"height_cm_age_months_at_{time_point}",
                F.round(
                    F.months_between(
                        F.col(f"height_datetime_at_{time_point}"), 
                        F.col("date_of_birth")
                    ),
                    2  # Round to 2 decimal places for precision
                )
            )
            # Add flag for negative age (measurement before birth)
            df = df.withColumn(
                f"height_negative_age_flag_at_{time_point}",
                F.when(F.col(f"height_cm_age_months_at_{time_point}") < 0, True).otherwise(None).cast(BooleanType())
            )
        
        # Calculate age in months for WEIGHT measurement at this time point
        if f"weight_datetime_at_{time_point}" in df.columns:
            df = df.withColumn(
                f"weight_kg_age_months_at_{time_point}",
                F.round(
                    F.months_between(
                        F.col(f"weight_datetime_at_{time_point}"), 
                        F.col("date_of_birth")
                    ),
                    2  # Round to 2 decimal places for precision
                )
            )
            # Add flag for negative age (measurement before birth)
            df = df.withColumn(
                f"weight_negative_age_flag_at_{time_point}",
                F.when(F.col(f"weight_kg_age_months_at_{time_point}") < 0, True).otherwise(None).cast(BooleanType())
            )
            
            # Also create age for BMI calculation (use weight datetime as reference)
            df = df.withColumn(
                f"bmi_calculated_age_months_at_{time_point}",
                F.col(f"weight_kg_age_months_at_{time_point}")
            )
            # Add flag for negative age for BMI
            df = df.withColumn(
                f"bmi_calculated_negative_age_flag_at_{time_point}",
                F.when(F.col(f"bmi_calculated_age_months_at_{time_point}") < 0, True).otherwise(None).cast(BooleanType())
            )
        
        # Convert height from inches to cm (1 inch = 2.54 cm)
        if f"height_at_{time_point}" in df.columns:
            df = df.withColumn(
                f"height_cm_at_{time_point}",
                F.when(
                    F.col(f"height_at_{time_point}").isNotNull(),
                    F.col(f"height_at_{time_point}").cast("double") * 2.54
                )
            )
        
        # ====================================================================
        # CORRECTED: Convert weight from OUNCES (oz_av) to kg
        # 1 ounce = 0.0283495231 kg
        # ====================================================================
        if f"weight_at_{time_point}" in df.columns:
            df = df.withColumn(
                f"weight_kg_at_{time_point}",
                F.when(
                    F.col(f"weight_at_{time_point}").isNotNull(),
                    F.col(f"weight_at_{time_point}").cast("double") * 0.0283495231
                )
            )
        
        # Calculate percentiles for height
        df = add_growth_percentiles(df, time_point, "height_cm", cdc_height_df, who_height_df)
        
        # Calculate percentiles for weight
        df = add_growth_percentiles(df, time_point, "weight_kg", cdc_weight_df, who_weight_df)
        
        # Calculate BMI from the converted height (cm) and weight (kg) values
        # BMI = weight(kg) / height(m)^2
        # We calculate this AFTER the percentiles so we can use the converted values
        if f"height_cm_at_{time_point}" in df.columns and f"weight_kg_at_{time_point}" in df.columns:
            df = df.withColumn(
                f"bmi_calculated_at_{time_point}",
                F.when(
                    (F.col(f"height_cm_at_{time_point}").isNotNull()) & 
                    (F.col(f"weight_kg_at_{time_point}").isNotNull()) &
                    (F.col(f"height_cm_at_{time_point}") > 0),  # Avoid division by zero
                    F.col(f"weight_kg_at_{time_point}") / 
                    ((F.col(f"height_cm_at_{time_point}") / 100) ** 2)
                ).otherwise(None)
            )
            
            # Calculate percentiles for BMI using the calculated BMI
            df = add_growth_percentiles(df, time_point, "bmi_calculated", cdc_bmi_df, who_bmi_df)
            
            # Rename BMI percentile and zscore columns to simpler names
            if f"bmi_calculated_percentile_at_{time_point}" in df.columns:
                df = df.withColumnRenamed(
                    f"bmi_calculated_percentile_at_{time_point}",
                    f"bmi_percentile_at_{time_point}"
                ).withColumnRenamed(
                    f"bmi_calculated_zscore_at_{time_point}",
                    f"bmi_zscore_at_{time_point}"
                )
        
        # Rename percentile columns to match original measurement names
        if f"height_cm_percentile_at_{time_point}" in df.columns:
            df = df.withColumnRenamed(
                f"height_cm_percentile_at_{time_point}",
                f"height_percentile_at_{time_point}"
            ).withColumnRenamed(
                f"height_cm_zscore_at_{time_point}",
                f"height_zscore_at_{time_point}"
            )
        
        if f"weight_kg_percentile_at_{time_point}" in df.columns:
            df = df.withColumnRenamed(
                f"weight_kg_percentile_at_{time_point}",
                f"weight_percentile_at_{time_point}"
            ).withColumnRenamed(
                f"weight_kg_zscore_at_{time_point}",
                f"weight_zscore_at_{time_point}"
            )
        
        # Drop temporary cm and kg columns (but keep the negative age flags and debug columns!)
        cols_to_drop = [
            f"height_cm_at_{time_point}",
            f"height_cm_age_months_at_{time_point}",
            f"weight_kg_at_{time_point}",
            f"weight_kg_age_months_at_{time_point}",
            f"bmi_calculated_at_{time_point}",
            f"bmi_calculated_age_months_at_{time_point}"
        ]
        for col in cols_to_drop:
            if col in df.columns:
                df = df.drop(col)
    
    # Drop the temporary sex_numeric column
    df = df.drop("sex_numeric")
    
    # Unpersist cached dataframes to free memory
    cdc_height_df.unpersist()
    cdc_weight_df.unpersist()
    cdc_bmi_df.unpersist()
    who_height_df.unpersist()
    who_weight_df.unpersist()
    who_bmi_df.unpersist()
    
    # Write the result
    output_dataset.write_dataframe(df)