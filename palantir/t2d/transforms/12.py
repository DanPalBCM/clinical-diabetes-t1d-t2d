from pyspark.sql import functions as F
from transforms.api import Input, Output, transform


@transform(
    output=Output("ri.foundry.main.dataset.xxxxx"),
    input_dataset=Input("ri.foundry.main.dataset.xxxxx"),
)
def compute(input_dataset, output):
    df = input_dataset.dataframe()

    # =========================================================================
    # NO A1C FILTER — Set 1 uses the full cohort.
    # =========================================================================

    # =========================================================================
    # PREPROCESSING: parse BP columns to numeric double.
    # =========================================================================
    for bp_col in [
        "systolic_blood_pressure_at_diagnosis",
        "systolic_blood_pressure_at_2_years",
        "systolic_blood_pressure_at_5_years",
    ]:
        raw = F.col(bp_col).cast("string")
        has_slash = raw.contains("/")
        df = df.withColumn(
            "_parsed_" + bp_col,
            F.when(has_slash, F.split(raw, "/").getItem(0).cast("double"))
             .otherwise(raw.cast("double"))
        )

    rows = []

    def count_when(condition):
        return df.filter(condition).count()

    # =========================================================================
    # NULL-SAFE HELPERS
    #
    # Spark null-trap: (null != 1) evaluates to NULL, not True.
    # In a filter, NULL is treated as False, so patients with null lab_crit
    # were silently dropped from the "ICD only" bucket.
    #
    # Fix: coalesce(expr, 0) before any != comparison so null → 0.
    #   lab_is_1(e)     → e == 1               (null-safe: null → False ✓)
    #   lab_is_not_1(e) → coalesce(e,0) != 1   (null-safe: null → 0 != 1 → True ✓)
    # =========================================================================
    def lab_is_1(expr):
        return expr == 1

    def lab_is_not_1(expr):
        # coalesce treats null as 0, so null lab_crit correctly reads as "not 1"
        return F.coalesce(expr, F.lit(0)) != 1

    # =========================================================================
    # HYPERTENSION
    # =========================================================================
    htn_timepoints = [
        (
            "OUTCOME_Hypertension_at_diagnosis",
            "systolic_blood_pressure_at_least_3_bp_diagnosis",
            "sbp_percentile_at_diagnosis", "dbp_percentile_at_diagnosis",
            "_parsed_systolic_blood_pressure_at_diagnosis", "diastolic_bp_dx_num",
            "Hypertension_diagnosis", "at_diagnosis",
        ),
        (
            "OUTCOME_Hypertension_at_2_years",
            "systolic_blood_pressure_at_least_3_bp_2_years",
            "sbp_percentile_at_2_years", "dbp_percentile_at_2_years",
            "_parsed_systolic_blood_pressure_at_2_years", "diastolic_bp_2yr_num",
            "Hypertension_2yr", "at_2_years",
        ),
        (
            "OUTCOME_Hypertension_at_5_years",
            "systolic_blood_pressure_at_least_3_bp_5_years",
            "sbp_percentile_at_5_years", "dbp_percentile_at_5_years",
            "_parsed_systolic_blood_pressure_at_5_years", "diastolic_bp_5yr_num",
            "Hypertension_5yr", "at_5_years",
        ),
    ]

    def sbp_pct_flag(col_name):
        val = F.upper(F.trim(F.col(col_name).cast("string")))
        return (
            F.when(F.col(col_name).isNull(), F.lit(None).cast("integer"))
             .when(val.rlike(r"^95TH") | val.contains(">95") | val.rlike(r">=\s*95"), F.lit(1))
             .otherwise(F.lit(0))
        )

    age = F.col("Age_at_diagnosis_min").cast("double")

    for (
        outcome_col, bp_flag_col, sbp_pct_col, dbp_pct_col,
        sys_parsed_col, dia_parsed_col, icd_col, timepoint
    ) in htn_timepoints:

        s1_mask = (
            F.col(outcome_col).cast("integer").isin(0, 1) &
            (F.col(bp_flag_col) == True)
        )

        sbp_num = F.col(sys_parsed_col)
        dbp_num = F.col(dia_parsed_col)
        sbp_pct = sbp_pct_flag(sbp_pct_col)
        dbp_pct = sbp_pct_flag(dbp_pct_col)

        lab_crit = (
            F.when(
                age < 13,
                F.when(F.col(sbp_pct_col).isNull() & F.col(dbp_pct_col).isNull(),
                       F.lit(None).cast("integer"))
                 .when((sbp_pct == 1) | (dbp_pct == 1), F.lit(1))
                 .otherwise(F.lit(0))
            ).otherwise(
                F.when(sbp_num.isNull() & dbp_num.isNull(),
                       F.lit(None).cast("integer"))
                 .when((sbp_num >= 130) | (dbp_num >= 80), F.lit(1))
                 .otherwise(F.lit(0))
            )
        )

        # Direct integer cast — works for Integer, Double (1.0→1), Boolean
        icd = F.col(icd_col).cast("integer")

        outcome_flag  = F.col(outcome_col).cast("integer")
        positive_mask = s1_mask & (outcome_flag == 1)

        for label, condition in [
            ("Set1: total denominator (outcome measured + BP flag)",
             s1_mask),
            ("Set1: OUTCOME = 1 (all positive)",
             s1_mask & (outcome_flag == 1)),
            ("Set1: OUTCOME = 0 (all negative)",
             s1_mask & (outcome_flag == 0)),

            # Mutually exclusive, exhaustive decomposition of positives.
            # lab_is_not_1() uses coalesce so null lab_crit → treated as "not 1" (True).
            ("Set1 positives: Lab only (lab=1, ICD=0)",
             positive_mask & lab_is_1(lab_crit) & (icd != 1)),
            ("Set1 positives: ICD only (ICD=1, lab not 1 or null)",
             positive_mask & (icd == 1) & lab_is_not_1(lab_crit)),
            ("Set1 positives: Both lab AND ICD",
             positive_mask & lab_is_1(lab_crit) & (icd == 1)),
            ("Set1 positives: OUTCOME=1 but neither lab=1 nor ICD=1 [should be 0]",
             positive_mask & lab_is_not_1(lab_crit) & (icd != 1)),

            ("Set1 negatives: OUTCOME=0 but ICD=1 [should be 0]",
             s1_mask & (outcome_flag == 0) & (icd == 1)),

            ("Set1: lab_crit = 1 (meets BP threshold)",
             s1_mask & lab_is_1(lab_crit)),
            ("Set1: ICD = 1",
             s1_mask & (icd == 1)),
        ]:
            rows.append({"outcome": "Hypertension", "timepoint": timepoint,
                         "criterion": label, "n": count_when(condition)})

    # =========================================================================
    # DYSLIPIDEMIA
    # =========================================================================
    dyslip_timepoints = [
        ("OUTCOME_Dyslipidemia_at_diagnosis",
         "total_cholesterol_at_diagnosis", "hdl_cholesterol_at_diagnosis",
         "Dyslipidemia_diagnosis", "at_diagnosis"),
        ("OUTCOME_Dyslipidemia_at_2_years",
         "total_cholesterol_at_2_years", "hdl_cholesterol_at_2_years",
         "Dyslipidemia_2yr", "at_2_years"),
        ("OUTCOME_Dyslipidemia_at_5_years",
         "total_cholesterol_at_5_years", "hdl_cholesterol_at_5_years",
         "Dyslipidemia_5yr", "at_5_years"),
    ]

    for outcome_col, total_col, hdl_col_name, icd_col, timepoint in dyslip_timepoints:

        total   = F.col(total_col).cast("double")
        hdl     = F.col(hdl_col_name).cast("double")
        non_hdl = total - hdl

        s1_mask = F.col(outcome_col).cast("integer").isin(0, 1)

        lab_crit = (
            F.when(total.isNull() & hdl.isNull(), F.lit(None).cast("integer"))
             .when((non_hdl >= 145) | (hdl < 40), F.lit(1))
             .otherwise(F.lit(0))
        )

        icd = F.col(icd_col).cast("integer")

        outcome_flag  = F.col(outcome_col).cast("integer")
        positive_mask = s1_mask & (outcome_flag == 1)

        for label, condition in [
            ("Set1: total denominator (outcome measured)",
             s1_mask),
            ("Set1: OUTCOME = 1 (all positive)",
             s1_mask & (outcome_flag == 1)),
            ("Set1: OUTCOME = 0 (all negative)",
             s1_mask & (outcome_flag == 0)),

            ("Set1 positives: Lab only (lab=1, ICD=0)",
             positive_mask & lab_is_1(lab_crit) & (icd != 1)),
            ("Set1 positives: ICD only (ICD=1, lab not 1 or null)",
             positive_mask & (icd == 1) & lab_is_not_1(lab_crit)),
            ("Set1 positives: Both lab AND ICD",
             positive_mask & lab_is_1(lab_crit) & (icd == 1)),
            ("Set1 positives: OUTCOME=1 but neither lab=1 nor ICD=1 [should be 0]",
             positive_mask & lab_is_not_1(lab_crit) & (icd != 1)),

            ("Set1 negatives: OUTCOME=0 but ICD=1 [should be 0]",
             s1_mask & (outcome_flag == 0) & (icd == 1)),

            ("Set1: non-HDL >= 145 (meets threshold)",
             s1_mask & (non_hdl >= 145)),
            ("Set1: HDL < 40 (meets threshold)",
             s1_mask & (hdl < 40)),
            ("Set1: lab_crit = 1 (meets either lab threshold)",
             s1_mask & lab_is_1(lab_crit)),
            ("Set1: ICD = 1",
             s1_mask & (icd == 1)),
        ]:
            rows.append({"outcome": "Dyslipidemia", "timepoint": timepoint,
                         "criterion": label, "n": count_when(condition)})

    # =========================================================================
    # WRITE OUTPUT
    # =========================================================================
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType

    schema = StructType([
        StructField("outcome",   StringType(),  True),
        StructField("timepoint", StringType(),  True),
        StructField("criterion", StringType(),  True),
        StructField("n",         IntegerType(), True),
    ])

    spark = df.sql_ctx.sparkSession
    result = spark.createDataFrame(rows, schema=schema)
    output.write_dataframe(result)