from pyspark.sql import functions as F
from pyspark.sql import Window
from transforms.api import Input, Output, transform

"""
Transform 11: Clinical Outcomes with proper null / 0 / 1 encoding.

OUTCOME ENCODING PRINCIPLE:
    1    = patient HAS the outcome
    0    = patient was ASSESSED and does NOT have the outcome
    null = patient was NOT ASSESSED (not in denominator)

CORRECTIONS APPLIED (v2):
  1. BP parsing uses robust slash-aware helper (same as Transform 10).
  2. Asymmetric ICD override applied CONSISTENTLY to all conditions:
       ICD=1                        → 1  (positive ICD always overrides)
       ICD=0 does NOT override null labs → stays null (not assessed)
  3. Microalbuminuria: asymmetric ICD pattern + confirmation logic using
     repeat lab flags.
  4. Metformin Response: NOT on metformin → null (not in denominator),
     NOT on_med=0 → outcome=0. Patients on metformin + additional meds
     are in the denominator (outcome=0), not excluded.
  5. GLP-1 Response: NOT on GLP1-RA → null (not in denominator).
  6. Insulin Independence: all components null → null; any component
     definitively fails → 0; all pass → 1.
  7. Optimal Glycemic Control: no A1c → null; A1c <7 → 1; A1c ≥7 → 0.

CORRECTIONS APPLIED (v3):
  8. Dyslipidemia: removed HDL < 40 mg/dL criterion. Now defined as
     Non-HDL cholesterol ≥ 145 mg/dL OR ICD diagnosis only.
"""


@transform(
    output=Output("ri.foundry.main.dataset.xxxxx"),
    input_dataset=Input("ri.foundry.main.dataset.xxxxx"),
    omop_measurements=Input("ri.foundry.main.dataset.xxxxx"),
)
def compute(input_dataset, omop_measurements, output):
    df = input_dataset.dataframe()
    omop = omop_measurements.dataframe()

    cols = set(df.columns)

    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================

    def col_or_null(col_name, dtype="double"):
        if col_name in cols:
            return F.col(col_name).cast(dtype)
        return F.lit(None).cast(dtype)

    def to_binary_expr(col_name):
        """Parse a column to binary 1/0/null."""
        if col_name not in cols:
            return F.lit(None).cast("integer")
        return (
            F.when(F.lower(F.col(col_name).cast("string")).isin("1", "yes", "true"), F.lit(1))
             .when(F.lower(F.col(col_name).cast("string")).isin("0", "no", "false"), F.lit(0))
             .otherwise(F.lit(None).cast("integer"))
        )

    def a1c_controlled(a1c_col_name, threshold=7.0):
        """
        1 = A1c < threshold (controlled)
        0 = A1c >= threshold (not controlled)
        null = A1c not available (not assessed)
        """
        if a1c_col_name not in cols:
            return F.lit(None).cast("integer")
        a1c = F.col(a1c_col_name).cast("double")
        return (
            F.when(a1c.isNull(), F.lit(None).cast("integer"))
             .when(a1c < threshold, F.lit(1))
             .otherwise(F.lit(0))
        )

    def parse_bp_percentile(col_name):
        """Parse BP percentile column: 1 if ≥95th, 0 if present but <95th, null if missing."""
        if col_name not in cols:
            return F.lit(None).cast("integer")
        val = F.upper(F.trim(F.col(col_name).cast("string")))
        return (
            F.when(F.col(col_name).isNull(), F.lit(None).cast("integer"))
             .when(
                 val.rlike(r"^95TH")
                 | val.contains(">95")
                 | val.rlike(r">=\s*95"),
                 F.lit(1)
             )
             .otherwise(F.lit(0))
        )

    def parse_diastolic_bp(sys_col_name, dia_col_name):
        """Robustly parse diastolic BP: dedicated column first, then slash extraction."""
        if dia_col_name in cols:
            dedicated = F.col(dia_col_name).cast("double")
        else:
            dedicated = F.lit(None).cast("double")

        if sys_col_name in cols:
            raw = F.col(sys_col_name).cast("string")
            has_slash = raw.contains("/")
            from_slash = (
                F.when(has_slash, F.split(raw, "/").getItem(1).cast("double"))
                 .otherwise(F.lit(None).cast("double"))
            )
        else:
            from_slash = F.lit(None).cast("double")

        return F.coalesce(dedicated, from_slash)

    def icd_lab_outcome(icd_expr, lab_value_expr, lab_threshold_expr, lab_raw_col):
        """
        Combine ICD flag and lab result into proper 1/0/null outcome.

        CRITICAL: lab_available is determined by whether the raw lab column
        is NOT NULL — not by evaluating a boolean expression (which returns
        False instead of null when the input is null).

        Args:
            icd_expr:           Column expression for ICD binary (1/0/null)
            lab_value_expr:     Column expression for the raw lab value (e.g. UACR)
            lab_threshold_expr: Column expression that is True when lab meets
                                the positive threshold (e.g. uacr >= 30)
            lab_raw_col:        The raw lab column used to check availability
                                (must be the actual F.col, not a derived boolean)

        Truth table:
          ICD=1, any labs         → 1
          ICD=0/null, lab present & positive → 1
          ICD=0, lab present & negative      → 0
          ICD=null, lab present & negative   → 0
          ICD=0, lab null                    → null  (not assessed)
          ICD=null, lab null                 → null  (not assessed)
        """
        lab_is_present = lab_raw_col.isNotNull()
        lab_is_positive = lab_is_present & lab_threshold_expr

        return (
            F.when(icd_expr == 1, F.lit(1))                                # ICD positive → 1
             .when(lab_is_positive, F.lit(1))                              # Lab positive → 1
             .when(lab_is_present & ~lab_threshold_expr & (icd_expr == 0),
                   F.lit(0))                                               # Lab neg + ICD neg → 0
             .when(lab_is_present & ~lab_threshold_expr & icd_expr.isNull(),
                   F.lit(0))                                               # Lab neg + no ICD → 0
             .otherwise(F.lit(None).cast("integer"))                       # Not assessed → null
        )

    # =========================================================================
    # STEP 1: PARSE BLOOD PRESSURE COLUMNS
    # =========================================================================

    bp_parse_map = [
        ("systolic_blood_pressure_at_diagnosis",  "diastolic_blood_pressure_at_diagnosis",  "systolic_bp_dx_num",  "diastolic_bp_dx_num"),
        ("systolic_blood_pressure_at_2_years",    "diastolic_blood_pressure_at_2_years",    "systolic_bp_2yr_num", "diastolic_bp_2yr_num"),
        ("systolic_blood_pressure_at_5_years",    "diastolic_blood_pressure_at_5_years",    "systolic_bp_5yr_num", "diastolic_bp_5yr_num"),
    ]

    for src_col, dia_col, sys_out, dia_out in bp_parse_map:
        if src_col in cols:
            raw = F.col(src_col).cast("string")
            has_slash = raw.contains("/")
            df = df.withColumn(
                sys_out,
                F.when(has_slash, F.split(raw, "/").getItem(0).cast("double"))
                 .otherwise(raw.cast("double"))
            )
            df = df.withColumn(dia_out, parse_diastolic_bp(src_col, dia_col))
        else:
            df = df.withColumn(sys_out, F.lit(None).cast("double"))
            df = df.withColumn(dia_out, F.lit(None).cast("double"))

    # =========================================================================
    # STEP 2: OUTCOME_Hypertension
    #
    # Definition:
    #   BP ≥ 95th percentile for age (<13 yr) OR ≥ 130/80 (≥13 yr)
    #   OR documentation of clinical diagnosis (ICD)
    #
    # Encoding:
    #   1    = meets BP criteria OR has ICD
    #   0    = BP assessed & below threshold, ICD negative or null
    #   null = no BP data AND no positive ICD (not assessed)
    # =========================================================================

    age = col_or_null("Age_at_diagnosis_min")

    htn_timepoints = [
        ("sbp_percentile_at_diagnosis", "dbp_percentile_at_diagnosis", "systolic_bp_dx_num",  "diastolic_bp_dx_num",  "Hypertension_diagnosis", "OUTCOME_Hypertension_at_diagnosis"),
        ("sbp_percentile_at_2_years",   "dbp_percentile_at_2_years",   "systolic_bp_2yr_num", "diastolic_bp_2yr_num", "Hypertension_2yr",       "OUTCOME_Hypertension_at_2_years"),
        ("sbp_percentile_at_5_years",   "dbp_percentile_at_5_years",   "systolic_bp_5yr_num", "diastolic_bp_5yr_num", "Hypertension_5yr",       "OUTCOME_Hypertension_at_5_years"),
    ]

    for sbp_pct_col, dbp_pct_col, sys_col, dia_col, icd_col, outcome_col in htn_timepoints:
        sbp_pct = parse_bp_percentile(sbp_pct_col)
        dbp_pct = parse_bp_percentile(dbp_pct_col)
        sbp_num = F.col(sys_col)
        dbp_num = F.col(dia_col)

        # Lab-based BP criterion
        bp_crit = F.when(
            age < 13,
            F.when(
                sbp_pct.isNull() & dbp_pct.isNull(),
                F.lit(None).cast("integer")
            ).when(
                (sbp_pct == 1) | (dbp_pct == 1),
                F.lit(1)
            ).otherwise(F.lit(0))
        ).otherwise(
            F.when(
                sbp_num.isNull() & dbp_num.isNull(),
                F.lit(None).cast("integer")
            ).when(
                (sbp_num >= 130) | (dbp_num >= 80),
                F.lit(1)
            ).otherwise(F.lit(0))
        )

        icd_crit = to_binary_expr(icd_col)

        bp_raw = bp_crit  # null when no BP, 0 or 1 when assessed

        df = df.withColumn(
            outcome_col,
            icd_lab_outcome(icd_crit, bp_crit, bp_crit == 1, bp_crit)
        )

    # =========================================================================
    # STEP 3: OUTCOME_Dyslipidemia
    #
    # Definition (v3 — updated):
    #   Non-HDL cholesterol ≥ 145 mg/dL (non-fasting)
    #   OR ICD for dyslipidemia/hypertriglyceridemia/hypercholesterolemia
    #
    # NOTE: HDL < 40 mg/dL criterion has been REMOVED per collaborator input.
    #
    # Non-HDL = Total cholesterol − HDL cholesterol.
    # Both total and HDL must be present to compute non-HDL.
    # =========================================================================

    dyslip_timepoints = [
        ("total_cholesterol_at_diagnosis", "hdl_cholesterol_at_diagnosis", "Dyslipidemia_diagnosis", "OUTCOME_Dyslipidemia_at_diagnosis"),
        ("total_cholesterol_at_2_years",   "hdl_cholesterol_at_2_years",   "Dyslipidemia_2yr",       "OUTCOME_Dyslipidemia_at_2_years"),
        ("total_cholesterol_at_5_years",   "hdl_cholesterol_at_5_years",   "Dyslipidemia_5yr",       "OUTCOME_Dyslipidemia_at_5_years"),
    ]

    for total_col, hdl_col_name, icd_col, outcome_col in dyslip_timepoints:
        total   = col_or_null(total_col)
        hdl     = col_or_null(hdl_col_name)
        non_hdl = total - hdl

        # Non-HDL requires BOTH total and HDL to be present
        lab_available = total.isNotNull() & hdl.isNotNull()
        lab_positive  = (non_hdl >= 145)

        icd_crit = to_binary_expr(icd_col)

        # Build a raw column that is non-null when lipid labs are computable
        lab_raw_marker = F.when(lab_available, F.lit(1)).otherwise(F.lit(None).cast("integer"))

        df = df.withColumn(
            outcome_col,
            icd_lab_outcome(icd_crit, lab_raw_marker, lab_positive, lab_raw_marker)
        )

    # =========================================================================
    # STEP 4: OUTCOME_Microalbuminuria
    #
    # Definition:
    #   UACR ≥ 30 mg/g OR ICD for microalbuminuria
    #
    # Confirmation logic (applied AFTER repeat lab flags are computed):
    #   If elevated UACR and additional measurements within 6mo exist:
    #     need ≥2 abnormal results to confirm
    #   If elevated UACR and NO additional measurements:
    #     single elevated = positive
    #
    # Uses asymmetric ICD pattern (same as HTN/dyslipidemia).
    # =========================================================================

    UACR_THRESHOLD = 30  # mg/g

    micro_timepoints = [
        ("urine_microalbumin_creatinine_ratio_at_diagnosis", "Microalbuminuria_diagnosis", "OUTCOME_Microalbuminuria_at_diagnosis"),
        ("urine_microalbumin_creatinine_ratio_at_2_years",   "Microalbuminuria_2yr",       "OUTCOME_Microalbuminuria_at_2_years"),
        ("urine_microalbumin_creatinine_ratio_at_5_years",   "Microalbuminuria_5yr",       "OUTCOME_Microalbuminuria_at_5_years"),
    ]

    for uacr_col, icd_col, outcome_col in micro_timepoints:
        uacr = col_or_null(uacr_col)

        icd_crit = to_binary_expr(icd_col)

        uacr_raw = F.col(uacr_col).cast("double") if uacr_col in cols else F.lit(None).cast("double")
        uacr_threshold = (uacr_raw >= UACR_THRESHOLD)

        # Preliminary outcome — will be refined in Step 5b
        df = df.withColumn(
            outcome_col,
            icd_lab_outcome(icd_crit, uacr_raw, uacr_threshold, uacr_raw)
        )

    # =========================================================================
    # STEP 5a: Microalbuminuria Repeat Lab Test (from OMOP)
    # =========================================================================

    DIAGNOSIS_DATE_COL = "date_of_diagnosis"
    repeat_output_cols = [
        "microalbuminuria_repeat_lab_test_at_diagnosis",
        "microalbuminuria_repeat_lab_test_at_2_years",
        "microalbuminuria_repeat_lab_test_at_5_years",
    ]

    if "OMOP_ID" in cols and DIAGNOSIS_DATE_COL in cols:
        micro_omop = (
            omop
            .filter(F.col("measurement_type") == "URINE_MICROALBUMIN")
            .select(
                F.col("PERSON_ID").cast("long").alias("PERSON_ID"),
                F.col("MEASUREMENT_DATETIME").cast("date").alias("meas_date"),
                F.col("VALUE_SOURCE_VALUE").cast("double").alias("uacr_value"),
            )
            .dropna(subset=["PERSON_ID", "meas_date"])
        )

        patient_anchors = (
            df.select(
                F.col("OMOP_ID").cast("long").alias("PERSON_ID"),
                F.col(DIAGNOSIS_DATE_COL).cast("date").alias("dx_date")
            )
            .dropna(subset=["PERSON_ID", "dx_date"])
            .withColumn("ref_date_diagnosis", F.col("dx_date"))
            .withColumn("ref_date_2yr",       F.date_add(F.col("dx_date"), 730))
            .withColumn("ref_date_5yr",       F.date_add(F.col("dx_date"), 1825))
        )

        def build_repeat_flag(patient_anchors, micro_omop, ref_date_col, out_col):
            """
            For each patient, count distinct UACR measurements within ±180 days
            of the reference date, and count how many of those are abnormal (≥30).
            """
            windowed = (
                patient_anchors
                .join(micro_omop, on="PERSON_ID", how="left")
                .filter(
                    F.col("meas_date").isNull() |
                    (F.abs(F.datediff(F.col("meas_date"), F.col(ref_date_col))) <= 180)
                )
            )

            return (
                windowed
                .groupBy("PERSON_ID")
                .agg(
                    F.countDistinct("meas_date").alias("n_meas_in_window"),
                    F.sum(
                        F.when(
                            F.col("uacr_value") >= UACR_THRESHOLD, F.lit(1)
                        ).otherwise(F.lit(0))
                    ).alias("n_abnormal_in_window"),
                )
                .withColumn(
                    out_col,
                    F.when(F.col("n_meas_in_window") >= 2, F.lit(1))
                     .otherwise(F.lit(0))
                )
                .withColumn(
                    f"{out_col}_n_abnormal",
                    F.col("n_abnormal_in_window")
                )
                .select("PERSON_ID", out_col, f"{out_col}_n_abnormal")
            )

        timepoint_configs = [
            ("ref_date_diagnosis", "microalbuminuria_repeat_lab_test_at_diagnosis"),
            ("ref_date_2yr",       "microalbuminuria_repeat_lab_test_at_2_years"),
            ("ref_date_5yr",       "microalbuminuria_repeat_lab_test_at_5_years"),
        ]

        for ref_col, out_col in timepoint_configs:
            repeat_flags = build_repeat_flag(patient_anchors, micro_omop, ref_col, out_col)
            df = (
                df.join(
                    repeat_flags,
                    on=(F.col("OMOP_ID").cast("long") == repeat_flags["PERSON_ID"]),
                    how="left"
                )
                .drop(repeat_flags["PERSON_ID"])
                .withColumn(out_col, F.coalesce(F.col(out_col), F.lit(0)).cast("integer"))
                .withColumn(
                    f"{out_col}_n_abnormal",
                    F.coalesce(F.col(f"{out_col}_n_abnormal"), F.lit(0)).cast("integer")
                )
            )
    else:
        for out_col in repeat_output_cols:
            df = df.withColumn(out_col, F.lit(None).cast("integer"))
            df = df.withColumn(f"{out_col}_n_abnormal", F.lit(None).cast("integer"))

    # =========================================================================
    # STEP 5b: Refine Microalbuminuria with confirmation logic
    #
    # If a patient has an elevated UACR at a timepoint:
    #   - Check if there are additional UACR measurements within 6 months
    #   - If additional measurements exist: need ≥2 abnormal results → positive
    #   - If NO additional measurements: single elevated → positive
    #
    # If the patient was already positive by ICD (outcome=1 from ICD alone),
    # that stays 1 regardless.
    # =========================================================================

    micro_refinement = [
        ("OUTCOME_Microalbuminuria_at_diagnosis", "microalbuminuria_repeat_lab_test_at_diagnosis",
         "urine_microalbumin_creatinine_ratio_at_diagnosis", "Microalbuminuria_diagnosis"),
        ("OUTCOME_Microalbuminuria_at_2_years", "microalbuminuria_repeat_lab_test_at_2_years",
         "urine_microalbumin_creatinine_ratio_at_2_years", "Microalbuminuria_2yr"),
        ("OUTCOME_Microalbuminuria_at_5_years", "microalbuminuria_repeat_lab_test_at_5_years",
         "urine_microalbumin_creatinine_ratio_at_5_years", "Microalbuminuria_5yr"),
    ]

    for outcome_col, repeat_flag_col, uacr_col_name, icd_col in micro_refinement:
        icd_crit = to_binary_expr(icd_col)
        repeat_flag = F.col(repeat_flag_col)  # 1 = has ≥2 measurements in window
        n_abnormal = F.col(f"{repeat_flag_col}_n_abnormal")

        uacr_raw = F.col(uacr_col_name).cast("double") if uacr_col_name in cols else F.lit(None).cast("double")
        uacr_elevated = (uacr_raw >= UACR_THRESHOLD)

        lab_positive_refined = (
            F.when(
                uacr_raw.isNull(), F.lit(False)  # no UACR → not positive
            ).when(
                ~uacr_elevated, F.lit(False)     # UACR below threshold → not positive
            ).when(
                # Elevated + additional measurements exist
                (repeat_flag == 1),
                (n_abnormal >= 2)  # need ≥2 abnormal total
            ).otherwise(
                # Elevated + no additional measurements → single counts
                F.lit(True)
            )
        )

        # Recompute outcome with refined lab logic
        df = df.withColumn(
            outcome_col,
            icd_lab_outcome(icd_crit, uacr_raw, lab_positive_refined, uacr_raw)
        )

    # =========================================================================
    # STEP 6: OUTCOME_Optimal_Glycemic_Control
    #
    # Definition: A1c < 7%
    # Encoding:
    #   1    = A1c < 7%
    #   0    = A1c ≥ 7%
    #   null = A1c not available
    # =========================================================================

    for a1c_col, outcome_col in [
        ("a1c_diagnosis", "OUTCOME_Optimal_Glycemic_Control_at_diagnosis"),
        ("a1c_2yr",       "OUTCOME_Optimal_Glycemic_Control_at_2_years"),
        ("a1c_5yr",       "OUTCOME_Optimal_Glycemic_Control_at_5_years"),
    ]:
        df = df.withColumn(outcome_col, a1c_controlled(a1c_col))

    # =========================================================================
    # STEP 7: OUTCOME_Insulin_Independence
    #
    # Definition: NOT on insulin AND A1c <7% AND no DKA
    # Encoding:
    #   1    = meets all three criteria
    #   0    = assessed but fails at least one criterion
    #   null = cannot determine (all relevant data missing)
    #
    # Logic detail:
    #   If ANY criterion definitively fails (insulin=1, A1c≥7, DKA=1) → 0
    #   If ALL criteria pass → 1
    #   If no criterion fails but some are null → null (can't confirm)
    # =========================================================================

    for insulin_col, a1c_col, dka_col, outcome_col in [
        ("Insulins_diagnosis", "a1c_diagnosis", "DKA_diagnosis", "OUTCOME_Insulin_Independence_at_diagnosis"),
        ("Insulins_2yr",       "a1c_2yr",       "DKA_2yr",       "OUTCOME_Insulin_Independence_at_2_years"),
        ("Insulins_5yr",       "a1c_5yr",       "DKA_5yr",       "OUTCOME_Insulin_Independence_at_5_years"),
    ]:
        on_insulin = to_binary_expr(insulin_col)
        a1c_ok     = a1c_controlled(a1c_col)
        dka_bin    = to_binary_expr(dka_col)

        any_fails = (
            (on_insulin == 1) |  # on insulin → fail
            (a1c_ok == 0) |      # A1c ≥ 7 → fail
            (dka_bin == 1)       # has DKA → fail
        )

        all_pass = (
            (on_insulin == 0) &
            (a1c_ok == 1) &
            ((dka_bin == 0) | dka_bin.isNull())  # no DKA or DKA unknown (generous)
        )

        any_assessed = (
            on_insulin.isNotNull() | a1c_ok.isNotNull() | dka_bin.isNotNull()
        )

        df = df.withColumn(
            outcome_col,
            F.when(any_fails, F.lit(0))
             .when(all_pass, F.lit(1))
             .when(~any_assessed, F.lit(None).cast("integer"))  # nothing assessed
             .otherwise(F.lit(None).cast("integer"))  # partial data, can't confirm
        )

    # =========================================================================
    # STEP 8: OUTCOME_Metformin_Response
    #
    # Definition: Achieving A1c <7% with metformin ALONE
    # Denominator: ALL patients on metformin (including those on additional meds)
    #
    # Encoding:
    #   1    = on metformin alone AND A1c <7%
    #   0    = on metformin but: A1c ≥7%, OR on additional meds
    #          (these patients are in the denominator but did not "respond")
    #   null = NOT on metformin (not in denominator)
    #        = on metformin but A1c not available (can't assess response)
    # =========================================================================

    additional_med_cols_map = {
        "diagnosis": [
            "Insulins_diagnosis", "GLP1_agonists_diagnosis",
            "Sulfonylureas_diagnosis", "Thiazolidinediones_diagnosis",
            "SGLT2_inhibitors_diagnosis", "DPP4_inhibitors_diagnosis",
            "Meglitinides_diagnosis", "Alpha_glucosidase_inhibitors_diagnosis",
            "Amylin_analogue_diagnosis",
        ],
        "2yr": [
            "Insulins_2yr", "GLP1_agonists_2yr",
            "Sulfonylureas_2yr", "Thiazolidinediones_2yr",
            "SGLT2_inhibitors_2yr", "DPP4_inhibitors_2yr",
            "Meglitinides_2yr", "Alpha_glucosidase_inhibitors_2yr",
            "Amylin_analogue_2yr",
        ],
        "5yr": [
            "Insulins_5yr", "GLP1_agonists_5yr",
            "Sulfonylureas_5yr", "Thiazolidinediones_5yr",
            "SGLT2_inhibitors_5yr", "DPP4_inhibitors_5yr",
            "Meglitinides_5yr", "Alpha_glucosidase_inhibitors_5yr",
            "Amylin_analogue_5yr",
        ],
    }

    for med_col, a1c_col, tp_key, outcome_col in [
        ("Biguanide_diagnosis", "a1c_diagnosis", "diagnosis", "OUTCOME_Metformin_Response_at_diagnosis"),
        ("Biguanide_2yr",       "a1c_2yr",       "2yr",       "OUTCOME_Metformin_Response_at_2_years"),
        ("Biguanide_5yr",       "a1c_5yr",       "5yr",       "OUTCOME_Metformin_Response_at_5_years"),
    ]:
        on_metformin = to_binary_expr(med_col)
        a1c_ok = a1c_controlled(a1c_col)

        addl_cols = additional_med_cols_map.get(tp_key, [])
        existing_addl_cols = [c for c in addl_cols if c in cols]

        if existing_addl_cols:
            on_additional = F.lit(0).cast("integer")
            for addl_col in existing_addl_cols:
                addl_bin = to_binary_expr(addl_col)
                on_additional = F.when(addl_bin == 1, F.lit(1)).otherwise(on_additional)
        else:
            on_additional = F.lit(None).cast("integer")

        df = df.withColumn(
            outcome_col,
            F.when(
                (on_metformin != 1) | on_metformin.isNull(),
                F.lit(None).cast("integer")
            ).when(
                a1c_ok.isNull(),
                F.lit(None).cast("integer")
            ).when(
                on_additional == 1,
                F.lit(0)  # on other meds → denominator only, always 0
            ).when(
                a1c_ok == 1,
                F.lit(1)  # metformin alone + A1c <7 → response
            ).otherwise(
                F.lit(0)  # metformin alone + A1c ≥7 → no response
            )
        )

    # =========================================================================
    # STEP 9: OUTCOME_GLP1RA_Response
    #
    # Definition: Achieving A1c <7% with GLP1-RA
    # Denominator: ALL patients on GLP1-RA at a given time point
    #
    # Encoding:
    #   1    = on GLP1-RA AND A1c <7%
    #   0    = on GLP1-RA but A1c ≥7%
    #   null = NOT on GLP1-RA (not in denominator)
    #        = on GLP1-RA but A1c not available
    # =========================================================================

    for med_col, a1c_col, outcome_col in [
        ("GLP1_agonists_diagnosis", "a1c_diagnosis", "OUTCOME_GLP1RA_Response_at_diagnosis"),
        ("GLP1_agonists_2yr",       "a1c_2yr",       "OUTCOME_GLP1RA_Response_at_2_years"),
        ("GLP1_agonists_5yr",       "a1c_5yr",       "OUTCOME_GLP1RA_Response_at_5_years"),
    ]:
        on_glp1 = to_binary_expr(med_col)
        a1c_ok  = a1c_controlled(a1c_col)

        df = df.withColumn(
            outcome_col,
            F.when(
                (on_glp1 != 1) | on_glp1.isNull(),
                F.lit(None).cast("integer")
            ).when(
                a1c_ok.isNull(),
                F.lit(None).cast("integer")
            ).when(
                a1c_ok == 1,
                F.lit(1)
            ).otherwise(
                F.lit(0)
            )
        )

    # =========================================================================
    # STEP 10: EXCLUDE SPECIFIC PATIENT
    # =========================================================================

    df = df.filter(F.col("mrn").cast("string") != "REDACTED_MRN")

    output.write_dataframe(df)