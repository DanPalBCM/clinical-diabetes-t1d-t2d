"""
Transform 1 – Patient Visit Grid (v2)
=======================================
Same as original except outputs to new dataset.
Keeps all SDOH columns raw; binarization happens in Transform 6.
"""

from pyspark.sql import functions as F
from pyspark.sql.types import DateType
from transforms.api import Input, Output, transform

VISITS = []
for i in range(10):
    start_days = i * 182
    end_days = (i + 1) * 182
    mid_days = (start_days + end_days) // 2
    VISITS.append((f"v{i + 1}", start_days, end_days, mid_days))

# Manual per-patient corrections (duplicate/mismatched records to drop, plus one
# diagnosis-date/age fix below). The real MRNs are PHI and are redacted from this
# public copy -- REDACTED_MRN_1/2/3 are placeholders, not real identifiers. When
# running this transform inside Foundry against the real dataset, supply the
# actual MRNs via a private, non-committed override (e.g. an untracked local
# module or a Foundry-side parameter), not by editing this file in place.
PATIENTS_TO_REMOVE = ["REDACTED_MRN_1", "REDACTED_MRN_2"]
DIAGNOSIS_DATE_CORRECTION_MRN = "REDACTED_MRN_3"


@transform(
    output=Output("/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d/transform_1_t2d_v2"),
    a1c_dataset=Input("ri.foundry.main.dataset.e3a1679b-3346-4b2d-a8df-2c584df29293"),
    demographics=Input("ri.foundry.main.dataset.6d14e466-1f30-47c9-a654-c3fc0d5e18cf"),
    crossref=Input("ri.foundry.main.dataset.37a797f5-1004-43bf-87e2-e90dff75c50e"),
    socio_factors=Input("ri.foundry.main.dataset.c432ae23-98c6-4ef2-b41b-2c043b251bc1"),
)
def compute(a1c_dataset, demographics, crossref, socio_factors, output):
    a1c = a1c_dataset.dataframe()
    demo = demographics.dataframe()
    xref = crossref.dataframe()
    socio = socio_factors.dataframe()

    # STEP 0: Manual corrections
    a1c = a1c.filter(~F.col("mrn").cast("string").isin(PATIENTS_TO_REMOVE))
    a1c = a1c.withColumn(
        "date_of_diagnosis",
        F.when(
            F.col("mrn").cast("string") == DIAGNOSIS_DATE_CORRECTION_MRN,
            F.to_date(F.lit("1900-01-01")),  # placeholder; see redaction note above
        ).otherwise(F.col("date_of_diagnosis")),
    ).withColumn(
        "age_at_diagnosis",
        F.when(
            F.col("mrn").cast("string") == DIAGNOSIS_DATE_CORRECTION_MRN,
            F.lit(-1),
        ).otherwise(F.col("age_at_diagnosis")),
    )

    # STEP 1: Unique patients via groupBy + min()
    patients = a1c.groupBy(F.col("mrn").cast("string").alias("mrn")).agg(
        F.min("age_at_diagnosis").cast("integer").alias("age_at_diagnosis"),
        F.min("date_of_diagnosis").cast("date").alias("date_of_diagnosis"),
    )

    # STEP 2: OMOP_ID
    xref_clean = (
        xref.select(
            F.col("PAT_MRN_ID").cast("string").alias("mrn"),
            F.col("PEDSNET_ID").cast("long").alias("OMOP_ID"),
        )
        .filter(F.col("mrn").isNotNull() & F.col("OMOP_ID").isNotNull())
        .dropDuplicates(["mrn"])
    )
    patients = patients.join(xref_clean, "mrn", "left")

    # STEP 3: Demographics
    demo_clean = (
        demo.select(
            F.col("mrn").cast("string").alias("mrn"),
            F.col("date_of_birth").cast("date").alias("date_of_birth"),
            F.col("sex"),
            F.col("ethnic_group"),
            F.col("language"),
            F.col("patient_race"),
        )
        .filter(F.col("mrn").isNotNull())
        .dropDuplicates(["mrn"])
    )
    demo_clean = demo_clean.withColumn(
        "ethnic_group", F.when(F.col("ethnic_group") == "Hispanic or Latino", "Hispanic or Latino").otherwise("Other")
    )
    demo_clean = demo_clean.withColumn(
        "language",
        F.when(F.col("language") == "English", "English")
        .when(F.col("language") == "Spanish", "Spanish")
        .otherwise("Other"),
    )
    demo_clean = demo_clean.withColumn(
        "patient_race",
        F.when(F.array_contains(F.col("patient_race"), "White"), "White")
        .when(F.array_contains(F.col("patient_race"), "Black or African American"), "Black or African American")
        .when(F.array_contains(F.col("patient_race"), "Asian"), "Asian")
        .otherwise("Other"),
    )
    patients = patients.join(demo_clean, "mrn", "left")

    # STEP 4: SDOH (static) – extract raw values; binarization in Transform 6
    socio_clean = socio.withColumn("json_clean", F.trim(F.regexp_replace(F.col("val_llm"), "```json|```", "")))
    binary_factors = [
        ("socio_adverse_childhood_experience", "$.binary_factors.adverse_childhood_experience.value"),
        ("socio_alcohol_abuse", "$.binary_factors.alcohol_abuse.value"),
        ("socio_drug_substance_abuse", "$.binary_factors.drug_substance_abuse.value"),
        ("socio_food_insecurity", "$.binary_factors.food_insecurity.value"),
        ("socio_housing_instability", "$.binary_factors.housing_instability.value"),
        ("socio_physical_sexual_abuse", "$.binary_factors.physical_sexual_abuse.value"),
        ("socio_smoking", "$.binary_factors.smoking.value"),
        ("socio_transportation_barrier", "$.binary_factors.transportation_barrier.value"),
    ]
    for col_name, json_path in binary_factors:
        extracted = F.get_json_object(F.col("json_clean"), json_path)
        socio_clean = socio_clean.withColumn(
            col_name,
            F.when(extracted == "true", 1).when(extracted == "false", 0).otherwise(F.lit(None).cast("integer")),
        )
    categorical_factors = [
        ("socio_education_level_parents_guardian", "$.categorical_factors.education_level_parents.value"),
        ("socio_employment_status_parents_guardian", "$.categorical_factors.employment_status_parents.value"),
        ("socio_financial_strain", "$.categorical_factors.financial_strain.value"),
        ("socio_insurance_status", "$.categorical_factors.insurance_status.value"),
        ("socio_physical_activity", "$.categorical_factors.physical_activity.value"),
        ("socio_social_family_support", "$.categorical_factors.social_family_support.value"),
    ]
    for col_name, json_path in categorical_factors:
        socio_clean = socio_clean.withColumn(col_name, F.get_json_object(F.col("json_clean"), json_path))

    socio_cols = ["mrn"] + [c for c, _ in binary_factors] + [c for c, _ in categorical_factors]
    socio_final = socio_clean.select(socio_cols).withColumn("mrn", F.col("mrn").cast("string"))
    patients = patients.join(socio_final, "mrn", "left")

    # STEP 5: Visit windows
    for label, start_d, end_d, mid_d in VISITS:
        patients = (
            patients.withColumn(f"target_date_{label}", F.date_add("date_of_diagnosis", mid_d).cast(DateType()))
            .withColumn(f"window_start_{label}", F.date_add("date_of_diagnosis", start_d).cast(DateType()))
            .withColumn(f"window_end_{label}", F.date_add("date_of_diagnosis", end_d).cast(DateType()))
        )

    output.write_dataframe(patients)


"""
Transform 2 – Temporal Measurements (v2)
==========================================
Changes vs v1:
  - A1c capped at 14 (values >= 14 → 14)
  - EGFR, SERUM_CYSTATIN_C removed
  - GAD65_ANTIBODY, INSULIN_ANTIBODY, ZNT8_ANTIBODY removed
  - UACR_RATIO capped at 300 mg/g (sentinel/extreme values)
  - HEIGHT, WEIGHT, BMI_CALCULATED, all percentiles/z-scores DROPPED
    except BMI_ZSCORE (which is kept)

Output: OMOP_ID | measurement_type | visit | value | meas_date
"""

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import pandas_udf
import pandas as pd
from transforms.api import Input, Output, transform

NUM_VISITS = 10

# Measurement types to EXCLUDE from the final output
EXCLUDE_MEASUREMENT_TYPES = {
    "EGFR",
    "SERUM_CYSTATIN_C",
    "GAD65_ANTIBODY",
    "INSULIN_ANTIBODY",
    "ZNT8_ANTIBODY",
    "HEIGHT",
    "WEIGHT",
    "BMI",
    "BMI_CALCULATED",
    "HEIGHT_PERCENTILE",
    "HEIGHT_ZSCORE",
    "WEIGHT_PERCENTILE",
    "WEIGHT_ZSCORE",
    "BMI_PERCENTILE",
    # Original BP types replaced by inpatient/outpatient split
    "SYSTOLIC_BLOOD_PRESSURE",
    "DIASTOLIC_BLOOD_PRESSURE",
}

VALID_RANGES = {
    "HBA1C": (3.0, 25.0),
    "GLUCOSE": (10.0, 1000.0),
    "TOTAL_CHOLESTEROL": (50.0, 500.0),
    "HDL_CHOLESTEROL": (5.0, 200.0),
    "LDL_CHOLESTEROL": (5.0, 500.0),
    "TRIGLYCERIDES": (10.0, 5000.0),
    "ALT": (1.0, 5000.0),
    "AST": (1.0, 5000.0),
    "BUN": (1.0, 200.0),
    "SERUM_CREATININE": (0.1, 30.0),
    "SYSTOLIC_BLOOD_PRESSURE": (50.0, 300.0),
    "DIASTOLIC_BLOOD_PRESSURE": (20.0, 200.0),
    "HEIGHT": (10.0, 100.0),
    "WEIGHT": (50.0, 20000.0),
    "BMI": (10.0, 100.0),
    "BETA_HYDROXYBUTYRATE": (0.0, 50.0),
    "SERUM_C_PEPTIDE": (0.0, 50.0),
    "URINE_MICROALBUMIN": (0.0, 10000.0),
    "URINE_CREATININE": (0.0, 1000.0),
    "BLOOD_PH": (6.0, 8.0),
    "BICARBONATE": (1.0, 60.0),
    "PCO2": (5.0, 120.0),
}

# Clarity UACR ratio component codes (all mg/g or equivalent)
UACR_RATIO_CODES = [
    "60089",
    "1230001190",
    "63208",
    "16028",
    "80255",
    "1230001187",
    "9310",
]


# ── Pandas UDFs for z-score / percentile ──

@pandas_udf(DoubleType())
def calculate_z_score_pandas(value: pd.Series, L: pd.Series, M: pd.Series, S: pd.Series) -> pd.Series:
    import numpy as np

    mask = value.notna() & M.notna() & L.notna() & S.notna() & (M != 0) & (S != 0)
    result = pd.Series([None] * len(value), dtype=float)
    if mask.any():
        v = value[mask].values
        vL = L[mask].values
        vM = M[mask].values
        vS = S[mask].values
        z = np.full(len(v), np.nan)
        nz = vL != 0
        if nz.any():
            z[nz] = ((v[nz] / vM[nz]) ** vL[nz] - 1) / (vL[nz] * vS[nz])
        zr = vL == 0
        if zr.any():
            z[zr] = np.log(v[zr] / vM[zr]) / vS[zr]
        result[mask] = z
    return result


@pandas_udf(DoubleType())
def z_to_percentile_pandas(z: pd.Series) -> pd.Series:
    from scipy.special import erf
    import numpy as np

    result = pd.Series([None] * len(z), dtype=float)
    mask = z.notna()
    if mask.any():
        result[mask] = 0.5 * (1 + erf(z[mask].values / np.sqrt(2))) * 100
    return result


@transform(
    output=Output("/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d/transform_2_t2d_v2"),
    patient_grid=Input("/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d/transform_1_t2d_v2"),
    intermediate_meas=Input("ri.foundry.main.dataset.be8ae69b-a1c7-41cf-8fc0-1a75c5555661"),
    a1c_measurements=Input("ri.foundry.main.dataset.e3a1679b-3346-4b2d-a8df-2c584df29293"),
    lab_result=Input("ri.foundry.main.dataset.0f55df7f-be01-4952-9126-9e8f2f2892bc"),
    omop_raw_meas=Input("ri.foundry.main.dataset.179a6b21-3b5b-4e86-b0e4-8f295454ca20"),
    visit_occurrence=Input("ri.foundry.main.dataset.e8c60da1-8650-402a-8326-f9d75488061b"),
    cdc_height=Input("ri.foundry.main.dataset.12bd8beb-7102-422f-86ee-5e2fae6400a3"),
    cdc_weight=Input("ri.foundry.main.dataset.8fe0fbe2-3186-4283-b5bb-1038a6e9e8b7"),
    cdc_bmi=Input("ri.foundry.main.dataset.f3c79000-37f8-4a5d-9dd7-10be29c0dce6"),
    who_bmi_girls=Input("ri.foundry.main.dataset.01b778a4-b1fb-4fbc-9342-327fb14f0a02"),
    who_bmi_boys=Input("ri.foundry.main.dataset.6322d24c-4313-4fba-9a37-1338bdbd6705"),
    who_height_boys=Input("ri.foundry.main.dataset.508427fc-9a57-4cad-9034-7b241eff769f"),
    who_height_girls=Input("ri.foundry.main.dataset.3fa5e170-3ac9-4965-9176-3cda3249eceb"),
    who_weight_boys=Input("ri.foundry.main.dataset.58e35aba-7752-4884-b838-791d5b0adca0"),
    who_weight_girls=Input("ri.foundry.main.dataset.2d5f92ca-ecfe-4f98-8f32-cb53aa559f4f"),
)
def compute(
    patient_grid,
    intermediate_meas,
    a1c_measurements,
    lab_result,
    omop_raw_meas,
    visit_occurrence,
    output,
    cdc_height,
    cdc_weight,
    cdc_bmi,
    who_bmi_girls,
    who_bmi_boys,
    who_height_boys,
    who_height_girls,
    who_weight_boys,
    who_weight_girls,
):
    grid = patient_grid.dataframe()
    meas = intermediate_meas.dataframe()
    a1c_raw = a1c_measurements.dataframe()
    labs = lab_result.dataframe()
    raw_meas = omop_raw_meas.dataframe()
    visits = visit_occurrence.dataframe()

    # ================================================================
    # STEP 1: Prepare patient grid columns
    # ================================================================
    patient_cols = ["OMOP_ID", "mrn", "date_of_birth", "sex"]
    for i in range(1, NUM_VISITS + 1):
        patient_cols += [f"target_date_v{i}", f"window_start_v{i}", f"window_end_v{i}"]
    patients = grid.select(patient_cols).filter(F.col("OMOP_ID").isNotNull())

    patients = patients.withColumn(
        "sex_numeric",
        F.when(F.upper(F.col("sex")) == "MALE", 1)
        .when(F.upper(F.col("sex")) == "FEMALE", 2)
        .when(F.upper(F.col("sex")) == "M", 1)
        .when(F.upper(F.col("sex")) == "F", 2)
        .otherwise(None),
    )

    mrn_to_omop = (
        patients.select(
            F.col("mrn").cast("string").alias("grid_mrn"),
            "OMOP_ID",
        )
        .filter(F.col("grid_mrn").isNotNull())
        .dropDuplicates(["grid_mrn"])
    )

    # ================================================================
    # STEP 2: Prepare OMOP measurements — parse numeric values
    # ================================================================
    meas_clean = meas.select(
        F.col("PERSON_ID").cast("long").alias("OMOP_ID"),
        F.col("MEASUREMENT_DATETIME").cast("date").alias("meas_date"),
        F.col("measurement_type"),
        F.col("VALUE_SOURCE_VALUE").alias("raw_value"),
    ).filter(
        F.col("OMOP_ID").isNotNull()
        & F.col("meas_date").isNotNull()
        & F.col("measurement_type").isNotNull()
        & F.col("raw_value").isNotNull()
    )

    # Exclude dropped measurement types from OMOP data
    meas_clean = meas_clean.filter(
        ~F.col("measurement_type").isin(
            "EGFR", "SERUM_CYSTATIN_C",
            "GAD65_ANTIBODY", "INSULIN_ANTIBODY", "ZNT8_ANTIBODY",
        )
    )

    # Parse numeric — handle "120/80" BP format
    meas_clean = meas_clean.withColumn(
        "value",
        F.when(F.col("raw_value").contains("/"), F.split(F.col("raw_value"), "/").getItem(0).cast("double")).otherwise(
            F.col("raw_value").cast("double")
        ),
    ).filter(F.col("value").isNotNull())

    # Extract DBP from slash format
    dbp_from_sbp = (
        meas_clean.filter((F.col("measurement_type") == "SYSTOLIC_BLOOD_PRESSURE") & F.col("raw_value").contains("/"))
        .withColumn("dbp_val", F.split(F.col("raw_value"), "/").getItem(1).cast("double"))
        .filter(F.col("dbp_val").isNotNull())
        .select(
            "OMOP_ID",
            "meas_date",
            F.lit("DIASTOLIC_BLOOD_PRESSURE").alias("measurement_type"),
            F.col("dbp_val").alias("value"),
        )
    )

    meas_clean = meas_clean.select("OMOP_ID", "meas_date", "measurement_type", "value").unionByName(dbp_from_sbp)

    # Apply outlier removal using VALID_RANGES
    for mtype, (vmin, vmax) in VALID_RANGES.items():
        meas_clean = meas_clean.filter(
            ~((F.col("measurement_type") == mtype) & ((F.col("value") < vmin) | (F.col("value") > vmax)))
        )

    # Filter to cohort patients
    meas_clean = meas_clean.join(patients.select("OMOP_ID").distinct(), "OMOP_ID", "inner")

    # ================================================================
    # STEP 2d: Split BP into inpatient/outpatient using OMOP visit_occurrence
    #
    # Uses the raw OMOP measurement table (has VISIT_OCCURRENCE_ID) joined
    # with visit_occurrence (has VISIT_CONCEPT_ID) to determine encounter type:
    #   9202 = Outpatient
    #   9201 = Inpatient (includes surgery, hospital encounters)
    #   9203 = Emergency Room (grouped with inpatient)
    # ================================================================
    bp_types = ["SYSTOLIC_BLOOD_PRESSURE", "DIASTOLIC_BLOOD_PRESSURE"]
    non_bp_data = meas_clean.filter(~F.col("measurement_type").isin(bp_types))

    # Extract BP from raw OMOP measurement table (which has VISIT_OCCURRENCE_ID)
    bp_raw = (
        raw_meas.filter(
            F.col("MEASUREMENT_SOURCE_VALUE").isin("SYSTOLIC", "DIASTOLIC")
            & F.col("VALUE_SOURCE_VALUE").isNotNull()
            & F.col("PERSON_ID").isNotNull()
        )
        .select(
            F.col("PERSON_ID").cast("long").alias("OMOP_ID"),
            F.col("MEASUREMENT_DATETIME").cast("date").alias("meas_date"),
            F.col("MEASUREMENT_SOURCE_VALUE").alias("bp_type"),
            F.col("VALUE_SOURCE_VALUE").alias("raw_value"),
            F.col("VISIT_OCCURRENCE_ID").cast("long").alias("visit_occ_id"),
        )
    )

    # Join with visit_occurrence to get encounter type
    visit_types = visits.select(
        F.col("VISIT_OCCURRENCE_ID").cast("long").alias("vo_id"),
        F.col("VISIT_CONCEPT_ID").cast("long").alias("visit_concept"),
    ).filter(F.col("vo_id").isNotNull())

    bp_with_encounter = bp_raw.join(
        visit_types,
        bp_raw["visit_occ_id"] == visit_types["vo_id"],
        "left",
    ).drop("vo_id")

    # Classify encounter type
    bp_with_encounter = bp_with_encounter.withColumn(
        "setting",
        F.when(F.col("visit_concept") == 9202, "OUTPATIENT")
        .when(F.col("visit_concept").isin(9201, 9203), "INPATIENT")
        .otherwise("OUTPATIENT")  # Default to outpatient if unknown
    )

    # Parse BP values (same logic as main parser)
    bp_with_encounter = bp_with_encounter.withColumn(
        "value",
        F.when(F.col("raw_value").contains("/"),
               F.split(F.col("raw_value"), "/").getItem(0).cast("double"))
        .otherwise(F.col("raw_value").cast("double"))
    ).filter(F.col("value").isNotNull())

    # Create SBP rows
    sbp_rows = (
        bp_with_encounter.filter(F.col("bp_type") == "SYSTOLIC")
        .withColumn("measurement_type", F.concat(F.lit("SBP_"), F.col("setting")))
        .filter((F.col("value") >= 50.0) & (F.col("value") <= 300.0))
        .select("OMOP_ID", "meas_date", "measurement_type", "value")
    )

    # Create DBP rows — extract from slash format for SBP records
    dbp_from_slash = (
        bp_with_encounter.filter(
            (F.col("bp_type") == "SYSTOLIC") & F.col("raw_value").contains("/")
        )
        .withColumn("dbp_val", F.split(F.col("raw_value"), "/").getItem(1).cast("double"))
        .filter(F.col("dbp_val").isNotNull() & (F.col("dbp_val") >= 20.0) & (F.col("dbp_val") <= 200.0))
        .withColumn("measurement_type", F.concat(F.lit("DBP_"), F.col("setting")))
        .select("OMOP_ID", "meas_date", "measurement_type", F.col("dbp_val").alias("value"))
    )

    # Also get explicit DIASTOLIC records
    dbp_explicit = (
        bp_with_encounter.filter(F.col("bp_type") == "DIASTOLIC")
        .withColumn("measurement_type", F.concat(F.lit("DBP_"), F.col("setting")))
        .filter((F.col("value") >= 20.0) & (F.col("value") <= 200.0))
        .select("OMOP_ID", "meas_date", "measurement_type", "value")
    )

    # Combine all BP rows and filter to cohort
    all_bp = sbp_rows.unionByName(dbp_from_slash).unionByName(dbp_explicit)
    all_bp = all_bp.join(patients.select("OMOP_ID").distinct(), "OMOP_ID", "inner")

    # Recombine BP with non-BP measurements
    meas_clean = non_bp_data.unionByName(all_bp)

    # ================================================================
    # STEP 2b: Prepare dedicated A1c dataset
    # ================================================================
    a1c_clean = (
        a1c_raw.select(
            F.col("mrn").cast("string").alias("a1c_mrn"),
            F.col("order_value"),
            F.col("reference_unit"),
            F.col("result_time").cast("date").alias("a1c_date"),
        )
        .filter(
            F.col("a1c_date").isNotNull()
            & F.col("order_value").isNotNull()
            & (
                (F.lower(F.trim(F.col("reference_unit"))) == "%")
                | (F.lower(F.trim(F.col("reference_unit"))) == "% of total hgb")
            )
        )
        .withColumn(
            "a1c_value",
            F.when(
                F.col("order_value").startswith("<"),
                F.regexp_extract(F.col("order_value"), r"<(\d+\.?\d*)", 1).cast("double"),
            ).otherwise(
                F.regexp_extract(F.col("order_value"), r"(\d+\.?\d*)", 1).cast("double")
            ),
        )
        .filter((F.col("a1c_value") > 0) & (F.col("a1c_value") <= 20.0))
        .select("a1c_mrn", "a1c_date", "a1c_value")
    )

    # ── Cap A1c at 14 ──
    a1c_clean = a1c_clean.withColumn(
        "a1c_value",
        F.when(F.col("a1c_value") >= 14.0, 14.0).otherwise(F.col("a1c_value"))
    )

    a1c_with_omop = (
        a1c_clean.join(mrn_to_omop, a1c_clean["a1c_mrn"] == mrn_to_omop["grid_mrn"], "inner")
        .select(
            "OMOP_ID",
            F.col("a1c_date").alias("meas_date"),
            F.lit("HBA1C").alias("measurement_type"),
            F.col("a1c_value").alias("value"),
        )
    )

    # ================================================================
    # STEP 2c: Prepare Clarity UACR ratio dataset
    # ================================================================
    uacr_clarity = (
        labs.filter(
            F.col("clarity_component_code").isin(UACR_RATIO_CODES)
            & F.col("order_num_value").isNotNull()
            & (F.col("order_num_value") > 0)
        )
        .select(
            F.col("mrn").cast("string").alias("uacr_mrn"),
            F.col("result_date").cast("date").alias("uacr_date"),
            F.col("order_num_value").cast("double").alias("uacr_value"),
        )
        .filter(F.col("uacr_date").isNotNull())
    )

    # ── Cap UACR_RATIO at 300 mg/g (sentinel values like 9999 → 300) ──
    uacr_clarity = uacr_clarity.withColumn(
        "uacr_value",
        F.when(F.col("uacr_value") > 300.0, 300.0).otherwise(F.col("uacr_value"))
    )

    uacr_with_omop = (
        uacr_clarity.join(mrn_to_omop, uacr_clarity["uacr_mrn"] == mrn_to_omop["grid_mrn"], "inner")
        .select(
            "OMOP_ID",
            F.col("uacr_date").alias("meas_date"),
            F.lit("UACR_RATIO").alias("measurement_type"),
            F.col("uacr_value").alias("value"),
        )
    )

    # ================================================================
    # STEP 3: Assign OMOP measurements to closest visit window
    # ================================================================
    results = []
    for i in range(1, NUM_VISITS + 1):
        v = f"v{i}"
        target_col = f"target_date_{v}"
        start_col = f"window_start_{v}"
        end_col = f"window_end_{v}"

        visit_meas = (
            meas_clean.alias("m")
            .join(
                patients.select("OMOP_ID", target_col, start_col, end_col, "date_of_birth", "sex_numeric").alias("p"),
                F.col("m.OMOP_ID") == F.col("p.OMOP_ID"),
                "inner",
            )
            .filter((F.col("meas_date") >= F.col(start_col)) & (F.col("meas_date") <= F.col(end_col)))
            .withColumn("days_from_target", F.abs(F.datediff("meas_date", target_col)))
            .withColumn("visit", F.lit(v))
            .withColumn("age_months_at_meas", F.round(F.months_between(F.col("meas_date"), F.col("date_of_birth")), 2))
            .select(
                F.col("m.OMOP_ID").alias("OMOP_ID"),
                "measurement_type",
                "value",
                "meas_date",
                "days_from_target",
                "visit",
                "age_months_at_meas",
                "sex_numeric",
            )
        )
        results.append(visit_meas)

    all_visits = results[0]
    for r in results[1:]:
        all_visits = all_visits.unionByName(r)

    # Keep closest measurement per patient/type/visit
    w = Window.partitionBy("OMOP_ID", "measurement_type", "visit").orderBy("days_from_target")
    closest = (
        all_visits.withColumn("_rank", F.row_number().over(w))
        .filter(F.col("_rank") == 1)
        .drop("_rank", "days_from_target")
    )

    # ================================================================
    # STEP 3b: Overlay dedicated A1c (primary) over OMOP HBA1C (fallback)
    # ================================================================
    a1c_visit_results = []
    for i in range(1, NUM_VISITS + 1):
        v = f"v{i}"
        target_col = f"target_date_{v}"
        start_col = f"window_start_{v}"
        end_col = f"window_end_{v}"

        a1c_visit = (
            a1c_with_omop.alias("a")
            .join(
                patients.select("OMOP_ID", target_col, start_col, end_col, "date_of_birth", "sex_numeric").alias("p"),
                F.col("a.OMOP_ID") == F.col("p.OMOP_ID"),
                "inner",
            )
            .filter((F.col("meas_date") >= F.col(start_col)) & (F.col("meas_date") <= F.col(end_col)))
            .withColumn("days_from_target", F.abs(F.datediff("meas_date", target_col)))
            .withColumn("visit", F.lit(v))
            .withColumn("age_months_at_meas", F.round(F.months_between(F.col("meas_date"), F.col("date_of_birth")), 2))
            .select(
                F.col("a.OMOP_ID").alias("OMOP_ID"),
                "measurement_type",
                "value",
                "meas_date",
                "days_from_target",
                "visit",
                "age_months_at_meas",
                "sex_numeric",
            )
        )
        a1c_visit_results.append(a1c_visit)

    all_a1c_visits = a1c_visit_results[0]
    for r in a1c_visit_results[1:]:
        all_a1c_visits = all_a1c_visits.unionByName(r)

    # Keep closest dedicated A1c per patient-visit
    w_a1c = Window.partitionBy("OMOP_ID", "visit").orderBy("days_from_target")
    closest_a1c = (
        all_a1c_visits.withColumn("_rank", F.row_number().over(w_a1c))
        .filter(F.col("_rank") == 1)
        .drop("_rank", "days_from_target")
    )

    # Remove OMOP HBA1C for patients who have dedicated A1c data in that visit
    a1c_patients_visits = closest_a1c.select("OMOP_ID", "visit").distinct()
    closest = closest.join(
        a1c_patients_visits.withColumn("_has_a1c", F.lit(True)),
        ["OMOP_ID", "visit"],
        "left",
    )
    closest = closest.filter(
        ~((F.col("measurement_type") == "HBA1C") & (F.col("_has_a1c") == True))
    ).drop("_has_a1c")

    # Union dedicated A1c into closest
    closest = closest.unionByName(closest_a1c)

    # ── Also cap OMOP HBA1C at 14 (for patients using OMOP fallback) ──
    closest = closest.withColumn(
        "value",
        F.when((F.col("measurement_type") == "HBA1C") & (F.col("value") >= 14.0), 14.0)
        .otherwise(F.col("value"))
    )

    # ================================================================
    # STEP 3c: Overlay Clarity UACR (primary) over OMOP-derived UACR
    # ================================================================
    uacr_visit_results = []
    for i in range(1, NUM_VISITS + 1):
        v = f"v{i}"
        target_col = f"target_date_{v}"
        start_col = f"window_start_{v}"
        end_col = f"window_end_{v}"

        uacr_visit = (
            uacr_with_omop.alias("u")
            .join(
                patients.select("OMOP_ID", target_col, start_col, end_col, "date_of_birth", "sex_numeric").alias("p"),
                F.col("u.OMOP_ID") == F.col("p.OMOP_ID"),
                "inner",
            )
            .filter((F.col("meas_date") >= F.col(start_col)) & (F.col("meas_date") <= F.col(end_col)))
            .withColumn("days_from_target", F.abs(F.datediff("meas_date", target_col)))
            .withColumn("visit", F.lit(v))
            .withColumn("age_months_at_meas", F.round(F.months_between(F.col("meas_date"), F.col("date_of_birth")), 2))
            .select(
                F.col("u.OMOP_ID").alias("OMOP_ID"),
                "measurement_type",
                "value",
                "meas_date",
                "days_from_target",
                "visit",
                "age_months_at_meas",
                "sex_numeric",
            )
        )
        uacr_visit_results.append(uacr_visit)

    all_uacr_visits = uacr_visit_results[0]
    for r in uacr_visit_results[1:]:
        all_uacr_visits = all_uacr_visits.unionByName(r)

    # Keep closest Clarity UACR per patient-visit
    w_uacr = Window.partitionBy("OMOP_ID", "visit").orderBy("days_from_target")
    closest_clarity_uacr = (
        all_uacr_visits.withColumn("_rank", F.row_number().over(w_uacr))
        .filter(F.col("_rank") == 1)
        .drop("_rank", "days_from_target")
    )

    # Add Clarity UACR_RATIO rows
    closest = closest.unionByName(closest_clarity_uacr)

    # ================================================================
    # STEP 4: Compute BMI z-score (only — drop raw HEIGHT, WEIGHT, BMI)
    # ================================================================
    hw = (
        closest.filter(F.col("measurement_type").isin("HEIGHT", "WEIGHT"))
        .groupBy("OMOP_ID", "visit")
        .pivot("measurement_type", ["HEIGHT", "WEIGHT"])
        .agg(
            F.first("value").alias("val"),
            F.first("meas_date").alias("dt"),
            F.first("age_months_at_meas").alias("age"),
            F.first("sex_numeric").alias("sex"),
        )
    )

    hw = hw.withColumn("height_cm", F.col("HEIGHT_val") * 2.54)
    hw = hw.withColumn("weight_kg", F.col("WEIGHT_val") * 0.0283495231)

    hw = hw.withColumn(
        "bmi_calc",
        F.when(
            F.col("height_cm").isNotNull() & F.col("weight_kg").isNotNull() & (F.col("height_cm") > 0),
            F.col("weight_kg") / ((F.col("height_cm") / 100) ** 2),
        ),
    )

    hw = hw.withColumn("bmi_age", F.col("WEIGHT_age"))
    hw = hw.withColumn("bmi_date", F.col("WEIGHT_dt"))
    hw = hw.withColumn("bmi_sex", F.col("WEIGHT_sex"))

    # ================================================================
    # STEP 5: Prepare CDC / WHO reference tables (for BMI z-score only)
    # ================================================================
    cdc_bmi_df = (
        cdc_bmi.dataframe()
        .withColumnRenamed("Agemos", "age_months")
        .withColumnRenamed("Sex", "sex")
        .select("age_months", "sex", "L", "M", "S")
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

    # ================================================================
    # STEP 6: Compute BMI z-score using CDC/WHO reference
    # ================================================================
    def add_zscore(hw_df, value_col, age_col, sex_col, cdc_ref, who_ref, prefix):
        df = hw_df.withColumn("_rid", F.monotonically_increasing_id())

        df = df.withColumn("_use_who", (F.col(age_col) >= 0) & (F.col(age_col) < 24))
        df = df.withColumn("_use_cdc", (F.col(age_col) >= 24) & (F.col(age_col) <= 240))

        who_prep = who_ref.select(
            F.col("age_months").alias("_who_age"),
            F.col("sex").alias("_who_sex"),
            F.col("L").alias("_who_L"),
            F.col("M").alias("_who_M"),
            F.col("S").alias("_who_S"),
        )
        df = df.join(who_prep, (F.col(sex_col) == F.col("_who_sex")) & F.col("_use_who"), "left")
        df = df.withColumn("_who_dist", F.abs(F.col(age_col) - F.col("_who_age")))
        w_who = Window.partitionBy("_rid").orderBy("_who_dist")
        df = df.withColumn("_wr", F.row_number().over(w_who))
        df = df.filter((F.col("_wr") == 1) | F.col("_wr").isNull()).drop("_wr", "_who_dist")

        cdc_prep = cdc_ref.select(
            F.col("age_months").alias("_cdc_age"),
            F.col("sex").alias("_cdc_sex"),
            F.col("L").alias("_cdc_L"),
            F.col("M").alias("_cdc_M"),
            F.col("S").alias("_cdc_S"),
        )
        df = df.join(cdc_prep, (F.col(sex_col) == F.col("_cdc_sex")) & F.col("_use_cdc"), "left")
        df = df.withColumn("_cdc_dist", F.abs(F.col(age_col) - F.col("_cdc_age")))
        w_cdc = Window.partitionBy("_rid").orderBy("_cdc_dist")
        df = df.withColumn("_cr", F.row_number().over(w_cdc))
        df = df.filter((F.col("_cr") == 1) | F.col("_cr").isNull()).drop("_cr", "_cdc_dist")

        df = df.withColumn(
            f"{prefix}_zscore",
            F.when(
                F.col("_use_who") & F.col("_who_L").isNotNull(),
                calculate_z_score_pandas(
                    F.col(value_col).cast("double"),
                    F.col("_who_L").cast("double"),
                    F.col("_who_M").cast("double"),
                    F.col("_who_S").cast("double"),
                ),
            )
            .when(
                F.col("_use_cdc") & F.col("_cdc_L").isNotNull(),
                calculate_z_score_pandas(
                    F.col(value_col).cast("double"),
                    F.col("_cdc_L").cast("double"),
                    F.col("_cdc_M").cast("double"),
                    F.col("_cdc_S").cast("double"),
                ),
            )
            .otherwise(None),
        )

        drop_cols = [
            "_rid", "_use_who", "_use_cdc",
            "_who_age", "_who_sex", "_who_L", "_who_M", "_who_S",
            "_cdc_age", "_cdc_sex", "_cdc_L", "_cdc_M", "_cdc_S",
        ]
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(c)

        return df

    hw = add_zscore(hw, "bmi_calc", "bmi_age", "bmi_sex", cdc_bmi_df, who_bmi_df, "bmi")

    # ================================================================
    # STEP 7: Only keep BMI_ZSCORE as derived measurement row
    # ================================================================
    bmi_zscore_rows = hw.filter(F.col("bmi_zscore").isNotNull()).select(
        "OMOP_ID", "visit",
        F.lit("BMI_ZSCORE").alias("measurement_type"),
        F.col("bmi_zscore").alias("value"),
        F.col("bmi_date").alias("meas_date"),
        F.col("bmi_age").alias("age_months_at_meas"),
        F.col("bmi_sex").alias("sex_numeric"),
    )

    all_measurements = closest.unionByName(bmi_zscore_rows)

    # ================================================================
    # STEP 8: Final output — exclude dropped measurement types
    # ================================================================
    final = all_measurements.filter(
        ~F.col("measurement_type").isin(list(EXCLUDE_MEASUREMENT_TYPES))
    ).select("OMOP_ID", "measurement_type", "visit", "value", "meas_date")

    for ref in [cdc_bmi_df, who_bmi_df]:
        ref.unpersist()

    output.write_dataframe(final)


"""
Transform 3 – Temporal Medications (v2)
========================================
Changes vs v1:
  - Dropped 7 medication classes per user request:
    SGLT2_inhibitors, DPP4_inhibitors, Sulfonylureas,
    Alpha_glucosidase_inhibitors, Meglitinides,
    Thiazolidinediones, Amylin_analogue
  - Kept only: Insulins, Biguanide, GLP1_agonists

Output: OMOP_ID | medication_class | visit | value (0 or 1)
"""

from pyspark.sql import functions as F
from transforms.api import Input, Output, transform

NUM_VISITS = 10

# Only 3 medication classes retained
MED_CLASSES = {
    "Insulins": {
        "include": ["INSULIN"],
        "exclude": [
            "INSULIN ANTIBOD",
            "INSULIN SYRINGE",
            "INSULIN PEN NEEDLE",
            "INSULIN AB",
            "NOVOFINE",
            "LANTUS SOLOSTAR",
        ],
    },
    "Biguanide": {
        "include": ["METFORMIN", "GLUCOPHAGE", "GLUMETZA", "RIOMET"],
        "exclude": [],
    },
    "GLP1_agonists": {
        "include": [
            "LIRAGLUTIDE",
            "EXENATIDE",
            "DULAGLUTIDE",
            "SEMAGLUTIDE",
            "LIXISENATIDE",
            "ALBIGLUTIDE",
            "VICTOZA",
            "BYETTA",
            "BYDUREON",
            "TRULICITY",
            "OZEMPIC",
            "RYBELSUS",
            "WEGOVY",
            "SAXENDA",
            "MOUNJARO",
            "TIRZEPATIDE",
            "ZEPBOUND",
            "MANJARO",
        ],
        "exclude": [],
    },
}


@transform(
    output=Output("/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d/transform_3_t2d_v2"),
    patient_grid=Input("/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d/transform_1_t2d_v2"),
    drug_exposure=Input("ri.foundry.main.dataset.cd4dba7f-a319-480f-9497-9a324a37b308"),
)
def compute(patient_grid, drug_exposure, output):
    grid = patient_grid.dataframe()
    drugs = drug_exposure.dataframe()

    # ================================================================
    # STEP 1: Patient visit windows
    # ================================================================
    patient_cols = ["OMOP_ID"]
    for i in range(1, NUM_VISITS + 1):
        patient_cols += [f"window_start_v{i}", f"window_end_v{i}"]
    patients = grid.select(patient_cols).filter(F.col("OMOP_ID").isNotNull())

    # ================================================================
    # STEP 2: Prepare drug exposure records
    # ================================================================
    drugs_clean = (
        drugs.select(
            F.col("PERSON_ID").cast("long").alias("OMOP_ID"),
            F.col("DRUG_EXPOSURE_START_DATE").cast("date").alias("drug_date"),
            F.coalesce(
                F.col("DRUG_EXPOSURE_START_DATE").cast("date"), F.col("DRUG_EXPOSURE_START_DATETIME").cast("date")
            ).alias("drug_date_final"),
            F.upper(F.trim(F.col("DRUG_SOURCE_VALUE"))).alias("drug_name"),
        )
        .withColumn("drug_date", F.coalesce(F.col("drug_date_final"), F.col("drug_date")))
        .drop("drug_date_final")
        .filter(F.col("OMOP_ID").isNotNull() & F.col("drug_date").isNotNull() & F.col("drug_name").isNotNull())
    )

    # Filter to cohort patients only
    drugs_clean = drugs_clean.join(patients.select("OMOP_ID").distinct(), "OMOP_ID", "inner")

    # ================================================================
    # STEP 3: Classify each drug record into medication classes
    # ================================================================
    for med_class, config in MED_CLASSES.items():
        incl_expr = F.lit(False)
        for kw in config["include"]:
            incl_expr = incl_expr | F.col("drug_name").contains(kw)

        excl_expr = F.lit(False)
        for kw in config["exclude"]:
            excl_expr = excl_expr | F.col("drug_name").contains(kw)

        drugs_clean = drugs_clean.withColumn(f"is_{med_class}", F.when(incl_expr & ~excl_expr, 1).otherwise(0))

    # ================================================================
    # STEP 4: For each visit window, aggregate medication flags
    # ================================================================
    results = []
    for i in range(1, NUM_VISITS + 1):
        visit = f"v{i}"
        start_col = f"window_start_{visit}"
        end_col = f"window_end_{visit}"

        visit_drugs = (
            drugs_clean.alias("d")
            .join(
                patients.select("OMOP_ID", start_col, end_col).alias("p"),
                F.col("d.OMOP_ID") == F.col("p.OMOP_ID"),
                "inner",
            )
            .filter((F.col("drug_date") >= F.col(start_col)) & (F.col("drug_date") <= F.col(end_col)))
        )

        visit_agg = visit_drugs.groupBy(F.col("d.OMOP_ID").alias("OMOP_ID")).agg(
            *[F.max(F.col(f"is_{med_class}")).alias(med_class) for med_class in MED_CLASSES.keys()]
        )

        # Unpivot to long format
        med_class_names = list(MED_CLASSES.keys())
        stack_expr = ", ".join([f"'{mc}', `{mc}`" for mc in med_class_names])
        n_cols = len(med_class_names)

        visit_long = visit_agg.selectExpr(
            "OMOP_ID", f"stack({n_cols}, {stack_expr}) as (medication_class, value)"
        ).withColumn("visit", F.lit(visit))

        results.append(visit_long)

    # Union all visits
    all_meds = results[0]
    for r in results[1:]:
        all_meds = all_meds.unionByName(r)

    output.write_dataframe(all_meds)


"""
Transform 4 – Temporal Conditions (ICD Codes) (v2)
====================================================
No changes from v1 except output path.

Output: OMOP_ID | condition | visit | value (0 or 1)
"""

from pyspark.sql import functions as F
from transforms.api import Input, Output, transform

NUM_VISITS = 10

# All conditions kept here (outcomes depend on ICD flags).
# Conditions redundant with outcomes are filtered out in Transform 6.
CONDITION_CODES = {
    "DKA": [
        "250.11", "250.13", "250.10", "250.12",
        "E10.10", "E10.11", "E11.10", "E11.11",
    ],
    "Ketosis": ["276.2", "790.6", "E87.2"],
    "Dyslipidemia": [
        "272.", "E78.0", "E78.1", "E78.2", "E78.3", "E78.4", "E78.5", "E78.6",
    ],
    "Hypertension": [
        "401.", "402.", "403.", "404.", "405.",
        "I10.", "I11.", "I12.", "I13.", "I15.",
        "H35.03", "I67.4",
    ],
    "Diabetic_Retinopathy": [
        "362.01", "362.02", "362.03", "362.04", "362.05", "362.06",
        "E08.35", "E09.35", "E10.35", "E11.35", "E13.35",
        "E08.31", "E08.37", "E09.31", "E09.37",
        "E10.31", "E10.37", "E11.31", "E11.37", "E13.31", "E13.37",
        "E08.32", "E09.32", "E10.32", "E11.32", "E13.32",
        "E08.33", "E09.33", "E10.33", "E11.33", "E13.33",
        "E08.34", "E09.34", "E10.34", "E11.34", "E13.34",
    ],
    "Microalbuminuria": ["791.0", "R80.9"],
    "Neuropathy": [
        "250.61", "250.63", "250.60", "250.62", "357.2",
        "E10.40", "E10.41", "E10.42", "E10.43", "E10.44", "E10.49",
        "E11.40", "E11.41", "E11.42", "E11.45",
    ],
    "Hypoglycemia": [
        "250.3", "250.8", "251.0", "251.1", "251.2", "270.3", "775.0", "775.6", "962.39",
        "E08.641", "E08.649", "E09.641", "E09.649",
        "E10.641", "E10.649", "E11.641", "E11.649", "E13.641", "E13.649",
        "E15", "E16.0", "E16.1", "E16.2",
        "T38.3X1A", "T38.3X1D", "T38.3X1S",
        "T38.3X2A", "T38.3X2D", "T38.3X2S",
        "T38.3X3A", "T38.3X3D", "T38.3X3S",
        "T38.3X4A", "T38.3X4D", "T38.3X4S",
        "T38.3X5A", "T38.3X5D", "T38.3X5S",
    ],
}


@transform(
    output=Output("/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d/transform_4_t2d_v2"),
    patient_grid=Input("/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d/transform_1_t2d_v2"),
    condition_occurrence=Input("ri.foundry.main.dataset.d81c73f9-fb8a-4a0b-ba95-10e7dcbd6958"),
)
def compute(patient_grid, condition_occurrence, output):
    grid = patient_grid.dataframe()
    cond = condition_occurrence.dataframe()

    # ================================================================
    # STEP 1: Patient visit windows
    # ================================================================
    patient_cols = ["OMOP_ID"]
    for i in range(1, NUM_VISITS + 1):
        patient_cols += [f"window_start_v{i}", f"window_end_v{i}"]
    patients = grid.select(patient_cols).filter(F.col("OMOP_ID").isNotNull())

    # ================================================================
    # STEP 2: Prepare conditions — extract ICD codes from pipe format
    # ================================================================
    cond_clean = cond.select(
        F.col("PERSON_ID").cast("long").alias("OMOP_ID"),
        F.coalesce(F.col("CONDITION_START_DATE").cast("date"), F.col("CONDITION_START_DATETIME").cast("date")).alias(
            "cond_date"
        ),
        F.when(
            F.col("CONDITION_SOURCE_VALUE").contains("|"),
            F.trim(F.element_at(F.split(F.col("CONDITION_SOURCE_VALUE"), "\\|"), 2)),
        )
        .otherwise(F.trim(F.col("CONDITION_SOURCE_VALUE")))
        .alias("icd_code"),
    ).filter(
        F.col("OMOP_ID").isNotNull()
        & F.col("cond_date").isNotNull()
        & F.col("icd_code").isNotNull()
        & (F.col("icd_code") != "")
    )

    # Filter to cohort patients
    cond_clean = cond_clean.join(patients.select("OMOP_ID").distinct(), "OMOP_ID", "inner")

    # ================================================================
    # STEP 3: Classify conditions using ICD code matching
    # ================================================================
    for cond_name, codes in CONDITION_CODES.items():
        condition_filter = None
        for code in codes:
            exact_match = F.col("icd_code") == code
            prefix_match = F.col("icd_code").startswith(code)
            cond_expr = exact_match | prefix_match
            condition_filter = cond_expr if condition_filter is None else (condition_filter | cond_expr)

        cond_clean = cond_clean.withColumn(f"is_{cond_name}", F.when(condition_filter, 1).otherwise(0))

    # ================================================================
    # STEP 4: For each visit window, aggregate condition flags
    # ================================================================
    results = []
    for i in range(1, NUM_VISITS + 1):
        visit = f"v{i}"
        start_col = f"window_start_{visit}"
        end_col = f"window_end_{visit}"

        visit_conds = (
            cond_clean.alias("c")
            .join(
                patients.select("OMOP_ID", start_col, end_col).alias("p"),
                F.col("c.OMOP_ID") == F.col("p.OMOP_ID"),
                "inner",
            )
            .filter((F.col("cond_date") >= F.col(start_col)) & (F.col("cond_date") <= F.col(end_col)))
        )

        cond_names = list(CONDITION_CODES.keys())
        visit_agg = visit_conds.groupBy(F.col("c.OMOP_ID").alias("OMOP_ID")).agg(
            *[F.max(F.col(f"is_{cn}")).alias(cn) for cn in cond_names]
        )

        stack_expr = ", ".join([f"'{cn}', `{cn}`" for cn in cond_names])
        n_cols = len(cond_names)

        visit_long = visit_agg.selectExpr("OMOP_ID", f"stack({n_cols}, {stack_expr}) as (condition, value)").withColumn(
            "visit", F.lit(visit)
        )
        results.append(visit_long)

    all_conds = results[0]
    for r in results[1:]:
        all_conds = all_conds.unionByName(r)

    output.write_dataframe(all_conds)


"""
Transform 5 – Temporal Outcomes (v2)
======================================
Changes vs v1:
  - Updated _ADDITIONAL_MED_CLASSES to only include Insulins, GLP1_agonists
    (other classes dropped from pipeline). Metformin "alone" now means
    not on Insulins or GLP-1RA.
  - UACR_RATIO already capped at 300 in Transform 2.

Output: OMOP_ID | outcome | visit | value (1/0/null)
"""

from pyspark.sql import functions as F
from pyspark.sql import Window
from transforms.api import Input, Output, transform

NUM_VISITS = 10
UACR_THRESHOLD = 30

# Clarity UACR ratio component codes
UACR_RATIO_CODES = [
    "60089", "1230001190", "63208", "16028", "80255", "1230001187", "9310",
]

# AAP 2017 simplified BP thresholds (~95th percentile)
_CHILD_BP = [
    (1, 3, 100, 55),
    (4, 5, 104, 64),
    (6, 7, 108, 70),
    (8, 9, 110, 72),
    (10, 11, 114, 74),
    (12, 12, 118, 76),
]

# Lipid-lowering drug keywords
_LIPID_INCL = [
    "ATORVASTATIN", "ROSUVASTATIN", "SIMVASTATIN", "PRAVASTATIN",
    "LOVASTATIN", "FLUVASTATIN", "PITAVASTATIN",
    "FENOFIBRATE", "GEMFIBROZIL", "EZETIMIBE",
    "LIPITOR", "CRESTOR", "ZOCOR",
]
_LIPID_EXCL = ["NYSTATIN", "CILASTATIN", "IMIPENEM"]

# Additional med classes checked for metformin-alone logic
# NOTE: Only includes remaining tracked classes (Insulins, GLP1_agonists)
# since other classes (SGLT2i, DPP4i, SU, TZD, meglitinides, AGI, amylin)
# were dropped from the pipeline per user request.
_ADDITIONAL_MED_CLASSES = [
    "Insulins",
    "GLP1_agonists",
]


def _bp_elevated(sbp, dbp, age):
    """Column expression: 1 if BP elevated for given age."""
    expr = F.when(age >= 13, F.when((sbp >= 130) | (F.coalesce(dbp, F.lit(0)) >= 80), 1).otherwise(0))
    for lo, hi, st, dt in _CHILD_BP:
        expr = expr.when(
            (age >= lo) & (age <= hi), F.when((sbp >= st) | (F.coalesce(dbp, F.lit(0)) >= dt), 1).otherwise(0)
        )
    return expr.otherwise(F.lit(None).cast("integer"))


@transform(
    output=Output("/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d/transform_5_t2d_v2"),
    patient_grid=Input("/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d/transform_1_t2d_v2"),
    temporal_meas=Input("/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d/transform_2_t2d_v2"),
    temporal_meds=Input("/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d/transform_3_t2d_v2"),
    temporal_conds=Input("/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d/transform_4_t2d_v2"),
    intermediate_meas=Input("ri.foundry.main.dataset.be8ae69b-a1c7-41cf-8fc0-1a75c5555661"),
    drug_exposure=Input("ri.foundry.main.dataset.cd4dba7f-a319-480f-9497-9a324a37b308"),
    lab_result=Input("ri.foundry.main.dataset.0f55df7f-be01-4952-9126-9e8f2f2892bc"),
)
def compute(patient_grid, temporal_meas, temporal_meds, temporal_conds, intermediate_meas, drug_exposure, lab_result, output):
    grid = patient_grid.dataframe()
    meas = temporal_meas.dataframe()
    meds = temporal_meds.dataframe()
    conds = temporal_conds.dataframe()
    omop_raw = intermediate_meas.dataframe()
    drugs = drug_exposure.dataframe()
    labs = lab_result.dataframe()

    # ================================================================
    # STEP 1: Build per-visit feature table (wide format)
    # ================================================================
    meas_pivot = meas.groupBy("OMOP_ID", "visit").pivot("measurement_type").agg(F.first("value"))
    meds_pivot = meds.groupBy("OMOP_ID", "visit").pivot("medication_class").agg(F.first("value"))
    conds_pivot = conds.groupBy("OMOP_ID", "visit").pivot("condition").agg(F.first("value"))

    # ================================================================
    # STEP 2: Create visit rows from patient grid
    # ================================================================
    visits_df = None
    for i in range(1, NUM_VISITS + 1):
        v = f"v{i}"
        v_row = (
            grid.select(
                "OMOP_ID", "mrn", "age_at_diagnosis", "date_of_birth",
                F.col(f"target_date_{v}").alias("target_date"),
                F.col(f"window_start_{v}").alias("window_start"),
                F.col(f"window_end_{v}").alias("window_end"),
            )
            .filter(F.col("OMOP_ID").isNotNull())
            .withColumn("visit", F.lit(v))
            .withColumn("age_at_visit", F.floor(F.datediff("target_date", "date_of_birth") / 365.25).cast("integer"))
        )
        visits_df = v_row if visits_df is None else visits_df.unionByName(v_row)

    # Join all pivoted features
    df = visits_df
    df = df.join(meas_pivot, ["OMOP_ID", "visit"], "left")
    df = df.join(meds_pivot, ["OMOP_ID", "visit"], "left")
    df = df.join(conds_pivot, ["OMOP_ID", "visit"], "left")

    all_cols = set(df.columns)

    def safe(name, dt="double"):
        return F.col(name).cast(dt) if name in all_cols else F.lit(None).cast(dt)

    def safe_int(name):
        if name not in all_cols:
            return F.lit(None).cast("integer")
        return F.col(name).cast("integer")

    # ==================================================================
    # OUTCOME 1: HYPERTENSION (Outpatient – 3 Closest BP)
    # ==================================================================
    bp_raw = (
        omop_raw.filter(F.col("measurement_type") == "SYSTOLIC_BLOOD_PRESSURE")
        .select(
            F.col("PERSON_ID").cast("long").alias("bp_pid"),
            F.col("MEASUREMENT_DATETIME").cast("date").alias("bp_date"),
            F.col("VALUE_SOURCE_VALUE").alias("raw"),
        )
        .filter(F.col("bp_pid").isNotNull() & F.col("raw").isNotNull())
        .withColumn(
            "sbp",
            F.when(F.col("raw").contains("/"), F.split(F.col("raw"), "/").getItem(0).cast("double")).otherwise(
                F.col("raw").cast("double")
            ),
        )
        .withColumn(
            "dbp",
            F.when(F.col("raw").contains("/"), F.split(F.col("raw"), "/").getItem(1).cast("double")).otherwise(
                F.lit(None).cast("double")
            ),
        )
        .filter(F.col("sbp").isNotNull())
    )

    bp_daily = (
        bp_raw.groupBy("bp_pid", "bp_date")
        .agg(F.avg("sbp").alias("sbp"), F.avg("dbp").alias("dbp"), F.count("*").alias("cnt"))
        .filter(F.col("cnt") <= 2)
        .drop("cnt")
    )

    htn_results = []
    for i in range(1, NUM_VISITS + 1):
        v = f"v{i}"
        v_patients = visits_df.filter(F.col("visit") == v).select(
            "OMOP_ID", "target_date", "window_start", "window_end", "age_at_visit"
        )

        joined = (
            bp_daily.join(v_patients, bp_daily["bp_pid"] == v_patients["OMOP_ID"], "inner")
            .filter((F.col("bp_date") >= F.col("window_start")) & (F.col("bp_date") <= F.col("window_end")))
            .withColumn("abs_days", F.abs(F.datediff("bp_date", "target_date")))
            .withColumn("elev", _bp_elevated(F.col("sbp"), F.col("dbp"), F.col("age_at_visit")))
        )

        w_rank = Window.partitionBy("bp_pid").orderBy("abs_days", "bp_date")
        joined = joined.withColumn("_rank", F.row_number().over(w_rank))
        closest_3 = joined.filter(F.col("_rank") <= 3)

        stats = (
            closest_3.groupBy("bp_pid")
            .agg(
                F.count("*").cast("integer").alias("n_closest"),
                F.sum("elev").cast("integer").alias("n_elevated"),
            )
            .withColumn("all3_elevated", F.when((F.col("n_closest") >= 3) & (F.col("n_elevated") >= 3), 1).otherwise(0))
            .select(
                F.col("bp_pid").alias("OMOP_ID"),
                F.lit(v).alias("visit"),
                "n_closest", "n_elevated", "all3_elevated",
            )
        )
        htn_results.append(stats)

    htn_all = htn_results[0]
    for r in htn_results[1:]:
        htn_all = htn_all.unionByName(r)

    df = df.join(htn_all, ["OMOP_ID", "visit"], "left")

    htn_icd = F.coalesce(safe_int("Hypertension"), F.lit(0))
    all3 = F.coalesce(F.col("all3_elevated"), F.lit(0))
    n_bp = F.coalesce(F.col("n_closest"), F.lit(0))

    df = df.withColumn(
        "OUTCOME_Hypertension",
        F.when(htn_icd == 1, 1)
        .when(all3 == 1, 1)
        .when(n_bp >= 3, 0)
        .when((n_bp > 0) & ((htn_icd == 0) | htn_icd.isNull()), 0)
        .otherwise(F.lit(None).cast("integer")),
    )

    # ==================================================================
    # OUTCOME 2: DYSLIPIDEMIA
    # ==================================================================
    incl = F.lit(False)
    for k in _LIPID_INCL:
        incl = incl | F.upper(F.col("DRUG_SOURCE_VALUE")).contains(k)
    excl = F.lit(False)
    for k in _LIPID_EXCL:
        excl = excl | F.upper(F.col("DRUG_SOURCE_VALUE")).contains(k)

    lipid_rx = (
        drugs.filter(incl & ~excl)
        .select(
            F.col("PERSON_ID").cast("long").alias("lrx_pid"),
            F.coalesce(
                F.col("DRUG_EXPOSURE_START_DATE").cast("date"), F.col("DRUG_EXPOSURE_START_DATETIME").cast("date")
            ).alias("rxd"),
        )
        .filter(F.col("lrx_pid").isNotNull() & F.col("rxd").isNotNull())
    )

    lipid_results = []
    for i in range(1, NUM_VISITS + 1):
        v = f"v{i}"
        v_patients = visits_df.filter(F.col("visit") == v).select("OMOP_ID", "window_start", "window_end")
        matched = (
            lipid_rx.join(v_patients, lipid_rx["lrx_pid"] == v_patients["OMOP_ID"], "inner")
            .filter((F.col("rxd") >= F.col("window_start")) & (F.col("rxd") <= F.col("window_end")))
            .groupBy("lrx_pid")
            .agg(F.lit(1).cast("integer").alias("on_lipid_med"))
            .select(F.col("lrx_pid").alias("OMOP_ID"), F.lit(v).alias("visit"), "on_lipid_med")
        )
        lipid_results.append(matched)

    lipid_all = lipid_results[0]
    for r in lipid_results[1:]:
        lipid_all = lipid_all.unionByName(r)

    df = df.join(lipid_all, ["OMOP_ID", "visit"], "left")

    tc = safe("TOTAL_CHOLESTEROL")
    hd = safe("HDL_CHOLESTEROL")
    nh = tc - hd
    nh_high = F.when(tc.isNotNull() & hd.isNotNull(), F.when(nh >= 145, 1).otherwise(0))
    lab_avail = tc.isNotNull() | hd.isNotNull()

    dys_icd = F.coalesce(safe_int("Dyslipidemia"), F.lit(0))
    lm = F.coalesce(F.col("on_lipid_med"), F.lit(0))

    df = df.withColumn(
        "OUTCOME_Dyslipidemia",
        F.when((nh_high == 1) | (dys_icd == 1) | (lm == 1), 1)
        .when(lab_avail, 0)
        .otherwise(F.lit(None).cast("integer")),
    )

    # ==================================================================
    # OUTCOME 3: MICROALBUMINURIA
    # ==================================================================
    mrn_to_omop = (
        grid.select(F.col("mrn").cast("string").alias("grid_mrn"), F.col("OMOP_ID"))
        .filter(F.col("grid_mrn").isNotNull() & F.col("OMOP_ID").isNotNull())
        .dropDuplicates(["grid_mrn"])
    )

    clarity_uacr = (
        labs.filter(
            F.col("clarity_component_code").isin(UACR_RATIO_CODES)
            & F.col("order_num_value").isNotNull()
            & (F.col("order_num_value") > 0)
        )
        .select(
            F.col("mrn").cast("string").alias("c_mrn"),
            F.col("result_date").cast("date").alias("c_date"),
            F.col("order_num_value").cast("double").alias("c_uacr"),
        )
        .filter(F.col("c_date").isNotNull())
    )

    # Cap UACR at 300 (matching Transform 2)
    clarity_uacr = clarity_uacr.withColumn(
        "c_uacr", F.when(F.col("c_uacr") > 300.0, 300.0).otherwise(F.col("c_uacr"))
    )

    clarity_uacr = (
        clarity_uacr.join(mrn_to_omop, clarity_uacr["c_mrn"] == mrn_to_omop["grid_mrn"], "inner")
        .select("OMOP_ID", "c_date", "c_uacr")
    )

    clarity_micro_results = []
    for i in range(1, NUM_VISITS + 1):
        v = f"v{i}"
        v_patients = visits_df.filter(F.col("visit") == v).select("OMOP_ID", "window_start", "window_end")
        cw = clarity_uacr.join(v_patients, "OMOP_ID", "inner").filter(
            (F.col("c_date") >= F.col("window_start")) & (F.col("c_date") <= F.col("window_end"))
        )
        cagg = (
            cw.groupBy("OMOP_ID")
            .agg(
                F.count("c_uacr").cast("integer").alias("clarity_n_uacr"),
                F.sum(F.when(F.col("c_uacr") >= UACR_THRESHOLD, 1).otherwise(0)).cast("integer").alias("clarity_n_abnormal"),
            )
            .withColumn("visit", F.lit(v))
        )
        clarity_micro_results.append(cagg)

    clarity_micro_all = clarity_micro_results[0]
    for r in clarity_micro_results[1:]:
        clarity_micro_all = clarity_micro_all.unionByName(r)

    df = df.join(clarity_micro_all, ["OMOP_ID", "visit"], "left")

    # ── OMOP repeat-testing counts (FALLBACK) ──
    micro = (
        omop_raw.filter(F.col("measurement_type") == "URINE_MICROALBUMIN")
        .select(
            F.col("PERSON_ID").cast("long").alias("mpid"),
            F.col("MEASUREMENT_DATETIME").cast("date").alias("md"),
            F.col("VALUE_SOURCE_VALUE").cast("double").alias("mv"),
        )
        .filter(F.col("mpid").isNotNull() & F.col("mv").isNotNull())
    )
    creat = (
        omop_raw.filter(F.col("measurement_type") == "URINE_CREATININE")
        .select(
            F.col("PERSON_ID").cast("long").alias("cpid"),
            F.col("MEASUREMENT_DATETIME").cast("date").alias("cd"),
            F.col("VALUE_SOURCE_VALUE").cast("double").alias("cv"),
        )
        .filter(F.col("cpid").isNotNull() & F.col("cv").isNotNull() & (F.col("cv") > 0))
    )

    uacr_omop = (
        micro.alias("m")
        .join(creat.alias("c"), (F.col("m.mpid") == F.col("c.cpid")) & (F.col("m.md") == F.col("c.cd")), "inner")
        .select(F.col("m.mpid").alias("upid"), F.col("m.md").alias("ud"), (F.col("m.mv") / F.col("c.cv")).alias("uv"))
    )

    # Cap OMOP-derived UACR at 300 as well
    uacr_omop = uacr_omop.withColumn("uv", F.when(F.col("uv") > 300.0, 300.0).otherwise(F.col("uv")))

    omop_micro_results = []
    for i in range(1, NUM_VISITS + 1):
        v = f"v{i}"
        v_patients = visits_df.filter(F.col("visit") == v).select("OMOP_ID", "window_start", "window_end")
        uw = uacr_omop.join(v_patients, F.col("upid") == F.col("OMOP_ID"), "inner").filter(
            (F.col("ud") >= F.col("window_start")) & (F.col("ud") <= F.col("window_end"))
        )
        uagg = (
            uw.groupBy("upid")
            .agg(
                F.countDistinct("ud").cast("integer").alias("omop_n_uacr"),
                F.sum(F.when(F.col("uv") >= UACR_THRESHOLD, 1).otherwise(0)).cast("integer").alias("omop_n_abnormal"),
            )
            .select(F.col("upid").alias("OMOP_ID"), F.lit(v).alias("visit"), "omop_n_uacr", "omop_n_abnormal")
        )
        omop_micro_results.append(uagg)

    omop_micro_all = omop_micro_results[0]
    for r in omop_micro_results[1:]:
        omop_micro_all = omop_micro_all.unionByName(r)

    df = df.join(omop_micro_all, ["OMOP_ID", "visit"], "left")

    # ── Resolve UACR value: Clarity (primary) → OMOP (fallback) ──
    clarity_uacr_val = safe("UACR_RATIO")
    u_micro = safe("URINE_MICROALBUMIN")
    u_creat = safe("URINE_CREATININE")
    omop_uacr_val = F.when(u_micro.isNotNull() & u_creat.isNotNull() & (u_creat > 0), u_micro / u_creat)
    uacr_val = F.coalesce(clarity_uacr_val, omop_uacr_val)

    n_u = F.coalesce(F.col("clarity_n_uacr"), F.col("omop_n_uacr"), F.lit(0))
    n_a = F.coalesce(F.col("clarity_n_abnormal"), F.col("omop_n_abnormal"), F.lit(0))

    uacr_elevated = F.when(uacr_val.isNotNull(), F.when(uacr_val >= UACR_THRESHOLD, 1).otherwise(0))

    confirmed = (
        F.when(uacr_elevated.isNull(), F.lit(None).cast("integer"))
        .when(uacr_elevated == 0, 0)
        .when(n_u >= 2, F.when(n_a >= 2, 1).otherwise(0))
        .otherwise(1)
    )

    mic_icd = F.coalesce(safe_int("Microalbuminuria"), F.lit(0))

    df = df.withColumn(
        "OUTCOME_Microalbuminuria",
        F.when(mic_icd == 1, 1)
        .when(confirmed == 1, 1)
        .when(uacr_val.isNotNull() & (confirmed == 0) & ((mic_icd == 0) | mic_icd.isNull()), 0)
        .when(uacr_val.isNull() & ((mic_icd == 0) | mic_icd.isNull()), F.lit(None).cast("integer"))
        .otherwise(F.lit(None).cast("integer")),
    )

    # ==================================================================
    # OUTCOME 4: OPTIMAL GLYCEMIC CONTROL (A1c < 7%)
    # ==================================================================
    a1c = safe("HBA1C")

    df = df.withColumn(
        "OUTCOME_Optimal_Glycemic_Control",
        F.when(a1c.isNull(), F.lit(None).cast("integer")).when(a1c < 7.0, 1).otherwise(0),
    )

    # ==================================================================
    # OUTCOME 5: INSULIN INDEPENDENCE
    # ==================================================================
    on_insulin = F.coalesce(safe_int("Insulins"), F.lit(None).cast("integer"))
    dka_flag = F.coalesce(safe_int("DKA"), F.lit(None).cast("integer"))
    a1c_ok = F.when(a1c.isNull(), F.lit(None).cast("integer")).when(a1c < 7.0, 1).otherwise(0)

    any_fails = (on_insulin == 1) | (a1c_ok == 0) | (dka_flag == 1)
    all_pass = (on_insulin == 0) & (a1c_ok == 1) & ((dka_flag == 0) | dka_flag.isNull())
    any_assessed = on_insulin.isNotNull() | a1c_ok.isNotNull() | dka_flag.isNotNull()

    df = df.withColumn(
        "OUTCOME_Insulin_Independence",
        F.when(any_fails, 0)
        .when(all_pass, 1)
        .when(~any_assessed, F.lit(None).cast("integer"))
        .otherwise(F.lit(None).cast("integer")),
    )

    # ==================================================================
    # OUTCOME 6: METFORMIN RESPONSE (metformin alone + A1c < 7%)
    # NOTE: "alone" now means not on Insulins or GLP-1RA (other classes dropped)
    # ==================================================================
    on_met = F.coalesce(safe_int("Biguanide"), F.lit(None).cast("integer"))

    on_other = F.lit(0).cast("integer")
    for mc in _ADDITIONAL_MED_CLASSES:
        if mc in all_cols:
            mc_val = F.coalesce(F.col(mc).cast("integer"), F.lit(0))
            on_other = F.when(mc_val == 1, F.lit(1)).otherwise(on_other)

    df = df.withColumn(
        "OUTCOME_Metformin_Response",
        F.when((on_met != 1) | on_met.isNull(), F.lit(None).cast("integer"))
        .when(a1c_ok.isNull(), F.lit(None).cast("integer"))
        .when(on_other == 1, 0)
        .when(a1c_ok == 1, 1)
        .otherwise(0),
    )

    # ==================================================================
    # OUTCOME 7: GLP-1RA RESPONSE
    # ==================================================================
    on_glp1 = F.coalesce(safe_int("GLP1_agonists"), F.lit(None).cast("integer"))

    df = df.withColumn(
        "OUTCOME_GLP1RA_Response",
        F.when((on_glp1 != 1) | on_glp1.isNull(), F.lit(None).cast("integer"))
        .when(a1c_ok.isNull(), F.lit(None).cast("integer"))
        .when(a1c_ok == 1, 1)
        .otherwise(0),
    )

    # ==================================================================
    # UNPIVOT OUTCOMES TO LONG FORMAT
    # ==================================================================
    outcome_cols = [
        "OUTCOME_Hypertension",
        "OUTCOME_Dyslipidemia",
        "OUTCOME_Microalbuminuria",
        "OUTCOME_Optimal_Glycemic_Control",
        "OUTCOME_Insulin_Independence",
        "OUTCOME_Metformin_Response",
        "OUTCOME_GLP1RA_Response",
    ]

    stack_expr = ", ".join([f"'{oc}', `{oc}`" for oc in outcome_cols])
    n = len(outcome_cols)

    result = df.selectExpr("OMOP_ID", "visit", f"stack({n}, {stack_expr}) as (outcome, value)")

    output.write_dataframe(result)


"""
Transform 6 – LSTM-Ready Dataset Reshape (v2)
================================================
Changes vs v1:
  - SDOH binarization applied per user config:
    * socio_social_family_support → binary (1=Adequate+, 0=Limited/None)
    * socio_financial_strain → binary (1=At Risk, 0=Low Risk)
    * socio_employment_status_parents_guardian → binary (1=Employed, 0=Not)
    * socio_education_level_parents_guardian → binary (1=HS+, 0=Below HS)
    * socio_insurance_status → categorical (Private=2, Government=1, Uninsured=0)
    * socio_physical_activity: binarization REMOVED per reviewer feedback
  - Dropped medication classes removed from output
  - Dropped measurement types excluded
  - Raw SDOH categoricals replaced with binarized versions

Output: mrn | OMOP_ID | feature | v1 | v2 | ... | v10
"""
from pyspark.sql import functions as F
from transforms.api import Input, Output, transform

NUM_VISITS = 10
VISIT_COLS = [f"v{i}" for i in range(1, NUM_VISITS + 1)]

DATA_PULL_DATE = "2025-04-30"

# Conditions to exclude from final output (already captured in outcomes)
EXCLUDE_CONDITIONS = {"Dyslipidemia", "Hypertension", "Microalbuminuria", "Hypoglycemia"}

# Features to exclude from final output
EXCLUDE_FEATURES = {"socio_physical_activity"}

# SDOH binary factor column names (from Transform 1 grid)
SDOH_BINARY = [
    "socio_adverse_childhood_experience",
    "socio_alcohol_abuse",
    "socio_drug_substance_abuse",
    "socio_food_insecurity",
    "socio_housing_instability",
    "socio_physical_sexual_abuse",
    "socio_smoking",
    "socio_transportation_barrier",
]

# Raw SDOH categorical column names (from Transform 1 grid)
SDOH_CATEGORICAL_RAW = [
    "socio_education_level_parents_guardian",
    "socio_employment_status_parents_guardian",
    "socio_financial_strain",
    "socio_insurance_status",
    "socio_physical_activity",
    "socio_social_family_support",
]

# ── SDOH Binarization Config ──

_SOCIAL_FAMILY_SUPPORT_POSITIVE = [
    "Adequate", "Strong", "Excellent",
    "Family, Friends/peers",
    "Extended family, family and friends/peers",
    "Family, Friends/peers, School",
    "Limited to Adequate",
]
_SOCIAL_FAMILY_SUPPORT_NEGATIVE = ["Limited", "None", "Minimal"]

_FINANCIAL_STRAIN_POSITIVE = ["Moderate Risk", "High Risk", "Severe", "Medium Risk"]
_FINANCIAL_STRAIN_NEGATIVE = ["Low Risk"]

_EMPLOYMENT_ACTIVE_KW = ["employed", "both employed", "student"]
_EMPLOYMENT_INACTIVE_KW = ["unemployed", "disabled", "retired", "mixed"]

_EDUCATION_POSITIVE = [
    "High School", "Some College", "College",
    "Graduate", "Post Graduate", "High School/College",
    "Some College (mother); Graduate (father)",
    "High School (Mother), Some College (Father)",
    "Mother: Some College, Father: High School",
    "Mother: College, Father: High School",
    "Mother: Some College; Father: College",
    "Mother: High School; Father: Some College",
    "Mother: College; Father: High School",
    "Mother: High School, Father: Some College",
    "Mother: Some College; Father: High School",
    "Mother: High School; Father: Some College; Primary caregiver (grandmother): Some College",
]
_EDUCATION_NEGATIVE = [
    "Elementary",
    "Some High School",
    "Mother: Elementary, Father: High School",
    "Mother: Some High School (9th Grade); Father: High School (12th Grade)",
    "Mother: High School; Father: Elementary",
    "Mother: High School (incomplete, grade 9); Father: Some College",
    "Some College (Mother), Elementary (Father)",
    "Mother: High School, Father: Elementary",
]

_INSURANCE_MAP_PRIVATE = ["Private", "Multiple"]
_INSURANCE_MAP_GOVERNMENT = ["Medicaid", "CHIP", "TRICARE WEST", "Medicare"]
_INSURANCE_MAP_UNINSURED = ["Uninsured", "Self-Pay"]


@transform(
    output=Output("/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d/transform_6_t2d_v2"),
    patient_grid=Input("/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d/transform_1_t2d_v2"),
    temporal_meas=Input("/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d/transform_2_t2d_v2"),
    temporal_meds=Input("/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d/transform_3_t2d_v2"),
    temporal_conds=Input("/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d/transform_4_t2d_v2"),
    temporal_outcomes=Input("/Texas Children's-b814dd/TCH Diabetes Project/Agentic_Workflow/Data/temporal_v2_outputs_t2d/transform_5_t2d_v2"),
)
def compute(patient_grid, temporal_meas, temporal_meds, temporal_conds,
            temporal_outcomes, output):

    grid = patient_grid.dataframe()
    meas = temporal_meas.dataframe()
    meds = temporal_meds.dataframe()
    conds = temporal_conds.dataframe()
    outcomes = temporal_outcomes.dataframe()

    grid_cols = set(grid.columns)

    # ================================================================
    # 1. MEASUREMENTS → wide (patient × feature × visit)
    # ================================================================
    meas_wide = (
        meas.select("OMOP_ID", "measurement_type", "visit", "value")
        .withColumnRenamed("measurement_type", "feature")
        .groupBy("OMOP_ID", "feature")
        .pivot("visit", VISIT_COLS)
        .agg(F.first("value"))
    )

    # ================================================================
    # 2. MEDICATIONS → wide (fill nulls with 0)
    # ================================================================
    meds_wide = (
        meds.select("OMOP_ID", "medication_class", "visit",
                     F.col("value").cast("double").alias("value"))
        .withColumnRenamed("medication_class", "feature")
        .groupBy("OMOP_ID", "feature")
        .pivot("visit", VISIT_COLS)
        .agg(F.first("value"))
    )
    for v in VISIT_COLS:
        meds_wide = meds_wide.withColumn(v, F.coalesce(F.col(v), F.lit(0.0)))

    # ================================================================
    # 3. CONDITIONS → wide (fill nulls with 0)
    #    Filter out conditions redundant with outcomes
    # ================================================================
    conds_wide = (
        conds.filter(~F.col("condition").isin(list(EXCLUDE_CONDITIONS)))
        .select("OMOP_ID", "condition", "visit",
                 F.col("value").cast("double").alias("value"))
        .withColumnRenamed("condition", "feature")
        .groupBy("OMOP_ID", "feature")
        .pivot("visit", VISIT_COLS)
        .agg(F.first("value"))
    )
    for v in VISIT_COLS:
        conds_wide = conds_wide.withColumn(v, F.coalesce(F.col(v), F.lit(0.0)))

    # ================================================================
    # 4. OUTCOMES → wide (preserve nulls — they mean "not assessed")
    # ================================================================
    outcomes_wide = (
        outcomes.select("OMOP_ID", "outcome", "visit",
                         F.col("value").cast("double").alias("value"))
        .withColumnRenamed("outcome", "feature")
        .groupBy("OMOP_ID", "feature")
        .pivot("visit", VISIT_COLS)
        .agg(F.first("value"))
    )

    # ================================================================
    # 5. STATIC FEATURES: demographics, SDOH, diabetes duration
    #    NOTE: Include ALL patients (even those without OMOP_ID) to match
    #    original pipeline behavior. Patients without OMOP_ID will have
    #    NULL temporal features but retain their static features.
    # ================================================================
    demo = (
        grid.select(
            "OMOP_ID", "mrn", "age_at_diagnosis", "sex", "patient_race",
            "ethnic_group", "language", "date_of_diagnosis", "date_of_birth",
            *[c for c in SDOH_BINARY + SDOH_CATEGORICAL_RAW if c in grid_cols]
        )
    )

    # ── Sex: Male=1, Female=0 ──
    demo = demo.withColumn("sex_encoded",
        F.when(F.upper(F.col("sex")) == "MALE", 1.0)
         .when(F.upper(F.col("sex")) == "FEMALE", 0.0)
         .otherwise(F.lit(None).cast("double"))
    )

    # ── Ethnicity: Hispanic or Latino=1, Other=0 ──
    demo = demo.withColumn("ethnicity_encoded",
        F.when(F.col("ethnic_group") == "Hispanic or Latino", 1.0)
         .when(F.col("ethnic_group").isNotNull(), 0.0)
         .otherwise(F.lit(None).cast("double"))
    )

    # ── Language: English=0, Spanish=1, Other=2 ──
    demo = demo.withColumn("language_encoded",
        F.when(F.col("language") == "English", 0.0)
         .when(F.col("language") == "Spanish", 1.0)
         .when(F.col("language") == "Other", 2.0)
         .otherwise(F.lit(None).cast("double"))
    )

    # ── Race: one-hot ──
    demo = demo.withColumn("race_white",
        F.when(F.col("patient_race") == "White", 1.0).otherwise(0.0))
    demo = demo.withColumn("race_black",
        F.when(F.col("patient_race") == "Black or African American", 1.0).otherwise(0.0))
    demo = demo.withColumn("race_asian",
        F.when(F.col("patient_race") == "Asian", 1.0).otherwise(0.0))
    demo = demo.withColumn("race_other",
        F.when(F.col("patient_race") == "Other", 1.0).otherwise(0.0))

    # ── Diabetes duration ──
    demo = demo.withColumn("diabetes_duration",
        F.round(
            F.datediff(F.lit(DATA_PULL_DATE).cast("date"),
                       F.col("date_of_diagnosis").cast("date")) / 365.25,
            2
        ).cast("double")
    )

    # ── Age at diagnosis ──
    demo = demo.withColumn("age_at_diagnosis_val",
        F.col("age_at_diagnosis").cast("double"))

    # ================================================================
    # 5b. SDOH BINARIZATION (PySpark translation of user config)
    # ================================================================

    # ── Social/Family Support: 1=Adequate+, 0=Limited/None ──
    if "socio_social_family_support" in grid_cols:
        demo = demo.withColumn("socio_social_family_support_binary",
            F.when(F.col("socio_social_family_support").isin(_SOCIAL_FAMILY_SUPPORT_POSITIVE), 1.0)
             .when(F.col("socio_social_family_support").isin(_SOCIAL_FAMILY_SUPPORT_NEGATIVE), 0.0)
             .otherwise(F.lit(None).cast("double"))
        )

    # ── Financial Strain: 1=At Risk, 0=Low Risk ──
    if "socio_financial_strain" in grid_cols:
        demo = demo.withColumn("socio_financial_strain_binary",
            F.when(F.col("socio_financial_strain").isin(_FINANCIAL_STRAIN_POSITIVE), 1.0)
             .when(F.col("socio_financial_strain").isin(_FINANCIAL_STRAIN_NEGATIVE), 0.0)
             .otherwise(F.lit(None).cast("double"))
        )

    # ── Parental Employment: 1=Employed, 0=Not Employed (keyword match) ──
    if "socio_employment_status_parents_guardian" in grid_cols:
        lower_emp = F.lower(F.col("socio_employment_status_parents_guardian"))

        # Build inactive keyword conditions (check first to handle "mixed" etc.)
        inactive_expr = F.lit(False)
        for kw in _EMPLOYMENT_INACTIVE_KW:
            inactive_expr = inactive_expr | lower_emp.contains(kw)

        # Build active keyword conditions
        active_expr = F.lit(False)
        for kw in _EMPLOYMENT_ACTIVE_KW:
            active_expr = active_expr | lower_emp.contains(kw)

        demo = demo.withColumn("socio_parental_employment_binary",
            F.when(F.col("socio_employment_status_parents_guardian").isNull(), F.lit(None).cast("double"))
             .when(inactive_expr, 0.0)
             .when(active_expr, 1.0)
             .otherwise(F.lit(None).cast("double"))
        )

    # ── Parental Education: 1=HS or higher, 0=Below HS ──
    if "socio_education_level_parents_guardian" in grid_cols:
        demo = demo.withColumn("socio_parental_education_binary",
            F.when(F.col("socio_education_level_parents_guardian").isin(_EDUCATION_POSITIVE), 1.0)
             .when(F.col("socio_education_level_parents_guardian").isin(_EDUCATION_NEGATIVE), 0.0)
             .otherwise(F.lit(None).cast("double"))
        )

    # ── Insurance Status: Private=2, Government=1, Uninsured=0 ──
    if "socio_insurance_status" in grid_cols:
        demo = demo.withColumn("socio_insurance_category",
            F.when(F.col("socio_insurance_status").isin(_INSURANCE_MAP_PRIVATE), 2.0)
             .when(F.col("socio_insurance_status").isin(_INSURANCE_MAP_GOVERNMENT), 1.0)
             .when(F.col("socio_insurance_status").isin(_INSURANCE_MAP_UNINSURED), 0.0)
             .otherwise(F.lit(None).cast("double"))
        )

    # ================================================================
    # 5c. Build static feature rows
    # ================================================================
    static_features = {
        "age_at_diagnosis": "age_at_diagnosis_val",
        "sex": "sex_encoded",
        "ethnicity_hispanic": "ethnicity_encoded",
        "language": "language_encoded",
        "race_white": "race_white",
        "race_black": "race_black",
        "race_asian": "race_asian",
        "race_other": "race_other",
        "diabetes_duration": "diabetes_duration",
    }

    # Add SDOH binary factors (raw 0/1 from LLM extraction)
    for sdoh_col in SDOH_BINARY:
        if sdoh_col in grid_cols:
            static_features[sdoh_col] = sdoh_col

    # Add NEW binarized SDOH columns (replacing raw categoricals)
    binarized_sdoh_cols = {
        "socio_social_family_support_binary": "socio_social_family_support_binary",
        "socio_financial_strain_binary": "socio_financial_strain_binary",
        "socio_parental_employment_binary": "socio_parental_employment_binary",
        "socio_parental_education_binary": "socio_parental_education_binary",
        "socio_insurance_category": "socio_insurance_category",
    }
    for feat_name, src_col in binarized_sdoh_cols.items():
        if src_col in demo.columns:
            static_features[feat_name] = src_col

    # socio_physical_activity DROPPED — 99.6% missing, not useful

    static_rows = []
    for feat_name, src_col in static_features.items():
        if src_col not in demo.columns:
            continue
        feat_df = demo.select(
            "mrn",
            "OMOP_ID",
            F.lit(feat_name).alias("feature"),
            F.col(src_col).cast("double").alias("_val"),
        )
        for v in VISIT_COLS:
            feat_df = feat_df.withColumn(v, F.col("_val"))
        feat_df = feat_df.drop("_val")
        static_rows.append(feat_df)

    # ================================================================
    # 6. UNION ALL FEATURE TYPES
    # ================================================================
    # Temporal features (keyed by OMOP_ID — only patients with OMOP_ID)
    temporal_cols = ["OMOP_ID", "feature"] + VISIT_COLS

    all_temporal = meas_wide.select(temporal_cols)
    for part in [meds_wide, conds_wide, outcomes_wide]:
        all_temporal = all_temporal.unionByName(part.select(temporal_cols))

    # Join mrn onto temporal features
    patient_mrn = (
        grid.select("OMOP_ID", "mrn")
        .filter(F.col("OMOP_ID").isNotNull())
        .dropDuplicates(["OMOP_ID"])
    )
    all_temporal_with_mrn = all_temporal.join(patient_mrn, "OMOP_ID", "left")

    # Static features already have both mrn and OMOP_ID (including NULL OMOP_ID patients)
    select_cols = ["mrn", "OMOP_ID", "feature"] + VISIT_COLS

    all_features = all_temporal_with_mrn.select(select_cols)
    for part in static_rows:
        all_features = all_features.unionByName(part.select(select_cols))

    # ================================================================
    # 7. FINAL OUTPUT — all patients including those without OMOP_ID
    # ================================================================
    final = all_features.select(["mrn", "OMOP_ID", "feature"] + VISIT_COLS)
    final = final.orderBy("mrn", "feature")

    output.write_dataframe(final)
