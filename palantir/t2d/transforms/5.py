from pyspark.sql import functions as F
from pyspark.sql import Window
from transforms.api import Input, Output, transform


@transform(
    output_dataset=Output("ri.foundry.main.dataset.xxxxx"),
    t2d_enhanced_omop=Input("ri.foundry.main.dataset.xxxxx"),
    omop_phi_measurement=Input("ri.foundry.main.dataset.xxxxx")
)
def compute(t2d_enhanced_omop, omop_phi_measurement, output_dataset):
    # Step 1: Get distinct T2D patient IDs
    t2d_patients = (
        t2d_enhanced_omop.dataframe()
        .select(F.col("OMOP_ID").cast("long").alias("PERSON_ID"))
        .distinct()
    )
    
    # Step 2: Join measurements with T2D patients and pre-filter
    measurements_with_upper = (
        omop_phi_measurement.dataframe()
        .select(
            F.col("PERSON_ID").cast("long").alias("PERSON_ID"),
            "MEASUREMENT_DATETIME",
            "MEASUREMENT_SOURCE_VALUE",
            "UNIT_SOURCE_VALUE",
            "VALUE_SOURCE_VALUE"
        )
        .join(t2d_patients, on="PERSON_ID", how="inner")
        .filter(
            (F.col("MEASUREMENT_SOURCE_VALUE").isNotNull()) &
            (F.col("MEASUREMENT_SOURCE_VALUE") != "")
        )
        .withColumn("measurement_upper", F.upper(F.col("MEASUREMENT_SOURCE_VALUE")))
        .withColumn("unit_upper", F.upper(F.coalesce(F.col("UNIT_SOURCE_VALUE"), F.lit(""))))
    )
    
    # Step 3: Categorize measurements using when-otherwise chain
    categorized_measurements = measurements_with_upper.withColumn(
        "measurement_type",
        F.when(
            # Exact match for MICROALBUMIN, RANDOM URINE (W/CREATININE)|MICROALBUMIN|14957-5
            F.col("MEASUREMENT_SOURCE_VALUE") == "MICROALBUMIN, RANDOM URINE (W/CREATININE)|MICROALBUMIN|14957-5",
            F.lit("URINE_MICROALBUMIN")
        )
        .when(
            # Exact match for MICROALBUMIN, RANDOM URINE (W/CREATININE)|CREATININE, RANDOM URINE|2161-8
            F.col("MEASUREMENT_SOURCE_VALUE") == "MICROALBUMIN, RANDOM URINE (W/CREATININE)|CREATININE, RANDOM URINE|2161-8",
            F.lit("URINE_CREATININE")
        )
        .when(
            # Beta Hydroxybutyrate - matching common variations
            (
                # Full name variations
                F.col("measurement_upper").contains("BETA HYDROXYBUTYRATE") |
                F.col("measurement_upper").contains("BETA-HYDROXYBUTYRATE") |
                F.col("measurement_upper").contains("B HYDROXYBUTYRATE") |
                F.col("measurement_upper").contains("B-HYDROXYBUTYRATE") |
                # Abbreviations
                F.col("measurement_upper").contains("BOHB") |
                F.col("measurement_upper").contains("B-OHB") |
                F.col("measurement_upper").contains("BOHBUTYRATE") |
                # Beta ketones
                F.col("measurement_upper").contains("BETA KETONE") |
                F.col("measurement_upper").contains("BETA-KETONE") |
                # 3-hydroxybutyrate (scientific name)
                F.col("measurement_upper").contains("3-HYDROXYBUTYRATE") |
                F.col("measurement_upper").contains("3 HYDROXYBUTYRATE") |
                # BHB abbreviation (excluding TBHB)
                (F.col("measurement_upper").contains("BHB") & 
                 ~F.col("measurement_upper").contains("TBHB"))
            ),
            F.lit("BETA_HYDROXYBUTYRATE")
        )
        .when(
            # HbA1c variations
            F.col("measurement_upper").contains("HBA1C") |
            F.col("measurement_upper").contains("HB A1C") |
            F.col("measurement_upper").contains("HGB A1C") |
            F.col("measurement_upper").contains("HEMOGLOBIN A1C") |
            F.col("measurement_upper").contains("GLYCOHEMOGLOBIN") |
            F.col("measurement_upper").contains("GLYCOSYLATED HEMOGLOBIN") |
            F.col("measurement_upper").contains("GLYCATED HEMOGLOBIN") |
            F.col("measurement_upper").contains("GLYCATED HGB") |
            F.col("measurement_upper").contains("GLYCOSYLATED HGB") |
            (F.col("measurement_upper").contains("A1C") &
             ~F.col("measurement_upper").contains("CALC") &
             ~F.col("measurement_upper").contains("ZN")) |
            (F.col("measurement_upper") == "A1C") |
            F.col("measurement_upper").startswith("A1C ") |
            F.col("measurement_upper").contains("HGA1C"),
            F.lit("HBA1C")
        )
        .when(
            # Glucose (all types - consolidated category)
            F.col("measurement_upper").contains("GLUCOSE") &
            ~F.col("measurement_upper").contains("URINE") &
            ~F.col("measurement_upper").contains("2H") &
            ~F.col("measurement_upper").contains("2-H") &
            ~F.col("measurement_upper").contains("2 H") &
            ~F.col("measurement_upper").contains("2 HOUR") &
            ~F.col("measurement_upper").contains("2-HOUR") &
            ~F.col("measurement_upper").contains("TWO HOUR") &
            ~F.col("measurement_upper").contains("OGTT") &
            ~F.col("measurement_upper").contains("GLUCOSE TOLERANCE"),
            F.lit("GLUCOSE")
        )
        .when(
            # Height
            F.col("measurement_upper").contains("HEIGHT") |
            (F.col("measurement_upper") == "HT") |
            F.col("measurement_upper").startswith("HT ") |
            F.col("measurement_upper").endswith(" HT") |
            F.col("measurement_upper").contains("BODY HEIGHT") |
            F.col("measurement_upper").contains("STANDING HEIGHT"),
            F.lit("HEIGHT")
        )
        .when(
            # Weight
            F.col("measurement_upper").contains("WEIGHT") |
            (F.col("measurement_upper") == "WT") |
            F.col("measurement_upper").startswith("WT ") |
            F.col("measurement_upper").endswith(" WT") |
            F.col("measurement_upper").contains("BODY WEIGHT"),
            F.lit("WEIGHT")
        )
        .when(
            # Waist Circumference
            F.col("measurement_upper").contains("WAIST") |
            F.col("measurement_upper").contains("ABDOMINAL CIRCUMFERENCE") |
            F.col("measurement_upper").contains("WAIST CIRCUMFERENCE"),
            F.lit("WAIST_CIRCUMFERENCE")
        )
        .when(
            # BMI
            (F.col("measurement_upper").contains("BMI") |
             F.col("measurement_upper").contains("BODY MASS INDEX")) &
            ~F.col("measurement_upper").contains("PERCENTILE") &
            ~F.col("measurement_upper").contains("PERCENT") &
            ~F.col("measurement_upper").contains("TILE"),
            F.lit("BMI")
        )
        .when(
            # BMI Percentile
            (F.col("measurement_upper").contains("BMI") &
             F.col("measurement_upper").contains("PERCENTILE")) |
            (F.col("measurement_upper").contains("BMI") &
             F.col("measurement_upper").contains("ILE")) |
            F.col("measurement_upper").contains("BMI PERCENT") |
            (F.col("measurement_upper").contains("BODY MASS INDEX") &
             F.col("measurement_upper").contains("PERCENTILE")),
            F.lit("BMI_PERCENTILE")
        )
        .when(
            # Total Cholesterol - exact match against approved source values only
            F.col("MEASUREMENT_SOURCE_VALUE").isin(
                "LIPID PANEL, STANDARD|CHOLESTEROL, TOTAL|2093-3",
                "LIPID PANEL|CHOLESTEROL, TOTAL|2093-3",
                "LIPID PANEL|CHOLESTEROL|LABLIPIDS",
                "LIPID PANEL|CHOLESTEROL|2093-3",
                "LIPID PANEL (REFL)|CHOLESTEROL, TOTAL|2093-3",
                "LIPID PANEL|CHOLESTEROL, TOTAL|LABLIPIDS",
                "LIPID PANEL, NONFASTING W/O TRIGLYCERIDES|CHOLESTEROL, TOTAL|2093-3",
                "CHOLESTEROL|CHOLESTEROL|LABCHOL",
                "ADVANCED LIPID PANEL, CARDIO IQ(R)|CHOLESTEROL, TOTAL|2093-3",
                "CHOLESTEROL|CHOLESTEROL, TOTAL|2093-3",
                "LIPID PANEL + TRANSAMINASES (CENTER)|CHOLESTEROL|2093-3",
                "CHOLESTEROL, TOTAL|CHOLESTEROL, TOTAL|2093-3"
            ),
            F.lit("TOTAL_CHOLESTEROL")
        )
        .when(
            # HDL Cholesterol - strict matching for "HDL CHOLESTEROL" or "CHOLESTEROL HDL"
            (F.col("measurement_upper").contains("HDL CHOLESTEROL") |
             F.col("measurement_upper").contains("CHOLESTEROL HDL")) &
            ~F.col("measurement_upper").contains("NON-HDL") &
            ~F.col("measurement_upper").contains("NON HDL") &
            ~F.col("measurement_upper").contains("RATIO") &
            ~F.col("measurement_upper").contains("/HDL") &
            ~F.col("measurement_upper").contains("TO HDL") &
            ~F.col("measurement_upper").contains("CHOLESTEROL/HDL") &
            ~F.col("measurement_upper").contains("CHOL/HDL") &
            ~(F.col("unit_upper") == "(CALC)") &
            ~F.col("unit_upper").contains("RATIO"),
            F.lit("HDL_CHOLESTEROL")
        )
        .when(
            # LDL Cholesterol - strict matching for "LDL CHOLESTEROL" or "CHOLESTEROL LDL"
            (F.col("measurement_upper").contains("LDL CHOLESTEROL") |
             F.col("measurement_upper").contains("CHOLESTEROL LDL")) &
            ~F.col("measurement_upper").contains("RATIO") &
            ~F.col("measurement_upper").contains("/LDL") &
            ~F.col("measurement_upper").contains("TO LDL") &
            ~(F.col("unit_upper") == "(CALC)") &
            ~F.col("unit_upper").contains("RATIO"),
            F.lit("LDL_CHOLESTEROL")
        )
        .when(
            # Triglycerides
            F.col("measurement_upper").contains("TRIGLYCERIDE") |
            F.col("measurement_upper").contains("TRIG"),
            F.lit("TRIGLYCERIDES")
        )
        .when(
            # Systolic Blood Pressure
            F.col("measurement_upper").contains("SYSTOLIC") |
            F.col("measurement_upper").contains("SBP") |
            (F.col("measurement_upper") == "BP SYSTOLIC") |
            F.col("measurement_upper").contains("BLOOD PRESSURE SYSTOLIC"),
            F.lit("SYSTOLIC_BLOOD_PRESSURE")
        )
        .when(
            # Diastolic Blood Pressure
            F.col("measurement_upper").contains("DIASTOLIC") |
            F.col("measurement_upper").contains("DBP") |
            (F.col("measurement_upper") == "BP DIASTOLIC") |
            F.col("measurement_upper").contains("BLOOD PRESSURE DIASTOLIC"),
            F.lit("DIASTOLIC_BLOOD_PRESSURE")
        )
        .when(
            # Serum C-Peptide
            F.col("measurement_upper").contains("C-PEPTIDE") |
            F.col("measurement_upper").contains("C PEPTIDE") |
            F.col("measurement_upper").contains("CPEPTIDE") |
            F.col("measurement_upper").contains("CONNECTING PEPTIDE"),
            F.lit("SERUM_C_PEPTIDE")
        )
        .when(
            # ALT
            (F.col("measurement_upper").contains("ALT") |
             F.col("measurement_upper").contains("ALANINE AMINOTRANSFERASE") |
             F.col("measurement_upper").contains("SGPT")) &
            ~F.col("measurement_upper").contains("SALT") &
            ~F.col("measurement_upper").contains("HALT"),
            F.lit("ALT")
        )
        .when(
            # AST
            (F.col("measurement_upper").contains("AST") |
             F.col("measurement_upper").contains("ASPARTATE AMINOTRANSFERASE") |
             F.col("measurement_upper").contains("SGOT")) &
            ~F.col("measurement_upper").contains("FAST") &
            ~F.col("measurement_upper").contains("BLAST"),
            F.lit("AST")
        )
        .when(
            # BUN
            F.col("measurement_upper").contains("BUN") |
            F.col("measurement_upper").contains("BLOOD UREA NITROGEN") |
            F.col("measurement_upper").contains("UREA NITROGEN"),
            F.lit("BUN")
        )
        .when(
            # Serum Creatinine
            F.col("measurement_upper").contains("CREATININE") &
            ~F.col("measurement_upper").contains("URINE") &
            ~F.col("measurement_upper").contains("RATIO") &
            ~F.col("measurement_upper").contains("CLEARANCE"),
            F.lit("SERUM_CREATININE")
        )
        .when(
            # eGFR - only from metabolic panels and exclude "W/O EGFR" or "WITHOUT EGFR"
            (
                (F.col("measurement_upper").contains("BASIC METABOLIC PANEL") |
                 F.col("measurement_upper").contains("COMPREHENSIVE METABOLIC") |
                 F.col("measurement_upper").contains("BASIC METABOLIC PNL") |
                 F.col("measurement_upper").contains("COMPREHENSIVE METABOLIC PNL")) &
                F.col("measurement_upper").contains("EGFR") &
                ~F.col("measurement_upper").contains("W/O EGFR") &
                ~F.col("measurement_upper").contains("WITHOUT EGFR")
            ),
            F.lit("EGFR")
        )
        .when(
            # Serum Cystatin C
            F.col("measurement_upper").contains("CYSTATIN") |
            F.col("measurement_upper").contains("CYSTATIN C") |
            F.col("measurement_upper").contains("CYSTATIN-C"),
            F.lit("SERUM_CYSTATIN_C")
        )
        .when(
            # Urine Microalbumin (pattern-based matching)
            (F.col("measurement_upper").contains("MICROALBUMIN") |
             F.col("measurement_upper").contains("MICRO ALBUMIN") |
             (F.col("measurement_upper").contains("ALBUMIN") &
              F.col("measurement_upper").contains("URINE"))) &
            ~F.col("measurement_upper").contains("RATIO") &
            ~F.col("measurement_upper").contains("CREATININE"),
            F.lit("URINE_MICROALBUMIN")
        )
        .when(
            # Urine Creatinine (pattern-based matching)
            F.col("measurement_upper").contains("URINE") &
            F.col("measurement_upper").contains("CREATININE") &
            ~F.col("measurement_upper").contains("RATIO") &
            ~F.col("measurement_upper").contains("ALBUMIN"),
            F.lit("URINE_CREATININE")
        )
        .when(
            # GAD65 Antibody
            F.col("measurement_upper").contains("GAD") |
            F.col("measurement_upper").contains("GAD65") |
            F.col("measurement_upper").contains("GAD-65") |
            F.col("measurement_upper").contains("GLUTAMIC ACID DECARBOXYLASE") |
            F.col("measurement_upper").contains("ANTI-GAD"),
            F.lit("GAD65_ANTIBODY")
        )
        .when(
            # ICA512 Antibody
            F.col("measurement_upper").contains("ICA512") |
            F.col("measurement_upper").contains("ICA-512") |
            F.col("measurement_upper").contains("IA-2") |
            F.col("measurement_upper").contains("IA2") |
            F.col("measurement_upper").contains("ISLET CELL ANTIGEN") |
            F.col("measurement_upper").contains("ANTI-IA2"),
            F.lit("ICA512_ANTIBODY")
        )
        .when(
            # Insulin Antibody
            F.col("measurement_upper").contains("INSULIN") &
            (F.col("measurement_upper").contains("ANTIBOD") |
             F.col("measurement_upper").contains("IAA") |
             F.col("measurement_upper").contains("ANTI-INSULIN")),
            F.lit("INSULIN_ANTIBODY")
        )
        .when(
            # ZnT8 Antibody
            F.col("measurement_upper").contains("ZNT8") |
            F.col("measurement_upper").contains("ZNT-8") |
            F.col("measurement_upper").contains("ZINC TRANSPORTER") |
            F.col("measurement_upper").contains("ANTI-ZNT8"),
            F.lit("ZNT8_ANTIBODY")
        )
        .when(
            # Urine Ketone
            F.col("measurement_upper").contains("KETONE") &
            (F.col("measurement_upper").contains("URINE") |
             F.col("measurement_upper").contains("U-KETONE")),
            F.lit("URINE_KETONE")
        )
        .when(
            # Blood pH
            F.col("measurement_upper").contains("PH") &
            (F.col("measurement_upper").contains("BLOOD") |
             F.col("measurement_upper").contains("VENOUS") |
             F.col("measurement_upper").contains("ARTERIAL")),
            F.lit("BLOOD_PH")
        )
        .when(
            # Bicarbonate
            F.col("measurement_upper").contains("HCO3") |
            F.col("measurement_upper").contains("BICARBONATE") |
            F.col("measurement_upper").contains("CO2"),
            F.lit("BICARBONATE")
        )
        .otherwise(None)
    )
    
    # Step 4: Filter out null measurement types and sort
    result = (
        categorized_measurements
        .filter(F.col("measurement_type").isNotNull())
        .select(
            "PERSON_ID",
            "MEASUREMENT_DATETIME",
            "MEASUREMENT_SOURCE_VALUE",
            "UNIT_SOURCE_VALUE",
            "VALUE_SOURCE_VALUE",
            "measurement_type"
        )
        .orderBy("PERSON_ID", "measurement_type", "MEASUREMENT_DATETIME")
    )
    
    output_dataset.write_dataframe(result)