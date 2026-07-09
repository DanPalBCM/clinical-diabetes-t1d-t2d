"""
Palantir Foundry PySpark Transform 1: T2D Patient Demographics & Diagnosis Date
Purpose: Create base patient dataset with demographics, diagnosis date, and age at diagnosis
"""

from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output


@transform_df(
    Output("ri.foundry.main.dataset.xxxxx"),
    a1c_dataset=Input("ri.foundry.main.dataset.xxxxx"),
    demographics=Input("ri.foundry.main.dataset.xxxxx"),
    crossreference=Input("ri.foundry.main.dataset.xxxxx")
)
def compute(a1c_dataset, demographics, crossreference):
    """
    Create patient-level base dataset with demographics and diagnosis information
    """

    # ============================================================================
    # STEP 0: Manual corrections by review done on 2/24/2026 by Mustafa and Daniel
    # - Remove patients MASKED and MASKED (confirmed correct, excluded per review)
    # - Correct diagnosis date for patient MASKED to 2016-07-17, age at diagnosis to 13
    # ============================================================================
    patients_to_remove = ["MASKED", "MASKED"]
    a1c_dataset = a1c_dataset.filter(
        ~F.col("mrn").cast("string").isin(patients_to_remove)
    )

    a1c_dataset = a1c_dataset.withColumn(
        "date_of_diagnosis",
        F.when(
            F.col("mrn").cast("string") == "MASKED",
            F.to_date(F.lit("2016-07-17"))
        ).otherwise(F.col("date_of_diagnosis"))
    ).withColumn(
        "age_at_diagnosis",
        F.when(
            F.col("mrn").cast("string") == "MASKED",
            F.lit(13)
        ).otherwise(F.col("age_at_diagnosis"))
    )

    # ============================================================================
    # STEP 1: Get unique patients with earliest date_of_diagnosis and age_at_diagnosis
    # ============================================================================
    # Since A1C dataset has multiple rows per patient, get the minimum values
    unique_patients = a1c_dataset.groupBy(
        F.col("mrn").cast("string").alias("mrn")
    ).agg(
        F.min("age_at_diagnosis").alias("age_at_diagnosis"),
        F.min("date_of_diagnosis").alias("date_of_diagnosis")
    )
    
    # ============================================================================
    # STEP 2: Prepare demographics with preprocessing
    # ============================================================================
    demographics_clean = demographics.select(
        F.col("mrn").cast("string").alias("mrn"),
        F.col("date_of_birth"),
        F.col("ethnic_group"),
        F.col("language"),
        F.col("sex"),
        F.col("patient_race")
    )
    
    # Preprocess ethnic_group
    demographics_clean = demographics_clean.withColumn(
        "ethnic_group",
        F.when(F.col("ethnic_group") == "Hispanic or Latino", "Hispanic or Latino")
         .otherwise("Other")
    )
    
    # Preprocess language
    demographics_clean = demographics_clean.withColumn(
        "language",
        F.when(F.col("language") == "English", "English")
         .when(F.col("language") == "Spanish", "Spanish")
         .otherwise("Other")
    )
    
    # Preprocess patient_race - using array_contains for ARRAY<STRING> type
    demographics_clean = demographics_clean.withColumn(
        "patient_race",
        F.when(F.array_contains(F.col("patient_race"), "White"), "White")
         .when(F.array_contains(F.col("patient_race"), "Black or African American"), 
               "Black or African American")
         .when(F.array_contains(F.col("patient_race"), "Asian"), "Asian")
         .otherwise("Other")
    )
    
    # ============================================================================
    # STEP 3: Prepare crossreference table for OMOP_ID
    # ============================================================================
    crossreference_clean = crossreference.select(
        F.col("PAT_MRN_ID").cast("string").alias("mrn"),
        F.col("PEDSNET_ID").cast("string").alias("OMOP_ID")
    )
    
    # ============================================================================
    # STEP 4: Join unique patients with demographics and crossreference
    # ============================================================================
    patient_base = unique_patients.join(
        demographics_clean, 
        on="mrn", 
        how="inner"
    ).join(
        crossreference_clean,
        on="mrn",
        how="left"  # Using left join in case some MRNs don't have OMOP_ID
    )
    
    # ============================================================================
    # STEP 5: Calculate Age_at_diagnosis_min (age in whole years)
    # ============================================================================
    patient_base = patient_base.withColumn(
        "Age_at_diagnosis_min",
        F.floor(F.col("age_at_diagnosis"))
    )
    
    # ============================================================================
    # STEP 6: Pre-calculate target dates for next transforms
    # ============================================================================
    patient_base = patient_base.withColumn(
        "target_date_diagnosis",
        F.col("date_of_diagnosis")
    ).withColumn(
        "target_date_2yr",
        F.expr("date_add(date_of_diagnosis, 730)")  # 2 years
    ).withColumn(
        "target_date_5yr",
        F.expr("date_add(date_of_diagnosis, 1826)")  # 5 years
    )
    
    return patient_base.orderBy("mrn")