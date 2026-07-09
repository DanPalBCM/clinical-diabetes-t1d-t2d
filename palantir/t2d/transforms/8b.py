from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType
from transforms.api import transform, Input, Output

@transform(
    measurement_dataset=Input("ri.foundry.main.dataset.xxxxx"),
    socio_factors_df=Input("ri.foundry.main.dataset.xxxxx"),
    output_df=Output("ri.foundry.main.dataset.xxxxx")
)
def compute(measurement_dataset, socio_factors_df, output_df):
    # Load dataframes
    measurement_df = measurement_dataset.dataframe()
    socio_df = socio_factors_df.dataframe()
    
    # ========================================================================
    # Clean and prepare JSON data from val_llm column
    # ========================================================================
    
    # Remove markdown code fences if present and extract JSON
    socio_df = socio_df.withColumn(
        "json_clean",
        F.regexp_replace(F.col("val_llm"), "```json|```", "")
    )
    
    # Trim whitespace
    socio_df = socio_df.withColumn(
        "json_clean",
        F.trim(F.col("json_clean"))
    )
    
    # ========================================================================
    # Extract binary factors (convert boolean to 0/1 indicators)
    # ========================================================================
    
    socio_df = socio_df.withColumn(
        "socio_adverse_childhood_experience",
        F.when(F.get_json_object(F.col("json_clean"), "$.binary_factors.adverse_childhood_experience.value") == "true", 1).otherwise(0)
    )
    
    socio_df = socio_df.withColumn(
        "socio_alcohol_abuse",
        F.when(F.get_json_object(F.col("json_clean"), "$.binary_factors.alcohol_abuse.value") == "true", 1).otherwise(0)
    )
    
    socio_df = socio_df.withColumn(
        "socio_drug_substance_abuse",
        F.when(F.get_json_object(F.col("json_clean"), "$.binary_factors.drug_substance_abuse.value") == "true", 1).otherwise(0)
    )
    
    socio_df = socio_df.withColumn(
        "socio_food_insecurity",
        F.when(F.get_json_object(F.col("json_clean"), "$.binary_factors.food_insecurity.value") == "true", 1).otherwise(0)
    )
    
    socio_df = socio_df.withColumn(
        "socio_housing_instability",
        F.when(F.get_json_object(F.col("json_clean"), "$.binary_factors.housing_instability.value") == "true", 1).otherwise(0)
    )
    
    socio_df = socio_df.withColumn(
        "socio_physical_sexual_abuse",
        F.when(F.get_json_object(F.col("json_clean"), "$.binary_factors.physical_sexual_abuse.value") == "true", 1).otherwise(0)
    )
    
    socio_df = socio_df.withColumn(
        "socio_smoking",
        F.when(F.get_json_object(F.col("json_clean"), "$.binary_factors.smoking.value") == "true", 1).otherwise(0)
    )
    
    socio_df = socio_df.withColumn(
        "socio_transportation_barrier",
        F.when(F.get_json_object(F.col("json_clean"), "$.binary_factors.transportation_barrier.value") == "true", 1).otherwise(0)
    )
    
    # ========================================================================
    # Extract categorical factors (get value directly)
    # ========================================================================
    
    socio_df = socio_df.withColumn(
        "socio_education_level_parents_guardian",
        F.get_json_object(F.col("json_clean"), "$.categorical_factors.education_level_parents.value")
    )
    
    socio_df = socio_df.withColumn(
        "socio_employment_status_parents_guardian",
        F.get_json_object(F.col("json_clean"), "$.categorical_factors.employment_status_parents.value")
    )
    
    socio_df = socio_df.withColumn(
        "socio_financial_strain",
        F.get_json_object(F.col("json_clean"), "$.categorical_factors.financial_strain.value")
    )
    
    socio_df = socio_df.withColumn(
        "socio_insurance_status",
        F.get_json_object(F.col("json_clean"), "$.categorical_factors.insurance_status.value")
    )
    
    socio_df = socio_df.withColumn(
        "socio_physical_activity",
        F.get_json_object(F.col("json_clean"), "$.categorical_factors.physical_activity.value")
    )
    
    socio_df = socio_df.withColumn(
        "socio_social_family_support",
        F.get_json_object(F.col("json_clean"), "$.categorical_factors.social_family_support.value")
    )
    
    # ========================================================================
    # Join with measurement dataset
    # ========================================================================
    
    # Select only needed columns from socio_df
    socio_columns = ["mrn", 
                     "socio_adverse_childhood_experience",
                     "socio_alcohol_abuse",
                     "socio_drug_substance_abuse",
                     "socio_food_insecurity",
                     "socio_housing_instability",
                     "socio_physical_sexual_abuse",
                     "socio_smoking",
                     "socio_transportation_barrier",
                     "socio_education_level_parents_guardian",
                     "socio_employment_status_parents_guardian",
                     "socio_financial_strain",
                     "socio_insurance_status",
                     "socio_physical_activity",
                     "socio_social_family_support"]
    
    socio_df_clean = socio_df.select(socio_columns)
    
    # Cast mrn as string in both dataframes
    measurement_df = measurement_df.withColumn("mrn", F.col("mrn").cast(StringType()))
    socio_df_clean = socio_df_clean.withColumn("mrn", F.col("mrn").cast(StringType()))
    
    # Join on mrn
    result_df = measurement_df.join(
        socio_df_clean,
        "mrn",
        how="left"
    )
    
    # Write the result
    output_df.write_dataframe(result_df)