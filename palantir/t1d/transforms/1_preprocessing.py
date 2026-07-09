from pyspark.sql import functions as F
from transforms.api import Input, Output, transform


@transform(
    output_dataset=Output("ri.foundry.main.dataset.xxxxx"),
    t1d_enhanced_omop=Input("ri.foundry.main.dataset.xxxxx"),
    omop_person=Input("ri.foundry.main.dataset.xxxxx"),
)
def compute(t1d_enhanced_omop, omop_person, output_dataset):
    """
    Transform that creates new columns:
    - Copies PEDSNET_ID to person_id
    - Copies PATIENTID to PatientID (and removes original PATIENTID)
    - Renames Date_of_Dx to DiagnosisDate
    - Renames Gender to gender
    - Adds BirthDTS from OMOP person table
    """

    # Read the input dataframes
    df = t1d_enhanced_omop.dataframe()
    person_df = omop_person.dataframe()

    # Select relevant columns from person table and prepare for join
    # Using BIRTH_DATETIME if available, otherwise BIRTH_DATE
    person_birth = person_df.select(
        F.col("PERSON_ID"), F.coalesce(F.col("BIRTH_DATETIME"), F.col("BIRTH_DATE")).alias("BirthDTS")
    )

    # Add new columns and drop originals that collide
    result_df = (
        df.withColumn("person_id", F.col("PEDSNET_ID"))
        .withColumn("PatientID", F.col("PATIENTID"))
        .drop("PATIENTID")
        .withColumnRenamed("Date_of_Dx", "DiagnosisDate")
        .withColumnRenamed("Gender", "gender")
    )

    # Join with person table to get BirthDTS
    result_df = result_df.join(person_birth, result_df.PEDSNET_ID == person_birth.PERSON_ID, "left").drop(
        person_birth.PERSON_ID
    )  # Drop the duplicate PERSON_ID from join

    # Write the result to output dataset
    output_dataset.write_dataframe(result_df)
