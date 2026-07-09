from pyspark.sql import functions as F
from pyspark.sql import Window
from transforms.api import transform, Input, Output
from pyspark.sql.types import DoubleType, BooleanType, StringType, IntegerType
from datetime import datetime

@transform(
    input_df=Input("ri.foundry.main.dataset.xxxxx"),
    output_df=Output("ri.foundry.main.dataset.xxxxx")
)
def compute(input_df, output_df):
    # Load the input dataframe
    df = input_df.dataframe()
    
    # Get all column names
    all_columns = df.columns
    
    # Filter out columns that start with "_debug" and the "Unnamed_43" column
    columns_to_keep = [
        col for col in all_columns 
        if not col.startswith("_debug") and col != "Unnamed_43"
    ]
    
    # Select only the columns we want to keep
    cleaned_df = df.select(columns_to_keep)
    
    # Define the data pull date
    data_pull_date = F.lit("2025-04-30").cast("date")
    
    # Calculate diabetes duration in years
    # First ensure DiagnosisDate is in date format
    cleaned_df = cleaned_df.withColumn(
        "diabetes_duration",
        F.round(
            F.datediff(data_pull_date, F.col("date_of_diagnosis").cast("date")) / 365.25,
            2
        )
    )
    
    # Write the cleaned dataframe to output
    output_df.write_dataframe(cleaned_df)