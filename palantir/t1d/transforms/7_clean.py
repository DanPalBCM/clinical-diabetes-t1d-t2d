from pyspark.sql import functions as F
from pyspark.sql import Window
from transforms.api import transform, Input, Output
from pyspark.sql.types import DoubleType, BooleanType, StringType, IntegerType

@transform(
    input_df=Input("ri.foundry.main.dataset.xxxxx"), # output from step 4
    output_df=Output("ri.foundry.main.dataset.xxxxx") # FInal dataframe
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
    
    # Write the cleaned dataframe to output
    output_df.write_dataframe(cleaned_df)