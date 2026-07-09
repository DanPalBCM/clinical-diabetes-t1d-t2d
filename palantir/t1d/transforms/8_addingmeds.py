from pyspark.sql import functions as F
from transforms.api import Input, Output, transform


@transform(
    output_dataset=Output("ri.foundry.main.dataset.xxxxx"),
    source_features=Input("ri.foundry.main.dataset.xxxxx"),  # Dataset with OMOP_ features
    target_dataset=Input("ri.foundry.main.dataset.xxxxx")   # Dataset to enhance
)
def compute(source_features, target_dataset, output_dataset):
    # Load the source dataset with OMOP_ features
    source_df = source_features.dataframe()
    
    # Load the target dataset to be enhanced
    target_df = target_dataset.dataframe()
    
    # Get all column names from source that start with OMOP_
    omop_columns = [col for col in source_df.columns if col.startswith("OMOP_")]
    
    # Select identifier columns and OMOP_ columns from source
    # Using PEDSNET_ID as the join key (you can change to MRN if preferred)
    source_selected = source_df.select(
        F.col("PEDSNET_ID"),
        *omop_columns
    ).distinct()  # Remove duplicates if any
    
    # Join target dataset with source features
    # Using left join to preserve all records from target dataset
    result = target_df.join(
        source_selected,
        on="PEDSNET_ID",
        how="left"
    )
    
    # Write the enhanced dataset to output
    output_dataset.write_dataframe(result)