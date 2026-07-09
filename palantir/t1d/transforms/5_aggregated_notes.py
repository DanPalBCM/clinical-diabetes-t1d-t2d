from pyspark.sql import functions as F
from transforms.api import Input, Output, transform

@transform(
    output_dataset=Output("ri.foundry.main.dataset.xxxxx"),
    patient_list=Input("ri.foundry.main.dataset.xxxxx"),
    notes_dataset=Input("ri.foundry.main.dataset.xxxxx"),
    crossref_table=Input("ri.foundry.main.dataset.xxxxx"),
    provider_dataset=Input("ri.foundry.main.dataset.xxxxx")
)
def compute(patient_list, notes_dataset, crossref_table, provider_dataset, output_dataset):
    
    # Step 1: Load patient list and prepare diagnosis dates
    patients = (
        patient_list.dataframe()
        .select(
            F.col("MRN").cast("string").alias("mrn"),
            F.col("PERSON_ID"),
            F.col("DiagnosisDate").cast("date").alias("diagnosis_date")
        )
        .filter(F.col("mrn").isNotNull() & F.col("diagnosis_date").isNotNull())
    )
    
    # Step 2: Load cross-reference table and cast IDs to strings
    crossref = (
        crossref_table.dataframe()
        .select(
            F.col("patient_id").cast("string").alias("patient_id"),
            F.col("mrn").cast("string").alias("mrn")
        )
        .filter(F.col("patient_id").isNotNull() & F.col("mrn").isNotNull())
    )
    
    # Step 3: Load provider dataset and filter for social workers
    social_worker_ids = (
        provider_dataset.dataframe()
        .filter(F.lower(F.trim(F.col("provider_type"))) == "social worker")
        .select(F.col("user_id").cast("string").alias("user_id"))
        .filter(F.col("user_id").isNotNull())
        .distinct()
    )
    
    # Step 4: Load notes with patient_id, current_author_id, and cast to string
    notes = (
        notes_dataset.dataframe()
        .select(
            F.col("patient_id").cast("string").alias("patient_id"),
            F.col("contact_date").cast("date").alias("contact_date"),
            F.col("note_text"),
            F.col("note_type_ip"),
            F.col("current_author_id").cast("string").alias("current_author_id")
        )
        .filter(F.col("patient_id").isNotNull() & F.col("contact_date").isNotNull())
    )
    
    # Step 5: Filter notes to only include those written by social workers
    notes_by_social_workers = (
        notes
        .join(social_worker_ids, notes["current_author_id"] == social_worker_ids["user_id"], how="inner")
        .select(notes["*"])  # Keep only the notes columns
    )
    
    # Step 6: Join notes with cross-reference table to get mrn
    notes_with_mrn = (
        notes_by_social_workers
        .join(crossref, on="patient_id", how="inner")
    )
    
    # Step 7: Join notes with patients
    notes_with_diagnosis = (
        notes_with_mrn
        .join(patients, on="mrn", how="inner")
    )
    
    # Step 8: Calculate days from diagnosis (NO TIME FILTER - capturing all notes)
    filtered_notes = (
        notes_with_diagnosis
        .withColumn("days_from_diagnosis", 
                   F.datediff(F.col("contact_date"), F.col("diagnosis_date")))
    )
    
    # Step 9: Aggregate notes per patient
    aggregated_notes = (
        filtered_notes
        .groupBy("mrn", "PERSON_ID")
        .agg(
            F.concat_ws("\n---\n", F.collect_list("note_text")).alias("aggregated_notes"),
            F.count("*").alias("note_count")
        )
        .select("mrn", "PERSON_ID", "aggregated_notes", "note_count")
    )
    
    # Step 10: Write output
    output_dataset.write_dataframe(aggregated_notes)