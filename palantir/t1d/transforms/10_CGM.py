from pyspark.sql import functions as F
from pyspark.sql import Window
from transforms.api import Input, Output, transform


@transform(
    output_dataset=Output("ri.foundry.main.dataset.xxxxx"),
    dataset_a=Input("ri.foundry.main.dataset.xxxxx"),
    dataset_b=Input("ri.foundry.main.dataset.xxxxx")
)
def compute(dataset_a, dataset_b, output_dataset):
    """
    Merge CGM data from Dataset B into Dataset A (main dataset).
    Handles duplicate patients, birthday mismatches (parents entering their own DOB), 
    and aggregates metrics appropriately.
    
    KEY: Dataset B sometimes has parent birthdays instead of child birthdays.
    We match on name first, then validate with birthday rules.
    """
    
    # ========== STEP 1: Parse and Clean Dataset A ==========
    df_a = dataset_a.dataframe()
    
    # Parse "Last Name, First Name" format and remove middle names
    df_a_parsed = (
        df_a
        .withColumn("LastName_A", F.trim(F.split(F.col("Patient"), ",").getItem(0)))
        .withColumn("FirstName_temp", F.trim(F.split(F.col("Patient"), ",").getItem(1)))
        .withColumn(
            "FirstName_A",
            # Remove middle name (take only first word) and handle "Daniel-old acct" style names
            F.trim(F.regexp_replace(
                F.split(F.col("FirstName_temp"), " ").getItem(0),
                "-old acct$", ""
            ))
        )
        .withColumn("BirthDate_A", F.to_date(F.col("BirthDTS")))
        .withColumn("BirthYear_A", F.year(F.col("BirthDTS")))
        .drop("FirstName_temp")
    )
    
    # ========== STEP 2: Clean Dataset B ==========
    df_b = dataset_b.dataframe()
    
    df_b_cleaned = (
        df_b
        .withColumn(
            "FirstName_B_clean",
            # Remove middle name and handle special naming patterns
            F.trim(F.regexp_replace(
                F.split(F.trim(F.col("FirstName")), " ").getItem(0),
                "-old acct$", ""
            ))
        )
        .withColumn("LastName_B", F.trim(F.col("LastName")))
        .withColumn("BirthDate_B", F.to_date(F.col("DateOfBirth")))
        .withColumn("BirthYear_B", F.year(F.col("DateOfBirth")))
    )
    
    # ========== STEP 3: Aggregate Duplicates in Dataset B ==========
    # Group by name and DOB, aggregating measurements from the same patient
    
    # Calculate weights for weighted averages (based on NumReadings)
    window_for_weights = Window.partitionBy("FirstName_B_clean", "LastName_B", "BirthDate_B")
    
    df_b_with_weights = df_b_cleaned.withColumn(
        "TotalReadings_Group",
        F.sum("NumReadings").over(window_for_weights)
    ).withColumn(
        "ReadingWeight",
        F.when(F.col("TotalReadings_Group") > 0,
               F.col("NumReadings") / F.col("TotalReadings_Group")
        ).otherwise(0)
    )
    
    # Aggregate by patient (same name + DOB = same patient)
    df_b_aggregated = (
        df_b_with_weights
        .groupBy("FirstName_B_clean", "LastName_B", "BirthDate_B", "BirthYear_B")
        .agg(
            # Time range: earliest start, latest end
            F.min("StartTime").alias("StartTime"),
            F.max("EndTime").alias("EndTime"),
            
            # Counts: sum them up
            F.sum("DurationDays").alias("DurationDays"),
            F.sum("NumReadings").alias("NumReadings"),
            F.sum("NumLowReadings").alias("NumLowReadings"),
            F.sum("NumHighReadings").alias("NumHighReadings"),
            
            # Metrics: weighted average by NumReadings
            F.sum(F.col("MeanGlucose_mgdL") * F.col("ReadingWeight")).alias("MeanGlucose_mgdL"),
            F.sum(F.col("SD_mgdL") * F.col("ReadingWeight")).alias("SD_mgdL"),
            F.sum(F.col("CV_percent") * F.col("ReadingWeight")).alias("CV_percent"),
            F.sum(F.col("GMI") * F.col("ReadingWeight")).alias("GMI"),
            
            # Time percentages: sum them (they represent total time across all periods)
            F.sum("TimeAbove250_percent").alias("TimeAbove250_percent"),
            F.sum("TimeAbove180_percent").alias("TimeAbove180_percent"),
            F.sum("TimeInRange70_180_percent").alias("TimeInRange70_180_percent"),
            F.sum("Time181_250_percent").alias("Time181_250_percent"),
            F.sum("Time54_69_percent").alias("Time54_69_percent"),
            F.sum("TimeBelow70_percent").alias("TimeBelow70_percent"),
            F.sum("TimeBelow54_percent").alias("TimeBelow54_percent"),
            
            # Episodes: sum them
            F.sum("HypoEpisodes_Total").alias("HypoEpisodes_Total"),
            F.sum("SevereHypoEpisodes_Total").alias("SevereHypoEpisodes_Total")
        )
        .withColumn(
            # Recalculate per-day metrics
            "SevereHypoEpisodes_PerDay",
            F.when(F.col("DurationDays") > 0,
                   F.col("SevereHypoEpisodes_Total") / F.col("DurationDays")
            ).otherwise(F.lit(None))
        )
    )
    
    # ========== STEP 4: Handle Birthday Mismatches ==========
    # Create a window to check for name matches with different birthdays
    window_name_match = Window.partitionBy(
        F.upper(df_b_aggregated.FirstName_B_clean),
        F.upper(df_b_aggregated.LastName_B)
    )
    
    df_b_with_match_info = df_b_aggregated.withColumn(
        "NumBirthdaysForName",
        F.count("BirthDate_B").over(window_name_match)
    ).withColumn(
        "IsPediatric",
        F.col("BirthYear_B") >= 2000  # Pediatric if born 2000 or later
    )
    
    # ========== STEP 5: Join Dataset A with Dataset B ==========
    # Use case-insensitive name matching
    join_condition = (
        (F.upper(df_a_parsed.FirstName_A) == F.upper(df_b_with_match_info.FirstName_B_clean)) &
        (F.upper(df_a_parsed.LastName_A) == F.upper(df_b_with_match_info.LastName_B))
    )
    
    # Initial join on names only (LEFT JOIN to keep all patients from A)
    df_joined_names = df_a_parsed.join(
        df_b_with_match_info,
        join_condition,
        how="left"
    )
    
    # ========== STEP 6: Apply Birthday Matching Rules ==========
    # Handle cases where parents may have entered their DOB instead of child's
    df_with_flags = df_joined_names.withColumn(
        "BirthdayMatch",
        F.col("BirthDate_A") == F.col("BirthDate_B")
    ).withColumn(
        "A_IsPediatric",
        F.col("BirthYear_A") >= 2000
    ).withColumn(
        "B_IsPediatric",
        F.col("BirthYear_B") >= 2000
    )
    
    # Determine if we should use this B record
    df_with_match_decision = df_with_flags.withColumn(
        "UseThisMatch",
        F.when(
            # If birthdays match exactly, definitely use it
            F.col("BirthdayMatch") == True,
            True
        ).when(
            # If B is null (no match found), don't use it
            F.col("BirthDate_B").isNull(),
            False
        ).when(
            # If names match but one is adult and one is pediatric
            # Likely parent entered their DOB instead of child's - use pediatric record
            # Only use this B record if B is pediatric and A is adult
            (F.col("A_IsPediatric") == False) & (F.col("B_IsPediatric") == True),
            True
        ).when(
            # If names match but one is adult and one is pediatric
            # Don't use this B record if B is adult and A is pediatric
            (F.col("A_IsPediatric") == True) & (F.col("B_IsPediatric") == False),
            False
        ).otherwise(
            # Default: only use if birthdays match
            F.col("BirthdayMatch") == True
        )
    )
    
    # Flag invalid duplicates (same name, both pediatric, different DOBs)
    df_final = df_with_match_decision.withColumn(
        "Patient_invalid_duplicate",
        F.when(
            (F.col("UseThisMatch") == False) &
            (F.col("BirthDate_B").isNotNull()) &
            (F.col("A_IsPediatric") == True) &
            (F.col("B_IsPediatric") == True) &
            (F.col("BirthdayMatch") == False),
            True
        ).otherwise(False)
    )
    
    # ========== STEP 7: Handle Multiple Matches (Deduplication) ==========
    # One patient in A might match multiple patients in B with different birthdays
    # Keep only the best match per patient
    window_spec = Window.partitionBy("Patient", "BirthDTS").orderBy(
        F.col("BirthdayMatch").desc(),  # Prefer exact birthday matches
        F.col("UseThisMatch").desc(),   # Then prefer valid matches
        F.col("NumReadings").desc()     # Then prefer records with more data
    )
    
    df_deduplicated = df_final.withColumn(
        "row_num",
        F.row_number().over(window_spec)
    ).filter(F.col("row_num") == 1).drop("row_num")
    
    # ========== STEP 8: Null Out Invalid Matches and Select Final Columns ==========
    # Instead of filtering out mismatched records, null out B columns when UseThisMatch is False
    # This ensures ALL patients from Dataset A are retained
    result = df_deduplicated.select(
        # All original columns from Dataset A
        F.col("Patient"),
        F.col("BirthDTS"),
        *[col for col in df_a.columns if col not in ["Patient", "BirthDTS"]],
        
        # CGM metrics from Dataset B - null them out if UseThisMatch is False
        F.when(F.col("UseThisMatch") == True, F.col("StartTime")).alias("StartTime"),
        F.when(F.col("UseThisMatch") == True, F.col("EndTime")).alias("EndTime"),
        F.when(F.col("UseThisMatch") == True, F.col("DurationDays")).alias("DurationDays"),
        F.when(F.col("UseThisMatch") == True, F.col("NumReadings")).alias("NumReadings"),
        F.when(F.col("UseThisMatch") == True, F.col("NumLowReadings")).alias("NumLowReadings"),
        F.when(F.col("UseThisMatch") == True, F.col("NumHighReadings")).alias("NumHighReadings"),
        F.when(F.col("UseThisMatch") == True, F.col("MeanGlucose_mgdL")).alias("MeanGlucose_mgdL"),
        F.when(F.col("UseThisMatch") == True, F.col("SD_mgdL")).alias("SD_mgdL"),
        F.when(F.col("UseThisMatch") == True, F.col("CV_percent")).alias("CV_percent"),
        F.when(F.col("UseThisMatch") == True, F.col("GMI")).alias("GMI"),
        F.when(F.col("UseThisMatch") == True, F.col("TimeAbove250_percent")).alias("TimeAbove250_percent"),
        F.when(F.col("UseThisMatch") == True, F.col("TimeAbove180_percent")).alias("TimeAbove180_percent"),
        F.when(F.col("UseThisMatch") == True, F.col("TimeInRange70_180_percent")).alias("TimeInRange70_180_percent"),
        F.when(F.col("UseThisMatch") == True, F.col("Time181_250_percent")).alias("Time181_250_percent"),
        F.when(F.col("UseThisMatch") == True, F.col("Time54_69_percent")).alias("Time54_69_percent"),
        F.when(F.col("UseThisMatch") == True, F.col("TimeBelow70_percent")).alias("TimeBelow70_percent"),
        F.when(F.col("UseThisMatch") == True, F.col("TimeBelow54_percent")).alias("TimeBelow54_percent"),
        F.when(F.col("UseThisMatch") == True, F.col("HypoEpisodes_Total")).alias("HypoEpisodes_Total"),
        F.when(F.col("UseThisMatch") == True, F.col("SevereHypoEpisodes_Total")).alias("SevereHypoEpisodes_Total"),
        F.when(F.col("UseThisMatch") == True, F.col("SevereHypoEpisodes_PerDay")).alias("SevereHypoEpisodes_PerDay"),
        
        # Quality flags
        F.col("Patient_invalid_duplicate")
    )
    
    output_dataset.write_dataframe(result)