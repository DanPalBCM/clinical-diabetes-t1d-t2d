from pyspark.sql import functions as F
from transforms.api import Input, Output, transform


@transform(
    output_dataset=Output("ri.foundry.main.dataset.xxxxx"),
    input_dataset=Input("ri.foundry.main.dataset.xxxxx")
)
def compute(input_dataset, output_dataset):
    df = input_dataset.dataframe()

    # ============================================================================
    # BLOOD PRESSURE PROCESSING
    # ============================================================================

    bp_systolic = 'systolic_blood_pressure_at_diagnosis'
    bp_diastolic = 'diastolic_blood_pressure_at_diagnosis'

    def parse_bp_systolic(bp_string):
        if bp_string is None:
            return None
        bp_str = str(bp_string).strip()
        if '/' in bp_str:
            parts = bp_str.split('/')
            if len(parts) == 2:
                try:
                    systolic = float(parts[0].strip())
                    diastolic = float(parts[1].strip())
                    if 50 <= systolic <= 250 and 30 <= diastolic <= 150:
                        return systolic
                except Exception:
                    pass
        return None

    def parse_bp_diastolic(bp_string):
        if bp_string is None:
            return None
        bp_str = str(bp_string).strip()
        if '/' in bp_str:
            parts = bp_str.split('/')
            if len(parts) == 2:
                try:
                    systolic = float(parts[0].strip())
                    diastolic = float(parts[1].strip())
                    if 50 <= systolic <= 250 and 30 <= diastolic <= 150:
                        return diastolic
                except Exception:
                    pass
        return None

    parse_systolic_udf = F.udf(parse_bp_systolic, 'double')
    parse_diastolic_udf = F.udf(parse_bp_diastolic, 'double')

    cols = df.columns

    if bp_systolic in cols:
        # Check if systolic contains slash format by sampling
        sample_vals = df.select(bp_systolic).dropna().limit(100).collect()
        has_slash = any('/' in str(row[0]) for row in sample_vals)

        if has_slash:
            # Systolic has combined format - parse both from systolic column
            df = df.withColumn(bp_systolic, parse_systolic_udf(F.col(bp_systolic)))

            if bp_diastolic in cols:
                # Check if diastolic also has slash format
                sample_dias = df.select(bp_diastolic).dropna().limit(100).collect()
                dias_has_slash = any('/' in str(row[0]) for row in sample_dias)

                if dias_has_slash:
                    df = df.withColumn(bp_diastolic, parse_diastolic_udf(F.col(bp_diastolic)))
                else:
                    df = df.withColumn(bp_diastolic, F.col(bp_diastolic).cast('double'))
        else:
            # Already numeric
            df = df.withColumn(bp_systolic, F.col(bp_systolic).cast('double'))
            if bp_diastolic in cols:
                df = df.withColumn(bp_diastolic, F.col(bp_diastolic).cast('double'))

    elif bp_diastolic in cols:
        # Only diastolic exists and may have slash format
        sample_vals = df.select(bp_diastolic).dropna().limit(100).collect()
        has_slash = any('/' in str(row[0]) for row in sample_vals)

        if has_slash:
            df = df.withColumn(bp_diastolic, parse_diastolic_udf(F.col(bp_diastolic)))
        else:
            df = df.withColumn(bp_diastolic, F.col(bp_diastolic).cast('double'))

    # ============================================================================
    # ETHNICITY STANDARDIZATION
    # Hispanic or Latino, Other
    # ============================================================================

    if 'Ethnicity' in cols:
        df = df.withColumn(
            'Ethnicity',
            F.when(F.col('Ethnicity').isNull(), F.lit(None))
            .when(
                F.lower(F.col('Ethnicity')).contains('not hispanic') |
                F.lower(F.col('Ethnicity')).contains('not latino') |
                F.lower(F.col('Ethnicity')).contains('not latina'),
                F.lit('Other')
            )
            .when(
                F.lower(F.col('Ethnicity')).contains('hispanic') |
                F.lower(F.col('Ethnicity')).contains('latino') |
                F.lower(F.col('Ethnicity')).contains('latina'),
                F.lit('Hispanic or Latino')
            )
            .otherwise(F.lit('Other'))
        )

    # ============================================================================
    # RACE STANDARDIZATION
    # White, Black or African American, Asian, Other
    # ============================================================================

    if 'Race' in cols:
        df = df.withColumn(
            'Race',
            F.when(F.col('Race').isNull(), F.lit(None))
            .when(
                F.lower(F.col('Race')).contains('white') &
                ~F.lower(F.col('Race')).contains('non'),
                F.lit('White')
            )
            .when(
                F.lower(F.col('Race')).contains('black') |
                F.lower(F.col('Race')).contains('african american'),
                F.lit('Black or African American')
            )
            .when(
                F.lower(F.col('Race')).contains('asian'),
                F.lit('Asian')
            )
            .otherwise(F.lit('Other'))
        )

    # ============================================================================
    # LANGUAGE STANDARDIZATION
    # English, Spanish, Other
    # ============================================================================

    if 'Preferred_Language' in cols:
        df = df.withColumn(
            'Preferred_Language',
            F.when(F.col('Preferred_Language').isNull(), F.lit(None))
            .when(
                F.lower(F.col('Preferred_Language')).contains('english') |
                (F.lower(F.trim(F.col('Preferred_Language'))) == 'en'),
                F.lit('English')
            )
            .when(
                F.lower(F.col('Preferred_Language')).contains('spanish') |
                (F.lower(F.trim(F.col('Preferred_Language'))) == 'es') |
                F.lower(F.col('Preferred_Language')).contains('español'),
                F.lit('Spanish')
            )
            .otherwise(F.lit('Other'))
        )

    # ============================================================================
    # WRITE OUTPUT
    # ============================================================================

    output_dataset.write_dataframe(df)