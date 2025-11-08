import pandas as pd
import boto3
import os
import gc
from io import StringIO
import numpy as np
import re
# Configuration
S3_BUCKET = 'dsw-sagemaker-dev-s3'
S3_PREFIX = 'OMOP_data_extractions/Melax_Vishnu/'

#### input
DRUG_CLASSES = {
    'Hypnotic_benzodiazepine': [
        'temazepam', 'estazolam', 'triazolam', 'flurazepam',
        'restoril', 'prosom', 'halcion', 'dalmane'
    ],
    'Non_benzodiazepine_Z_drug': [
        'zolpidem', 'eszopiclone', 'zaleplon',
        'ambien', 'ambien cr', 'edluar', 'intermezzo', 'zolpimist', 'lunesta', 'sonata'
    ],
    'Melatonin_receptor_agonist': [
        'ramelteon', 'rozerem'
    ],
    'Melatonin_supplement': [
        'melatonin'
    ],
    'Orexin_receptor_antagonist': [
        'suvorexant', 'lemborexant', 'daridorexant',
        'belsomra', 'dayvigo', 'quviviq'
    ],
    'Sedating_antidepressant': [
        'trazodone', 'mirtazapine',
        'desyrel', 'remeron'
    ],
    'Antihistamine': [
        'diphenhydramine', 'doxylamine',
        'benadryl', 'sominex', 'unisom sleepgels', 'unisom sleeptabs'
    ],
    'Narcolepsy_treatment': [
        'modafinil', 'armodafinil', 'pitolisant', 'methylphenidate',
        'dextroamphetamine', 'amphetamine-dextroamphetamine',
        'provigil', 'nuvigil', 'wakix', 'methylin', 'dexedrine', 'adderall'
    ],
    'Narcolepsy_cataplexy_treatment': [
        'sodium oxybate', 'gamma hydroxybutyrate', 'venlafaxine',
        'fluoxetine', 'clomipramine', 'protriptyline',
        'xyrem', 'effexor', 'prozac', 'anafranil', 'vivactil', 'lumryz'
    ],
    'Restless_legs_syndrome': [
        'ropinirole', 'pramipexole', 'gabapentin enacarbil',
        'pregabalin', 'rotigotine', 'gabapentin',
        'requip', 'mirapex', 'horizant', 'lyrica', 'neupro', 'neurontin'
    ],
    'Sleep_apnea': [
        'solriamfetol', 'modafinil', 'armodafinil',
        'sunosi', 'provigil', 'nuvigil'
    ],
    'Maintenance_insomnia': [
        'doxepin', 'silenor'
    ]
}

CONDITION_CODES = {
    'Sleep_Disorders': {
        'ICD9': [
            # No ICD-9 codes provided in source data
        ],
        'ICD10': [
            # F51 codes - Sleep disorders not due to a substance or known physiological condition
            'F51.01',   # Primary insomnia
            'F51.02',   # Adjustment insomnia
            'F51.03',   # Paradoxical insomnia
            'F51.09',   # Other insomnia not due to a substance or known physiological condition
            'F51.11',   # Primary hypersomnia
            'F51.12',   # Insufficient sleep syndrome
            'F51.19',   # Other hypersomnia not due to a substance or known physiological condition
            'F51.3',    # Sleepwalking [somnambulism]
            'F51.4',    # Sleep terrors [night terrors]
            'F51.5',    # Nightmare disorder
            'F51.8',    # Other sleep disorders not due to a substance or known physiological condition
            
            # G25 codes - Extrapyramidal and movement disorders
            'G25.81',   # Restless legs syndrome
            
            # G47 codes - Sleep disorders
            'G47',      # Sleep disorders
            'G47.0',    # Insomnia
            'G47.01',   # Insomnia due to medical condition
            'G47.09',   # Other insomnia
            'G47.1',    # Hypersomnia
            'G47.11',   # Idiopathic hypersomnia with long sleep time
            'G47.12',   # Idiopathic hypersomnia without long sleep time
            'G47.13',   # Recurrent hypersomnia
            'G47.14',   # Hypersomnia due to medical condition
            'G47.19',   # Other hypersomnia
            'G47.2',    # Circadian rhythm sleep disorders
            'G47.20',   # Circadian rhythm sleep disorder, unspecified type
            'G47.21',   # Circadian rhythm sleep disorder, delayed sleep phase type
            'G47.22',   # Circadian rhythm sleep disorder, advanced sleep phase type
            'G47.23',   # Circadian rhythm sleep disorder, irregular sleep wake type
            'G47.24',   # Circadian rhythm sleep disorder, free running type
            'G47.25',   # Circadian rhythm sleep disorder, jet lag type
            'G47.26',   # Circadian rhythm sleep disorder, shift work type
            'G47.27',   # Circadian rhythm sleep disorder in conditions classified elsewhere
            'G47.29',   # Other circadian rhythm sleep disorder
            'G47.3',    # Sleep apnea
            'G47.31',   # Primary central sleep apnea
            'G47.33',   # Obstructive sleep apnea (adult) (pediatric)
            'G47.34',   # Idiopathic sleep related nonobstructive alveolar hypoventilation
            'G47.36',   # Sleep related hypoventilation in conditions classified elsewhere
            'G47.37',   # Central sleep apnea in conditions classified elsewhere
            'G47.39',   # Other sleep apnea
            'G47.4',    # Narcolepsy and cataplexy
            'G47.41',   # Narcolepsy
            'G47.411',  # Narcolepsy with cataplexy
            'G47.419',  # Narcolepsy without cataplexy
            'G47.42',   # Narcolepsy in conditions classified elsewhere
            'G47.5',    # Parasomnia
            'G47.51',   # Confusional arousals
            'G47.52',   # REM sleep behavior disorder
            'G47.53',   # Recurrent isolated sleep paralysis
            'G47.54',   # Parasomnia in conditions classified elsewhere
            'G47.59',   # Other parasomnia
            'G47.6',    # Sleep related movement disorders
            'G47.61',   # Periodic limb movement disorder
            'G47.62',   # Sleep related leg cramps
            'G47.63',   # Sleep related bruxism
            'G47.69',   # Other sleep related movement disorders
            'G47.8',    # Other sleep disorders
            'G47.9',    # Sleep disorder, unspecified
            
            # Z72 code - Problems related to lifestyle
            'Z72.820'   # Sleep deprivation
        ]
    },
    
    'Epilepsy_and_Seizures': {
        'ICD9': [
            # 345.x - Epilepsy and recurrent seizures
            '345.00',   # Generalized nonconvulsive epilepsy without intractable epilepsy
            '345.01',   # Generalized nonconvulsive epilepsy with intractable epilepsy
            '345.10',   # Generalized convulsive epilepsy without intractable epilepsy
            '345.11',   # Generalized convulsive epilepsy with intractable epilepsy
            '345.2',    # Petit mal status
            '345.3',    # Grand mal status
            '345.40',   # Partial epilepsy with impairment of consciousness without intractable epilepsy
            '345.41',   # Partial epilepsy with impairment of consciousness with intractable epilepsy
            '345.50',   # Partial epilepsy without impairment of consciousness without intractable epilepsy
            '345.51',   # Partial epilepsy without impairment of consciousness with intractable epilepsy
            '345.60',   # Infantile spasms without intractable epilepsy
            '345.61',   # Infantile spasms with intractable epilepsy
            '345.70',   # Epilepsia partialis continua without intractable epilepsy
            '345.71',   # Epilepsia partialis continua with intractable epilepsy
            '345.80',   # Other forms of epilepsy without intractable epilepsy
            '345.81',   # Other forms of epilepsy with intractable epilepsy
            '345.90',   # Unspecified epilepsy without intractable epilepsy
            '345.91',   # Unspecified epilepsy with intractable epilepsy
            
            # 780.3x - Convulsions
            '780.31',   # Febrile convulsions (simple)
            '780.32',   # Complex febrile convulsions
            '780.33',   # Post-traumatic seizures
            '780.39',   # Other convulsions
            
            # 779.0 - Convulsions in newborn
            '779.0'     # Convulsions in newborn
        ],
        'ICD10': [
            # G40 codes - Epilepsy and recurrent seizures
            'G40.001',  # Localization-related (focal) (partial) idiopathic epilepsy and epileptic syndromes with seizures of localized onset, not intractable, with status epilepticus
            'G40.009',  # Localization-related (focal) (partial) idiopathic epilepsy and epileptic syndromes with seizures of localized onset, not intractable, without status epilepticus
            'G40.011',  # Localization-related (focal) (partial) idiopathic epilepsy and epileptic syndromes with seizures of localized onset, intractable, with status epilepticus
            'G40.019',  # Localization-related (focal) (partial) idiopathic epilepsy and epileptic syndromes with seizures of localized onset, intractable, without status epilepticus
            'G40.301',  # Generalized idiopathic epilepsy and epileptic syndromes, not intractable, with status epilepticus
            'G40.309',  # Generalized idiopathic epilepsy and epileptic syndromes, not intractable, without status epilepticus
            'G40.311',  # Generalized idiopathic epilepsy and epileptic syndromes, intractable, with status epilepticus
            'G40.319',  # Generalized idiopathic epilepsy and epileptic syndromes, intractable, without status epilepticus
            'G40.811',  # Lennox-Gastaut syndrome, not intractable, with status epilepticus
            'G40.812',  # Lennox-Gastaut syndrome, not intractable, without status epilepticus
            'G40.813',  # Lennox-Gastaut syndrome, intractable, with status epilepticus
            'G40.814',  # Lennox-Gastaut syndrome, intractable, without status epilepticus
            'G40.821',  # Epileptic spasms, not intractable, with status epilepticus
            'G40.822',  # Epileptic spasms, not intractable, without status epilepticus
            'G40.823',  # Epileptic spasms, intractable, with status epilepticus
            'G40.824',  # Epileptic spasms, intractable, without status epilepticus
            'G40.833',  # Dravet syndrome, intractable, with status epilepticus
            'G40.834',  # Dravet syndrome, intractable, without status epilepticus
            'G40.841',  # Juvenile myoclonic epilepsy, not intractable, with status epilepticus
            'G40.842',  # Juvenile myoclonic epilepsy, not intractable, without status epilepticus
            'G40.843',  # Juvenile myoclonic epilepsy, intractable, with status epilepticus
            'G40.844',  # Juvenile myoclonic epilepsy, intractable, without status epilepticus
            'G40.89',   # Other epilepsy
            'G40.901',  # Epilepsy, unspecified, not intractable, with status epilepticus
            'G40.909',  # Epilepsy, unspecified, not intractable, without status epilepticus
            'G40.911',  # Epilepsy, unspecified, intractable, with status epilepticus
            'G40.919',  # Epilepsy, unspecified, intractable, without status epilepticus
            
            # G41 codes - Status epilepticus
            'G41.0',    # Grand mal status epilepticus
            'G41.1',    # Petit mal status epilepticus
            'G41.2',    # Complex partial status epilepticus
            'G41.8',    # Other status epilepticus
            'G41.9',    # Status epilepticus, unspecified
            
            # R56 codes - Convulsions, not elsewhere classified
            'R56.00',   # Simple febrile convulsions
            'R56.01',   # Complex febrile convulsions
            'R56.1',    # Post-traumatic seizures
            'R56.9',    # Unspecified convulsions
            
            # P90 code - Convulsions of newborn
            'P90'       # Convulsions of newborn
        ]
    }
}
# Example: Filter for specific conditions (optional)

# Set to None to include all conditions:
# CONDITION_CODES = None

# Example: Filter for specific measurements (optional)
MEASUREMENT_TYPES = {
    # Vital Signs
    'temperature': {
        'keywords': [
            'temp', 'temperature', 'body temp', 'core temp', 'body temperature',
            'temperature measurement', 'patient temperature', 'fever'
        ],
        'must_have': [],
        'exclude': ['room', 'ambient', 'environmental']
    },
    
    'heart_rate': {
        'keywords': [
            'hr', 'heart rate', 'pulse', 'bpm', 'beats per minute',
            'heart rhythm', 'cardiac rate', 'pulse rate'
        ],
        'must_have': [],
        'exclude': ['respiratory', 'breathing']
    },
    
    'blood_pressure': {
        'keywords': [
            'bp', 'blood pressure', 'systolic', 'diastolic', 'sbp', 'dbp',
            'hypertension', 'hypotension', 'pressure reading'
        ],
        'must_have': [],
        'exclude': ['intracranial', 'intraocular', 'airway']
    },
    
    'respiratory_rate': {
        'keywords': [
            'rr', 'resp rate', 'respiratory rate', 'breathing rate',
            'respirations', 'breaths per minute', 'respiration count'
        ],
        'must_have': [],
        'exclude': ['heart', 'pulse', 'cardiac']
    },
    
    'oxygen_saturation': {
        'keywords': [
            'spo2', 'o2 sat', 'oxygen sat', 'pulse ox', 'sat',
            'oxygen saturation', 'pulse oximetry', 'saturation level'
        ],
        'must_have': [],
        'exclude': ['oxygen therapy', 'oxygen flow']
    },
    
    # Neurological
    'seizure_activity': {
        'keywords': [
            'seizure', 'convulsion', 'epileptic', 'ictal', 'seizure count',
            'seizure frequency', 'seizure duration', 'seizure type', 'epileptic activity'
        ],
        'must_have': [],
        'exclude': ['medication', 'therapy', 'prevention']
    },
    
    'level_of_consciousness': {
        'keywords': [
            'loc', 'consciousness', 'alert', 'responsive', 'gcs', 'glasgow',
            'glasgow coma scale', 'alertness', 'responsiveness', 'mental status'
        ],
        'must_have': [],
        'exclude': ['seizure', 'medication']
    },
    
    'neurological_assessment': {
        'keywords': [
            'neuro', 'neurological', 'reflexes', 'motor', 'sensory',
            'neurological exam', 'neuro assessment', 'motor function', 'sensory function'
        ],
        'must_have': [],
        'exclude': ['therapy', 'medication']
    },
    
    # Respiratory
    'oxygen_therapy': {
        'keywords': [
            'o2', 'oxygen', 'fio2', 'nasal cannula', 'mask', 'ventilator',
            'oxygen therapy', 'oxygen flow', 'oxygen delivery', 'supplemental oxygen'
        ],
        'must_have': [],
        'exclude': ['saturation', 'monitoring']
    },
    
    'respiratory_support': {
        'keywords': [
            'cpap', 'bipap', 'mechanical ventilation', 'intubated',
            'ventilator support', 'respiratory assistance', 'breathing support'
        ],
        'must_have': [],
        'exclude': ['weaning', 'discontinued']
    },
    
    'breath_sounds': {
        'keywords': [
            'lung sounds', 'breath sounds', 'wheeze', 'rales', 'rhonchi',
            'respiratory sounds', 'lung assessment', 'chest sounds', 'auscultation'
        ],
        'must_have': [],
        'exclude': ['therapy', 'medication']
    },
    
    # Gastrointestinal
    'feeding': {
        'keywords': [
            'feeding', 'nutrition', 'intake', 'peg', 'g-tube', 'gastrostomy',
            'feeding tube', 'nutritional intake', 'oral feeding', 'tube feeding'
        ],
        'must_have': [],
        'exclude': ['medication', 'preparation']
    },
    
    'swallowing': {
        'keywords': [
            'swallow', 'dysphagia', 'aspiration', 'swallowing function',
            'swallow assessment', 'swallow study', 'deglutition'
        ],
        'must_have': [],
        'exclude': ['therapy', 'exercise']
    },
    
    'bowel_movement': {
        'keywords': [
            'bm', 'bowel', 'stool', 'constipation', 'bowel movement',
            'defecation', 'bowel function', 'elimination', 'bowel habits'
        ],
        'must_have': [],
        'exclude': ['medication', 'therapy']
    },
    
    # Mobility
    'mobility_assessment': {
        'keywords': [
            'mobility', 'ambulation', 'walking', 'wheelchair', 'bed mobility',
            'movement assessment', 'mobility status', 'functional mobility'
        ],
        'must_have': [],
        'exclude': ['therapy', 'equipment']
    },
    
    'physical_therapy': {
        'keywords': [
            'pt', 'physical therapy', 'range of motion', 'rom',
            'physical rehabilitation', 'mobility therapy', 'movement therapy'
        ],
        'must_have': [],
        'exclude': ['assessment', 'evaluation']
    },
    
    'occupational_therapy': {
        'keywords': [
            'ot', 'occupational therapy', 'fine motor', 'occupational rehabilitation',
            'functional therapy', 'daily living skills', 'adaptive skills'
        ],
        'must_have': [],
        'exclude': ['assessment', 'evaluation']
    },
    
    # Growth & Development
    'weight': {
        'keywords': [
            'weight', 'wt', 'body weight', 'patient weight',
            'measured weight', 'weight measurement', 'body mass'
        ],
        'must_have': [],
        'exclude': ['height', 'birth', 'ideal', 'target', 'gain', 'loss']
    },
    
    'height': {
        'keywords': [
            'height', 'ht', 'length', 'stature', 'body height',
            'standing height', 'height measurement', 'patient height'
        ],
        'must_have': [],
        'exclude': ['weight', 'sitting', 'fundal']
    },
    
    'head_circumference': {
        'keywords': [
            'head circ', 'hc', 'occipital frontal', 'ofc',
            'head circumference', 'cranial circumference', 'head measurement'
        ],
        'must_have': [],
        'exclude': ['chest', 'abdominal']
    },
    
    # Sleep
    'sleep_patterns': {
        'keywords': [
            'sleep', 'rest', 'sleep study', 'apnea', 'sleep pattern',
            'sleep quality', 'sleep duration', 'sleep assessment', 'polysomnography'
        ],
        'must_have': [],
        'exclude': ['medication', 'therapy']
    },
    
    'sleep_disturbance': {
        'keywords': [
            'insomnia', 'sleep disorder', 'restless', 'sleep disturbance',
            'sleep problems', 'sleep disruption', 'restless sleep'
        ],
        'must_have': [],
        'exclude': ['medication', 'therapy']
    },
    
    # Behavioral
    'behavioral_assessment': {
        'keywords': [
            'behavior', 'agitation', 'irritability', 'mood', 'behavioral assessment',
            'behavioral observation', 'behavioral changes', 'temperament'
        ],
        'must_have': [],
        'exclude': ['therapy', 'intervention']
    },
    
    'communication': {
        'keywords': [
            'communication', 'speech', 'vocalization', 'language',
            'communication skills', 'verbal communication', 'nonverbal communication'
        ],
        'must_have': [],
        'exclude': ['therapy', 'intervention']
    },
    
    'social_interaction': {
        'keywords': [
            'social', 'interaction', 'eye contact', 'engagement',
            'social skills', 'social behavior', 'social responsiveness'
        ],
        'must_have': [],
        'exclude': ['therapy', 'intervention']
    },
    
    # Pain & Comfort
    'pain_assessment': {
        'keywords': [
            'pain', 'comfort', 'pain scale', 'discomfort', 'faces scale',
            'pain level', 'pain score', 'pain rating', 'comfort level'
        ],
        'must_have': [],
        'exclude': ['medication', 'management']
    },
    
    'sedation': {
        'keywords': [
            'sedation', 'sedated', 'calm', 'agitated', 'sedation level',
            'consciousness level', 'arousal level', 'alertness level'
        ],
        'must_have': [],
        'exclude': ['medication', 'drug']
    }
}

MAIN_DIAGNOSIS  = {
    'ICD9': ['330.8'],
    'ICD10': ['F84.2']
}
# Set to None to include all measurements:
# MEASUREMENT_TYPES = None

# Example: Demographics filter (optional)
DEMOGRAPHICS_FILTER = {
    'age_min': 0,
    'age_max': 20,
    #'gender': ['Male', 'Female'],  # or use concept IDs: [8507, 8532]
    # 'race': ['White', 'Black', 'Asian']  # optional
}
def read_s3_csv(bucket, key, chunksize=None):
    """Read CSV from S3, optionally in chunks"""
    s3 = boto3.client('s3')
    
    if chunksize:
        # For large files, read in chunks
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj['Body'], chunksize=chunksize)
    else:
        # For smaller files, read entire file
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj['Body'])

def analyze_demographics():
    """Analyze demographics/person.csv"""
    print("\n" + "="*60)
    print("DEMOGRAPHICS ANALYSIS")
    print("="*60)
    
    try:
        # Read demographics file
        df = read_s3_csv(S3_BUCKET, f'{S3_PREFIX}demographics/person.csv')
        
        # Print column names
        print("\nColumn Names:")
        print(", ".join(df.columns.tolist()))
        
        # Basic statistics
        print(f"Total patients: {len(df):,}")
        
        # Race count - using uppercase column names
        if 'RACE_CONCEPT_ID' in df.columns or 'RACE_SOURCE_VALUE' in df.columns:
            race_col = 'RACE_SOURCE_VALUE' if 'RACE_SOURCE_VALUE' in df.columns else 'RACE_CONCEPT_ID'
            print("\nRace Distribution:")
            race_counts = df[race_col].value_counts()
            for race, count in race_counts.head(10).items():
                print(f"  {race}: {count:,} ({count/len(df)*100:.2f}%)")
        
        # Gender count - using uppercase column names
        if 'GENDER_CONCEPT_ID' in df.columns or 'GENDER_SOURCE_VALUE' in df.columns:
            gender_col = 'GENDER_SOURCE_VALUE' if 'GENDER_SOURCE_VALUE' in df.columns else 'GENDER_CONCEPT_ID'
            print("\nGender Distribution:")
            gender_counts = df[gender_col].value_counts()
            for gender, count in gender_counts.items():
                print(f"  {gender}: {count:,} ({count/len(df)*100:.2f}%)")
        
        # Ethnicity count - using uppercase column names
        if 'ETHNICITY_CONCEPT_ID' in df.columns or 'ETHNICITY_SOURCE_VALUE' in df.columns:
            ethnicity_col = 'ETHNICITY_SOURCE_VALUE' if 'ETHNICITY_SOURCE_VALUE' in df.columns else 'ETHNICITY_CONCEPT_ID'
            print("\nEthnicity Distribution:")
            ethnicity_counts = df[ethnicity_col].value_counts()
            for ethnicity, count in ethnicity_counts.head(10).items():
                print(f"  {ethnicity}: {count:,} ({count/len(df)*100:.2f}%)")
        
        # Store person_ids for later use - using uppercase column name
        person_ids = set(df['PERSON_ID'].unique()) if 'PERSON_ID' in df.columns else set()
        
        # Clean up
        del df
        gc.collect()
        
        return person_ids
        
    except Exception as e:
        print(f"Error analyzing demographics: {e}")
        return set()

def analyze_icd_codes(person_ids=None, chunk_size=100000):
    """Analyze ICD codes from condition_occurrence.csv in batches"""
    print("\n" + "="*60)
    print("ICD CODES ANALYSIS")
    print("="*60)
    
    try:
        # Prepare ICD code lists for filtering
        all_icd_codes = []
        for condition, codes in CONDITION_CODES.items():
            all_icd_codes.extend(codes.get('ICD9', []))
            all_icd_codes.extend(codes.get('ICD10', []))
        
        # Initialize counters
        icd_encounter_counts = {}
        icd_patient_counts = {}
        filtered_conditions = []
        total_rows = 0
        
        print(f"\nProcessing condition_occurrence.csv in chunks of {chunk_size:,} rows...")
        
        # Process file in chunks
        chunks = read_s3_csv(S3_BUCKET, f'{S3_PREFIX}icd_codes/condition_occurrence.csv', 
                           chunksize=chunk_size)
        
        total_unique_patients = set()

        for i, chunk in enumerate(chunks):
            total_rows += len(chunk)
            print(f"  Processing chunk {i+1} ({total_rows:,} rows processed)...", end='\r')
            
            # Filter by person_ids if provided - using uppercase column name
            if person_ids and 'PERSON_ID' in chunk.columns:
                chunk = chunk[chunk['PERSON_ID'].isin(person_ids)]
            
            # Identify ICD code column - check for uppercase variations
            icd_col = None
            for col in ['CONDITION_SOURCE_VALUE', 'Condition_Source_Value', 'condition_source_value']:
                if col in chunk.columns:
                    icd_col = col
                    break
            
            if icd_col:
                # Extract ICD codes from the pipe-separated format
                # Format is: "description | icd_code"
                chunk['extracted_icd'] = chunk[icd_col].apply(
                    lambda x: x.split('|')[1].strip() if pd.notna(x) and '|' in x else x if pd.notna(x) else None
                )
                
                # Count ICD codes per encounter
                for code in chunk['extracted_icd'].value_counts().index:
                    if pd.notna(code):
                        icd_encounter_counts[code] = icd_encounter_counts.get(code, 0) + \
                                                    chunk[chunk['extracted_icd'] == code].shape[0]
                
                # Count unique patients per ICD code
                if 'PERSON_ID' in chunk.columns:
                    total_unique_patients.update(chunk['PERSON_ID'].unique())
                    for code in chunk['extracted_icd'].unique():
                        if pd.notna(code):
                            patients = chunk[chunk['extracted_icd'] == code]['PERSON_ID'].nunique()
                            if code in icd_patient_counts:
                                icd_patient_counts[code] = max(icd_patient_counts[code], patients)
                            else:
                                icd_patient_counts[code] = patients
                
                # Filter for specific conditions
                for code in all_icd_codes:
                    # Check for exact match or prefix match
                    mask = (chunk['extracted_icd'] == code) | (chunk['extracted_icd'].str.startswith(code, na=False))
                    filtered = chunk[mask]
                    if not filtered.empty:
                        filtered_conditions.append(filtered)
        
        print(f"\n\nTotal rows processed: {total_rows:,}")
        
        # Print top 15 ICD codes by encounter count
        print("\nTop 15 ICD Codes by Encounter Count:")
        sorted_encounters = sorted(icd_encounter_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        for code, count in sorted_encounters:
            print(f"  {code}: {count:,} encounters")
        
        print("\nTop 15 ICD Codes by Patient Count:")
        print(f"Total unique patients: {len(total_unique_patients):,}")
        sorted_patients = sorted(icd_patient_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        for code, count in sorted_patients:
            if len(total_unique_patients) > 0:
                percentage = (count / len(total_unique_patients)) * 100
            else:
                percentage = 0
            print(f"  {code}: {count:,} patients ({percentage:.2f}%)")
        
        # Modify the filtered conditions summary:
        if filtered_conditions:
            filtered_df = pd.concat(filtered_conditions, ignore_index=True)
            print(f"\nFiltered Conditions Summary:")
            print(f"  Total filtered records: {len(filtered_df):,}")
            if 'PERSON_ID' in filtered_df.columns:
                unique_with_condition = filtered_df['PERSON_ID'].nunique()
                if len(total_unique_patients) > 0:
                    percentage = (unique_with_condition / len(total_unique_patients)) * 100
                else:
                    percentage = 0
                print(f"  Unique patients with target conditions: {unique_with_condition:,} ({percentage:.2f}%)")
            del filtered_df
        
        # Clean up
        gc.collect()
        
    except Exception as e:
        print(f"\nError analyzing ICD codes: {e}")


def analyze_measurements(person_ids=None, chunk_size=100000):
    """Analyze measurements from measurement.csv in batches"""
    print("\n" + "="*60)
    print("MEASUREMENTS ANALYSIS")
    print("="*60)
    
    try:
        # Initialize counters
        measurement_counts = {}
        patient_measurement_presence = {}
        total_rows = 0
        unique_patients = set()
        
        print(f"\nProcessing measurement.csv in chunks of {chunk_size:,} rows...")
        
        # Process file in chunks
        chunks = read_s3_csv(S3_BUCKET, f'{S3_PREFIX}measurements/measurement.csv', 
                           chunksize=chunk_size)
        
        for i, chunk in enumerate(chunks):
            total_rows += len(chunk)
            print(f"  Processing chunk {i+1} ({total_rows:,} rows processed)...", end='\r')
            
            # Reset index to ensure proper alignment
            chunk = chunk.reset_index(drop=True)
            
            # Filter by person_ids if provided
            if person_ids and 'PERSON_ID' in chunk.columns:
                chunk = chunk[chunk['PERSON_ID'].isin(person_ids)]
                # Reset index again after filtering
                chunk = chunk.reset_index(drop=True)
            
            # Track unique patients
            if 'PERSON_ID' in chunk.columns:
                unique_patients.update(chunk['PERSON_ID'].unique())
            
            # Use MEASUREMENT_SOURCE_VALUE column specifically
            if 'MEASUREMENT_SOURCE_VALUE' not in chunk.columns:
                print(f"\nWarning: MEASUREMENT_SOURCE_VALUE column not found. Available columns: {chunk.columns.tolist()}")
                continue
            
            # Count measurements
            for measure in chunk['MEASUREMENT_SOURCE_VALUE'].value_counts().index:
                if pd.notna(measure):
                    measurement_counts[measure] = measurement_counts.get(measure, 0) + \
                                                 chunk[chunk['MEASUREMENT_SOURCE_VALUE'] == measure].shape[0]
            
            # Track which patients have which measurements using regex
            if 'PERSON_ID' in chunk.columns:
                for measure_type, config in MEASUREMENT_TYPES.items():
                    # Initialize mask with proper index alignment
                    keyword_mask = pd.Series([False] * len(chunk), index=chunk.index)
                    
                    # Check for keyword matches using regex for flexible matching
                    for keyword in config['keywords']:
                        # Create case-insensitive regex pattern with word boundaries
                        pattern = r'(?i)\b' + re.escape(keyword) + r'\b'
                        temp_mask = chunk['MEASUREMENT_SOURCE_VALUE'].str.contains(
                            pattern, regex=True, na=False, case=False
                        )
                        # Ensure alignment
                        keyword_mask = keyword_mask | temp_mask.reindex(keyword_mask.index, fill_value=False)
                    
                    # Apply must_have filters with regex
                    if config['must_have']:
                        must_have_mask = pd.Series([False] * len(chunk), index=chunk.index)
                        for must in config['must_have']:
                            pattern = r'(?i)\b' + re.escape(must) + r'\b'
                            temp_mask = chunk['MEASUREMENT_SOURCE_VALUE'].str.contains(
                                pattern, regex=True, na=False, case=False
                            )
                            must_have_mask = must_have_mask | temp_mask.reindex(must_have_mask.index, fill_value=False)
                        keyword_mask = keyword_mask & must_have_mask
                    
                    # Apply exclusions with regex
                    for exclude in config['exclude']:
                        pattern = r'(?i)\b' + re.escape(exclude) + r'\b'
                        exclude_mask = chunk['MEASUREMENT_SOURCE_VALUE'].str.contains(
                            pattern, regex=True, na=False, case=False
                        )
                        keyword_mask = keyword_mask & ~exclude_mask.reindex(keyword_mask.index, fill_value=False)
                    
                    # Track patients with this measurement
                    if keyword_mask.any():
                        # Use loc to ensure proper indexing
                        patients_with_measure = chunk.loc[keyword_mask, 'PERSON_ID'].unique()
                        if measure_type not in patient_measurement_presence:
                            patient_measurement_presence[measure_type] = set()
                        patient_measurement_presence[measure_type].update(patients_with_measure)
        
        print(f"\n\nTotal rows processed: {total_rows:,}")
        print(f"Total unique patients: {len(unique_patients):,}")
        
        # Print top 15 measurements
        print("\nTop 15 Measurements by Count:")
        sorted_measurements = sorted(measurement_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        for measure, count in sorted_measurements:
            # Truncate long measurement names for display
            display_name = measure[:60] + "..." if len(str(measure)) > 60 else measure
            print(f"  {display_name}: {count:,}")
        
        # Calculate missing measurement percentages
        print("\nMeasurement Coverage (% of patients with measurement):")
        for measure_type in sorted(patient_measurement_presence.keys()):
            patients_with = patient_measurement_presence[measure_type]
            if unique_patients:
                percentage = (len(patients_with) / len(unique_patients)) * 100
                missing_percentage = 100 - percentage
                print(f"  {measure_type}:")
                print(f"    Patients with measurement: {len(patients_with):,} ({percentage:.2f}%)")
                print(f"    Patients missing measurement: {len(unique_patients) - len(patients_with):,} ({missing_percentage:.2f}%)")
        
        # Clean up
        gc.collect()
        
    except Exception as e:
        print(f"\nError analyzing measurements: {e}")
        import traceback
        traceback.print_exc()
        
def analyze_medications(person_ids=None, chunk_size=100000):
    """Analyze medications (assuming there's a medications folder with relevant CSV)"""
    print("\n" + "="*60)
    print("MEDICATIONS ANALYSIS")
    print("="*60)
    
    try:
        # Check if medications folder exists and get the CSV file
        s3 = boto3.client('s3')
        
        # Try to find medications file
        medication_file = None
        try:
            response = s3.list_objects_v2(
                Bucket=S3_BUCKET,
                Prefix=f'{S3_PREFIX}medications/'
            )
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('.csv'):
                        medication_file = obj['Key']
                        break
        except:
            print("No medications folder found. Skipping medication analysis.")
            return
        
        if not medication_file:
            print("No medication CSV file found. Skipping medication analysis.")
            return
        
        # Initialize counters
        med_encounter_counts = {}
        med_patient_counts = {}
        drug_class_counts = {}
        drug_class_patients = {drug_class: set() for drug_class in DRUG_CLASSES.keys()}
        total_rows = 0
        
        print(f"\nProcessing medications in chunks of {chunk_size:,} rows...")
        
        # Process file in chunks
        chunks = read_s3_csv(S3_BUCKET, medication_file, chunksize=chunk_size)
        
        total_unique_patients = set()

        for i, chunk in enumerate(chunks):
            total_rows += len(chunk)
            print(f"  Processing chunk {i+1} ({total_rows:,} rows processed)...", end='\r')
            
            # Filter by person_ids if provided
            if person_ids and 'PERSON_ID' in chunk.columns:
                chunk = chunk[chunk['PERSON_ID'].isin(person_ids)]
            
            # Use DRUG_SOURCE_VALUE column specifically
            if 'DRUG_SOURCE_VALUE' not in chunk.columns:
                print(f"\nWarning: DRUG_SOURCE_VALUE column not found. Available columns: {chunk.columns.tolist()}")
                continue
            
            # Count medications per encounter
            for med in chunk['DRUG_SOURCE_VALUE'].value_counts().index:
                if pd.notna(med):
                    med_encounter_counts[med] = med_encounter_counts.get(med, 0) + \
                                               chunk[chunk['DRUG_SOURCE_VALUE'] == med].shape[0]
            
            # Count unique patients per medication
            if 'PERSON_ID' in chunk.columns:
                total_unique_patients.update(chunk['PERSON_ID'].unique())
                for med in chunk['DRUG_SOURCE_VALUE'].unique():
                    if pd.notna(med):
                        patients = chunk[chunk['DRUG_SOURCE_VALUE'] == med]['PERSON_ID'].nunique()
                        if med in med_patient_counts:
                            med_patient_counts[med] = max(med_patient_counts[med], patients)
                        else:
                            med_patient_counts[med] = patients
            
            # Classify drugs using regex for flexible matching
            for drug_class, drugs in DRUG_CLASSES.items():
                # Reset index to ensure alignment
                chunk_reset = chunk.reset_index(drop=True)
                class_mask = pd.Series([False] * len(chunk_reset), index=chunk_reset.index)
                
                for drug in drugs:
                    # Create case-insensitive regex pattern with word boundaries
                    # This allows for "metformin" to match "Metformin HCL 500mg" etc.
                    pattern = r'(?i)\b' + re.escape(drug) + r'\b'
                    drug_mask = chunk_reset['DRUG_SOURCE_VALUE'].str.contains(
                        pattern, regex=True, na=False, case=False
                    )
                    # Ensure the mask has the same index as class_mask
                    class_mask = class_mask | drug_mask
                
                # Count prescriptions for this drug class using loc for proper indexing
                class_count = chunk_reset.loc[class_mask].shape[0]
                drug_class_counts[drug_class] = drug_class_counts.get(drug_class, 0) + class_count
                
                # Track unique patients for this drug class
                if 'PERSON_ID' in chunk_reset.columns and class_mask.any():
                    patients_in_class = chunk_reset.loc[class_mask, 'PERSON_ID'].unique()
                    drug_class_patients[drug_class].update(patients_in_class)
        
        print(f"\n\nTotal rows processed: {total_rows:,}")
        
        # Print top 15 medications by encounter count
        print("\nTop 15 Medications by Encounter Count:")
        sorted_encounters = sorted(med_encounter_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        for med, count in sorted_encounters:
            # Truncate long medication names for display
            display_name = med[:60] + "..." if len(str(med)) > 60 else med
            print(f"  {display_name}: {count:,} encounters")
        
        # Print top 15 medications by patient count
        print("\nTop 15 Medications by Patient Count:")
        sorted_patients = sorted(med_patient_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        for med, count in sorted_patients:
            display_name = med[:60] + "..." if len(str(med)) > 60 else med
            print(f"  {med}: {count:,} patients")
        
        # Print drug class summary with both prescription counts and patient counts
        print("\nDrug Class Summary:")
        print(f"Total unique patients with any medication: {len(total_unique_patients):,}")
        for drug_class in sorted(drug_class_counts.keys(), key=lambda x: drug_class_counts[x], reverse=True):
            prescription_count = drug_class_counts[drug_class]
            patient_count = len(drug_class_patients[drug_class])
            # Calculate percentage
            if len(total_unique_patients) > 0:
                percentage = (patient_count / len(total_unique_patients)) * 100
            else:
                percentage = 0
            print(f"  {drug_class}:")
            print(f"    Prescriptions: {prescription_count:,}")
            print(f"    Unique patients: {patient_count:,} ({percentage:.2f}%)")
        # Clean up
        gc.collect()
        
    except Exception as e:
        print(f"\nError analyzing medications: {e}")


def calculate_age_at_diagnosis(demographics_df, chunk_size=100000):
    """Calculate age at diagnosis based on first occurrence of target ICD codes"""
    print("\n" + "="*60)
    print("CALCULATING AGE AT DIAGNOSIS")
    print("="*60)
    
    try:
        # Prepare target ICD codes from MAIN_DIAGNOSIS
        target_icd_codes = []
        target_icd_codes.extend(MAIN_DIAGNOSIS.get('ICD9', []))
        target_icd_codes.extend(MAIN_DIAGNOSIS.get('ICD10', []))
        
        # Dictionary to store first diagnosis date per patient
        first_diagnosis_dates = {}
        
        print(f"\nProcessing condition_occurrence.csv for diagnosis dates...")
        
        # Process condition file in chunks
        chunks = read_s3_csv(S3_BUCKET, f'{S3_PREFIX}icd_codes/condition_occurrence.csv', 
                           chunksize=chunk_size)
        
        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}...", end='\r')
            
            # Identify ICD code column
            icd_col = None
            for col in ['CONDITION_SOURCE_VALUE', 'Condition_Source_Value', 'condition_source_value']:
                if col in chunk.columns:
                    icd_col = col
                    break
            
            if not icd_col:
                continue
            
            # Extract ICD codes
            chunk['extracted_icd'] = chunk[icd_col].apply(
                lambda x: x.split('|')[1].strip() if pd.notna(x) and '|' in x else x if pd.notna(x) else None
            )
            
            # Find date column
            date_col = None
            for col in ['CONDITION_START_DATE', 'Condition_Start_Date', 'condition_start_date']:
                if col in chunk.columns:
                    date_col = col
                    break
            
            if not date_col:
                print("\nWarning: No date column found in condition_occurrence.csv")
                continue
            
            # Convert date column to datetime
            chunk[date_col] = pd.to_datetime(chunk[date_col], errors='coerce')
            
            # Filter for target ICD codes
            for code in target_icd_codes:
                mask = (chunk['extracted_icd'] == code) | (chunk['extracted_icd'].str.startswith(code, na=False))
                filtered = chunk[mask]
                
                if not filtered.empty and 'PERSON_ID' in filtered.columns:
                    # Group by person and get minimum date
                    person_dates = filtered.groupby('PERSON_ID')[date_col].min()
                    
                    # Update first_diagnosis_dates with earlier dates
                    for person_id, date in person_dates.items():
                        if pd.notna(date):
                            if person_id not in first_diagnosis_dates or date < first_diagnosis_dates[person_id]:
                                first_diagnosis_dates[person_id] = date
        
        print(f"\n\nFound diagnosis dates for {len(first_diagnosis_dates):,} patients")
        
        # Calculate age at diagnosis
        # Ensure PERSON_ID is in demographics_df
        if 'PERSON_ID' not in demographics_df.columns:
            print("Error: PERSON_ID not found in demographics dataframe")
            return demographics_df
        
        # Find birth date column
        birth_col = None
        for col in ['BIRTH_DATETIME', 'Birth_Datetime', 'birth_datetime', 'BIRTH_DATE']:
            if col in demographics_df.columns:
                birth_col = col
                break
        
        if not birth_col:
            print("Warning: No birth date column found in demographics")
            return demographics_df
        
        # Convert birth dates to datetime
        demographics_df[birth_col] = pd.to_datetime(demographics_df[birth_col], errors='coerce')
        
        # Calculate age at diagnosis
        demographics_df['age_at_diagnosis'] = demographics_df.apply(
            lambda row: (first_diagnosis_dates[row['PERSON_ID']] - row[birth_col]).days / 365.25 
            if row['PERSON_ID'] in first_diagnosis_dates and pd.notna(row[birth_col]) 
            else None, axis=1
        )
        
        # Print summary statistics
        print("\nAge at Diagnosis Statistics:")
        if 'age_at_diagnosis' in demographics_df.columns:
            valid_ages = demographics_df['age_at_diagnosis'].dropna()
            if not valid_ages.empty:
                print(f"  Patients with diagnosis: {len(valid_ages):,}")
                print(f"  Mean age: {valid_ages.mean():.2f} years")
                print(f"  Median age: {valid_ages.median():.2f} years")
                print(f"  Min age: {valid_ages.min():.2f} years")
                print(f"  Max age: {valid_ages.max():.2f} years")
        
        return demographics_df
        
    except Exception as e:
        print(f"Error calculating age at diagnosis: {e}")
        return demographics_df

def apply_demographics_filter(demographics_df, filter_config=None):
    """Apply demographics filters to the dataframe"""
    print("\n" + "="*60)
    print("APPLYING DEMOGRAPHICS FILTERS")
    print("="*60)
    
    if filter_config is None:
        filter_config = DEMOGRAPHICS_FILTER
    
    if not filter_config:
        print("No filters to apply")
        return demographics_df
    
    original_count = len(demographics_df)
    filtered_df = demographics_df.copy()
    
    # Filter by age at diagnosis
    if 'age_min' in filter_config and 'age_max' in filter_config:
        if 'age_at_diagnosis' in filtered_df.columns:
            age_min = filter_config['age_min']
            age_max = filter_config['age_max']
            before_filter = len(filtered_df)
            filtered_df = filtered_df[
                (filtered_df['age_at_diagnosis'] >= age_min) & 
                (filtered_df['age_at_diagnosis'] <= age_max)
            ]
            after_filter = len(filtered_df)
            print(f"Age at diagnosis filter ({age_min}-{age_max} years):")
            print(f"  Before: {before_filter:,} patients")
            print(f"  After: {after_filter:,} patients")
            print(f"  Removed: {before_filter - after_filter:,} patients")
        else:
            print("Warning: age_at_diagnosis column not found, skipping age filter")
    
    # Filter by gender if specified
    if 'gender' in filter_config:
        gender_col = None
        for col in ['GENDER_SOURCE_VALUE', 'GENDER_CONCEPT_ID']:
            if col in filtered_df.columns:
                gender_col = col
                break
        
        if gender_col:
            before_filter = len(filtered_df)
            filtered_df = filtered_df[filtered_df[gender_col].isin(filter_config['gender'])]
            after_filter = len(filtered_df)
            print(f"Gender filter ({filter_config['gender']}):")
            print(f"  Before: {before_filter:,} patients")
            print(f"  After: {after_filter:,} patients")
            print(f"  Removed: {before_filter - after_filter:,} patients")
    
    # Filter by race if specified
    if 'race' in filter_config:
        race_col = None
        for col in ['RACE_SOURCE_VALUE', 'RACE_CONCEPT_ID']:
            if col in filtered_df.columns:
                race_col = col
                break
        
        if race_col:
            before_filter = len(filtered_df)
            filtered_df = filtered_df[filtered_df[race_col].isin(filter_config['race'])]
            after_filter = len(filtered_df)
            print(f"Race filter ({filter_config['race']}):")
            print(f"  Before: {before_filter:,} patients")
            print(f"  After: {after_filter:,} patients")
            print(f"  Removed: {before_filter - after_filter:,} patients")
    
    # Summary
    print(f"\nFilter Summary:")
    print(f"  Original patients: {original_count:,}")
    print(f"  Filtered patients: {len(filtered_df):,}")
    print(f"  Total removed: {original_count - len(filtered_df):,}")
    print(f"  Retention rate: {(len(filtered_df)/original_count)*100:.2f}%")
    
    return filtered_df

def main():
    """Main execution function with preprocessing"""
    print("Starting OMOP Data Analysis Pipeline")
    print("="*60)
    
    # Step 1: Load demographics
    print("\n" + "="*60)
    print("LOADING DEMOGRAPHICS")
    print("="*60)
    demographics_df = read_s3_csv(S3_BUCKET, f'{S3_PREFIX}demographics/person.csv')
    print(f"Loaded {len(demographics_df):,} patients")
    
    # Step 2: Calculate age at diagnosis (preprocessing)
    demographics_df = calculate_age_at_diagnosis(demographics_df)
    
    # Step 3: Apply demographics filters (preprocessing)
    demographics_df = apply_demographics_filter(demographics_df, DEMOGRAPHICS_FILTER)
    
    # Get filtered person_ids for subsequent analyses
    person_ids = set(demographics_df['PERSON_ID'].unique()) if 'PERSON_ID' in demographics_df.columns else set()
    
    # Step 4: Analyze demographics (with filtered data)
    print("\n" + "="*60)
    print("DEMOGRAPHICS ANALYSIS (FILTERED)")
    print("="*60)
    
    print(f"Total patients after filtering: {len(demographics_df):,}")
    
    # Print column names
    print("\nColumn Names:")
    print(", ".join(demographics_df.columns.tolist()))
    
    # Race distribution
    if 'RACE_CONCEPT_ID' in demographics_df.columns or 'RACE_SOURCE_VALUE' in demographics_df.columns:
        race_col = 'RACE_SOURCE_VALUE' if 'RACE_SOURCE_VALUE' in demographics_df.columns else 'RACE_CONCEPT_ID'
        print("\nRace Distribution:")
        race_counts = demographics_df[race_col].value_counts()
        for race, count in race_counts.head(10).items():
            print(f"  {race}: {count:,} ({count/len(demographics_df)*100:.2f}%)")
    
    # Gender distribution
    if 'GENDER_CONCEPT_ID' in demographics_df.columns or 'GENDER_SOURCE_VALUE' in demographics_df.columns:
        gender_col = 'GENDER_SOURCE_VALUE' if 'GENDER_SOURCE_VALUE' in demographics_df.columns else 'GENDER_CONCEPT_ID'
        print("\nGender Distribution:")
        gender_counts = demographics_df[gender_col].value_counts()
        for gender, count in gender_counts.items():
            print(f"  {gender}: {count:,} ({count/len(demographics_df)*100:.2f}%)")
    
    # Ethnicity distribution
    if 'ETHNICITY_CONCEPT_ID' in demographics_df.columns or 'ETHNICITY_SOURCE_VALUE' in demographics_df.columns:
        ethnicity_col = 'ETHNICITY_SOURCE_VALUE' if 'ETHNICITY_SOURCE_VALUE' in demographics_df.columns else 'ETHNICITY_CONCEPT_ID'
        print("\nEthnicity Distribution:")
        ethnicity_counts = demographics_df[ethnicity_col].value_counts()
        for ethnicity, count in ethnicity_counts.head(10).items():
            print(f"  {ethnicity}: {count:,} ({count/len(demographics_df)*100:.2f}%)")
    
    # Age at diagnosis distribution (if calculated)
    if 'age_at_diagnosis' in demographics_df.columns:
        print("\nAge at Diagnosis Distribution:")
        age_stats = demographics_df['age_at_diagnosis'].describe()
        print(f"  Mean: {age_stats['mean']:.2f} years")
        print(f"  Std: {age_stats['std']:.2f} years")
        print(f"  Min: {age_stats['min']:.2f} years")
        print(f"  25%: {age_stats['25%']:.2f} years")
        print(f"  50% (Median): {age_stats['50%']:.2f} years")
        print(f"  75%: {age_stats['75%']:.2f} years")
        print(f"  Max: {age_stats['max']:.2f} years")
    
    # Clean up demographics dataframe to save memory
    del demographics_df
    gc.collect()
    
    # Step 5: Analyze ICD codes with filtered person_ids
    analyze_icd_codes(person_ids)
    
    # Step 6: Analyze measurements with filtered person_ids
    analyze_measurements(person_ids)
    
    # Step 7: Analyze medications with filtered person_ids
    analyze_medications(person_ids)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
    