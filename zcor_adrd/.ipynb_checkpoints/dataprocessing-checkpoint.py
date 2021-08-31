import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def ternary_encoding(input_path, 
              INFERENCE_WINDOW = 104,
              separator = '|',
              delimiter = ':',
              min_inference_weeks = 1,
              phenotype_folder = '../data/phenotypes',
              verbose = False):
    """Encode the given inference windows of patients' medical histories into ternary encodings with respect to each of the phenotypes provided.
    
    Args:
      input_path ([type]): Path to dataframe with patients' records.
      INFERENCE_WINDOW (int, optional): Length of the inference window used for encoding. Defaults to 104.
      separator (str, optional): Character that separates individual timed records. Defaults to '|'.
      delimiter (str, optional): Character that delimits data within individual timed records. Defaults to ':'.
      min_inference_weeks (int, optional): Minimum number of weeks within specified inference window that must contain codes. Defaults to 1.
      phenotype_folder (str, optional): Folder containing phenotypes used for encoding. Defaults to 'data/phenotypes'.
      verbose (bool, optional): Verbosity switch. Defaults to False.
      
    Returns:
      NA
    """
    # Get the input dataframe
    input_df = pd.read_csv(input_path)
    
    # Get the phenotypes
    DX = [phn.split('.')[0] for phn in os.listdir(phenotype_folder) if phn.split("_")[0] == 'DX']
    phenotype_catalog = {
            'DX': DX,
            'TOTAL': DX}

    phenotype_codelist = {}
    for PHN in phenotype_catalog['DX']:
        with open("%s/%s.phn" % (phenotype_folder, PHN), "r") as f:
            raw = f.readlines()[0]
            phenotype_codelist[PHN] = set(raw[:-1].split())
    
    
    data_columns = [
            'patient_id', 
            'gender', 
            'age_at_screening',
            'sequence',
            'sequence_weeks',
            'sequence_codes',
            'abs_sequence_weeks',
            'prediction_point',
            'first_week', 
            'last_week',
            'sequence_length',
        ] + \
        [i for i in phenotype_codelist.keys()] + \
        ['%s_weeks' % i for i in phenotype_codelist.keys()] + \
        ['%s_codes' % i for i in phenotype_codelist.keys()]
    DATA = {i: [] for i in data_columns}
    empty_inference_windows = 0
    insufficient_inference_weeks = 0
    input_size = input_df.shape[0]
    # ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~
    #  ~~~ ITERATE OVER PATIENTS ~~~~
    # ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ ~~~~
    for i, row in input_df.reset_index(drop=True).iterrows():

        body = row['record'].split(separator)[1:]
        MIN_WEEK = row.prediction_point - INFERENCE_WINDOW
        MAX_WEEK = row.prediction_point
        
        ALL_CODES = []
        ALL_WEEKS = []
        ALL_ABS_WEEKS = []
        
        for record in body:
            content = record.split(delimiter)
            w = int(content[0])
            c = content[1].strip()[:7]
            if MIN_WEEK <= w <= MAX_WEEK:
                ALL_WEEKS.append(w - MIN_WEEK)
                ALL_ABS_WEEKS.append(w)
                ALL_CODES.append(c)
                
        # Filter the inadequate patients out
        if not len(ALL_WEEKS):
            empty_inference_windows += 1
            continue
        if len(set(ALL_WEEKS)) < min_inference_weeks:
            insufficient_inference_weeks += 1
            continue
        # -------------------
        # Populate the mould encoding (binary, '0's and '2's) 
        # ---------------------
        mould = ["0"] * INFERENCE_WINDOW
        for week in ALL_WEEKS:
            mould[(week-1)] = "2"

        # ~~~ ~~~~~~~~ ~~~~~~~~~
        # Populate the phenotype encodings (ternary, '0's, '1's, and '2's)
        # ~~~~~~~~ ~~~~~~~~~ ~~~
        phenotype_weeks = {}
        phenotype_codes = {}
        
        for phn_name, phn_codes in phenotype_codelist.items():
            # trace the weeks when the phenotype code has appeared
            phenotype_weeks[phn_name] = [ALL_WEEKS[i] for i in range(len(ALL_WEEKS)) if ALL_CODES[i] in phn_codes]
            phenotype_codes[phn_name] = [ALL_CODES[i] for i in range(len(ALL_WEEKS)) if ALL_CODES[i] in phn_codes]
            
            # if at least one phenotype code is found in the inference window, populate the mould
            if len(phenotype_weeks[phn_name]):
                phn_mould = mould.copy()
                for week in phenotype_weeks[phn_name]:
                    phn_mould[(week-1)] = "1"
                DATA[phn_name].append(' '.join(phn_mould))
            else:
                DATA[phn_name].append('None')
        
        DATA['patient_id'].append(row.patient_id)
        DATA['gender'].append(row.gender)
        DATA['age_at_screening'].append(int(row.age) + (row.prediction_point/52))
        
        DATA['sequence'].append(' '.join(mould))
        DATA['sequence_weeks'].append(' '.join([str(i) for i in ALL_WEEKS]))
        DATA['sequence_codes'].append(' '.join(ALL_CODES))
        DATA['abs_sequence_weeks'].append(' '.join([str(i) for i in ALL_ABS_WEEKS]))

        for phn_name in phenotype_codes.keys():
            DATA['%s_weeks' % phn_name].append(' '.join([str(i) for i in phenotype_weeks[phn_name]]) if len(phenotype_weeks[phn_name]) else 'None')
            DATA['%s_codes' % phn_name].append(' '.join([str(i) for i in phenotype_codes[phn_name]]) if len(phenotype_codes[phn_name]) else 'None')
            
        DATA['sequence_length'].append(INFERENCE_WINDOW)
        DATA['prediction_point'].append(row.prediction_point)
        DATA['first_week'].append(np.min(ALL_ABS_WEEKS))
        DATA['last_week'].append(np.max(ALL_ABS_WEEKS))
        
    if verbose:
        print('~ ~ ~ ~ ~')
        print("%d patients excluded due to empty inference window" % empty_inference_windows)
        print("%d more patients excluded due to insufficient number of non-null weeks" % insufficient_inference_weeks)
        print('~ ~ ~ ~ ~')
    DF = pd.DataFrame(DATA).drop_duplicates()
    DF = DF.merge(input_df[['patient_id', 'target']], on = 'patient_id')
    
    # Split the dataframe into PFSA set (for PFSA generation) and LLK set (for LLK-based features).
    X = DF.drop(['target'], 1)
    y = DF.target

    X_pfsa, X_llk, y_pfsa, y_llk = train_test_split(X, y, test_size=0.5)
    PFSA_DATA = X_pfsa.copy()
    PFSA_DATA['target'] = y_pfsa
    LLK_DATA = X_llk.copy()
    LLK_DATA['target'] = y_llk

    PFSA_DATA.to_csv("../data/PFSA_SET.csv", index = False)
    LLK_DATA.to_csv("../data/LLK_SET.csv", index = False)
