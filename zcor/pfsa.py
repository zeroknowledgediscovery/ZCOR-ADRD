import os
import time
import tempfile
import shutil

import pandas as pd
import numpy as np

from functools import reduce

from zcor.z3 import *

def infer_pfsa(phenotype_catalog,
                 PFSA_SET,
                 PFSA_PATH,
                 STOPLIST,
                 POS_EPSILON = 0.2,
                 NEG_EPSILON = 0.2,
                 GENESESS_PATH = 'bin/genESeSS',           
                 verbose = False): 
    """Infer PFSA models for every phenotype provided with given data
    
    Args:
      phenotype_catalog ([type]): Collection of phenotype names provided.
      PFSA_SET ([type]): Dataframe with ternary encodings for given patients.
      PFSA_PATH (str): Filepath to save the inferred PFSA models to.
      STOPLIST ([type]): List of phenotypes to exclude for inference.
      POS_EPSILON (float, optional): Epsilon value for the inference of the positive PFSA models. Defaults to 0.2.
      NEG_EPSILON (float, optional): Epsilon value for the inference of the negative PFSA models. Defaults to 0.2.
      GENESESS_PATH (str, optional): Path to genESeSS binary used for PFSA inference. Defaults to 'bin/genESeSS'.
      verbose (bool, optional): Verbosity switch. Defaults to False.
    """
    # Exclude the phenotypes found in STOPLIST
    phenotype_catalog = {i: [j for j in content if j not in STOPLIST] for i, content in phenotype_catalog.items()}
    # Generate PFSA for every phenotype listed
    for i, PHN in enumerate(phenotype_catalog['TOTAL']):
        if verbose:
            print('%d/%d > %s' % (i+1, len(phenotype_catalog['TOTAL']), PHN))
        PFSA_DF = PFSA_SET[(~PFSA_SET[PHN].isnull())]        
        PFSA_DF = PFSA_DF[PFSA_DF[PHN].str.contains('1')]
        Z=Z3Classifier(result_path = PFSA_PATH % (PHN), 
                       genesess_path = GENESESS_PATH)
        Z.fit(PFSA_DF[[PHN, 'target']], 
              peps = POS_EPSILON, 
              neps = NEG_EPSILON)

def generate_pfsa_features(phenotype_catalog,
                 DF, 
                 POS_PFSA_PATH,
                 NEG_PFSA_PATH,
                 STOPLIST,
                 used_channels = ['DX'],
                 ID_COLUMN = 'patient_id',
                 FEATURE_COLUMNS = [],
                 STAT_COLUMNS = [],
                 LLK_PATH = 'bin/llk',
                 GENESESS_PATH = 'bin/genESeSS',
                 verbose = False):
    
    """Generate loglikelihoods for every generated PFSA model and derive features to use for the classifier.
    
    Args:
      phenotype_catalog ([type]): Collection of phenotype names provided.
      DF ([type]): Dataframe with ternary encodings for given patients.
      POS_PFSA_PATH (str): Filepath to the positive inferred PFSA files.
      NEG_PFSA_PATH (str): Filepath to the negative inferred PFSA files.
      STOPLIST ([type]): List of phenotypes to exclude for inference.
      used_channels ([type], optional): Data channels used for the PFSA (DX for diagnoses, RX for prescriptions etc.). Defaults to ['DX'].
      FEATURE_COLUMNS ([type], optional): Names of columns from the input dataframe to return in the output dataframe as features. Defaults to [].
      STAT_COLUMNS ([type], optional): Names of columns from the input dataframe to return in the output dataframe for genral reference. Defaults to [].
      LLK_PATH (str, optional): Path to llk binary used for loglikelihoods computation. Defaults to 'bin/llk'.
      GENESESS_PATH (str, optional): Path to genESeSS binary used for PFSA inference. Defaults to 'bin/genESeSS'.
      verbose (bool, optional): Verbosity switch. Defaults to False.
    """
    feature_dfs = []
    features = []
    
    JOIN_COLUMNS = [ID_COLUMN] + FEATURE_COLUMNS + STAT_COLUMNS
    
    # Exclude the phenotypes found in STOPLIST
    phenotype_catalog = {i: [j for j in content if j not in STOPLIST] for i, content in phenotype_catalog.items()}
    
    # Compute the loglikelihoods and derivatives for every phenotype encoding
    start = time.time()
    for i, PHN in enumerate(phenotype_catalog['TOTAL']):
        
        df_columns = ['patient_id',
               '%s_sld' % PHN, '%s_ratio' % PHN,
               '%s_abs_neg' % PHN, '%s_abs_pos' % PHN]
        
        if verbose:
            print('%d/%d > %s' % (i+1, len(phenotype_catalog['TOTAL']), PHN))
        # Filter out NaN sequences
        PHN_DF = DF[(~DF[PHN].isnull())]
        # Filter out sequences without phenotype codes
        PHN_DF = PHN_DF[PHN_DF[PHN].str.contains('1')]
    
        ## Check if we are deailing with a nonempty dataframe
        if not PHN_DF.shape[0]:
            df = pd.DataFrame({i: [] for i in (df_columns + FEATURE_COLUMNS)})
            feature_dfs.append(df)
            now = time.time()
            if verbose:
                print("%.1f minutes elapsed" % ((now - start)/60))
            features.append(PHN)
            continue
        
        preds = PHN_DF[JOIN_COLUMNS]
        z3_temp_path = tempfile.mkdtemp()
        Z = Z3Classifier(use_own_pfsa = True, 
                         posmod = POS_PFSA_PATH % (PHN),
                         negmod = NEG_PFSA_PATH % (PHN),
                         result_path = z3_temp_path,
                         llk_path = LLK_PATH,
                         genesess_path = GENESESS_PATH)
        LL = Z.predict_loglike(PHN_DF[[PHN]])
        # Remove the folder immediately after loglikelihoods are computed
        shutil.rmtree(z3_temp_path)
        
        # Get sequence likelihood defect (pos-neg difference) and other features
        preds[PHN + '_sld'] = list(np.array(list(LL[0])) - np.array(list(LL[1])))
        preds[PHN + '_ratio'] = list(np.array(list(LL[0]))/np.array(list(LL[1])))
        preds[PHN + '_abs_pos'] = list(LL[0])
        preds[PHN + '_abs_neg'] = list(LL[1])
        pfsa_features = ['sld', 'abs_pos', 'abs_neg', 'ratio']
        
        ## Save the phenotype features
        feature_dfs.append(preds)
        now = time.time()
        features.append(PHN)
    
    ### Join up all the phenotype features into one dataframe
    fit_df = reduce(lambda left,right: pd.merge(left, right, on = JOIN_COLUMNS,
                                            how='outer').drop_duplicates(), feature_dfs)
    fit_df = fit_df.replace([np.inf, -np.inf], np.nan)
    X = fit_df
    # Get back the column names
    X = pd.DataFrame(X, columns = fit_df.columns)
    
    # Aggregate the features for each data channel (e.g. DX, RX) separately
    for channel in used_channels:
        if channel != 'TOTAL':
            channel_cols = [i for i in X.columns if any(j in i for j in phenotype_catalog[channel])]
        # compute aggregations for every derived feature across phenotypes
        for pfsa_feature in pfsa_features:
            X['MEAN_%s_%s' % (channel, pfsa_feature)] = X[[i for i in channel_cols if pfsa_feature in i]].mean(1)
            X['MAX_%s_%s' % (channel, pfsa_feature)] = X[[i for i in channel_cols if pfsa_feature in i]].max(1)
            X['RANGE_%s_%s' % (channel, pfsa_feature)] = X[[i for i in channel_cols if pfsa_feature in i]].max(1) - X[[i for i in X.columns if pfsa_feature in i]].min(1)
            X['STD_%s_%s' % (channel, pfsa_feature)] = X[[i for i in channel_cols if pfsa_feature in i]].std(1)
    return X
        
