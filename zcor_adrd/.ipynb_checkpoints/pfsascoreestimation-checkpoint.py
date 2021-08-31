import os
import time
import tempfile
import shutil
import pandas as pd
import numpy as np
from functools import reduce

from zcor_adrd.z3 import *


def generate_pfsa_features(
                 DF, 
                 save_path,
                 POS_PFSA_PATH = '../models/%s/POS.pfsa',
                 NEG_PFSA_PATH = '../models/%s/NEG.pfsa',
                 STOPLIST = [],
                 used_channels = ['DX'],
                 ID_COLUMN = 'patient_id',
                 FEATURE_COLUMNS = [],
                 STAT_COLUMNS = [],
                 phenotype_folder = '../data/phenotypes',
                 LLK_PATH = '../bin/llk',
                 GENESESS_PATH = '../bin/genESeSS',
                 verbose = False):
    
    """Generate loglikelihoods for every generated PFSA model and derive features to use for the classifier.
    
    Args:
      DF ([type]): Dataframe with ternary encodings for given patients.
      savepath (str): File
      POS_PFSA_PATH (str, optional): Filepath to the positive inferred PFSA files. Defaults to 'models/%s/POS.pfsa'.
      NEG_PFSA_PATH (str, optional): Filepath to the negative inferred PFSA files. Defaults to 'models/%s/NEG.pfsa'.
      STOPLIST ([type], optional): List of phenotypes to exclude for inference. Defaults to [].
      used_channels ([type], optional): Data channels used for the PFSA (DX for diagnoses, RX for prescriptions etc.). Defaults to ['DX'].
      FEATURE_COLUMNS ([type], optional): Names of columns from the input dataframe to return in the output dataframe as features. Defaults to [].
      STAT_COLUMNS ([type], optional): Names of columns from the input dataframe to return in the output dataframe for genral reference. Defaults to [].
      phenotype_folder (str, optional): Folder containing phenotypes used for encoding. Defaults to 'data/phenotypes'.
      LLK_PATH (str, optional): Path to llk binary used for loglikelihoods computation. Defaults to 'bin/llk'.
      GENESESS_PATH (str, optional): Path to genESeSS binary used for PFSA inference. Defaults to 'bin/genESeSS'.
      verbose (bool, optional): Verbosity switch. Defaults to False.
    """
    feature_dfs = []
    features = []
    
    # Assert execution permission
    os.chmod(GENESESS_PATH, 0o777)
    os.chmod(LLK_PATH, 0o777)
    
    JOIN_COLUMNS = [ID_COLUMN] + FEATURE_COLUMNS + STAT_COLUMNS
    
    
    # Get the phenotypes
    DX = [phn.split('.')[0] for phn in os.listdir(phenotype_folder) if phn.split("_")[0] == 'DX']
    phenotype_catalog = {
            'DX': DX,
            'TOTAL': DX}
    
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
    
    X = X.merge(DF[['patient_id', 'target']], on = 'patient_id')
    X.to_csv(save_path, index = False)
