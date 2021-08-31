import os
import time
import tempfile
import shutil

import pandas as pd
import numpy as np

from functools import reduce

from zcor_adrd.z3 import *

def generate_pfsa(
                 PFSA_SET,
                 PFSA_PATH = '../models/%s',
                 STOPLIST = [],
                 POS_EPSILON = 0.5,
                 NEG_EPSILON = 0.5,
                 phenotype_folder = '../data/phenotypes',
                 GENESESS_PATH = '../bin/genESeSS',           
                 verbose = False): 
    """Infer PFSA models for every phenotype provided with given data
    
    Args:
      PFSA_SET ([type]): Dataframe with ternary encodings for given patients.
      PFSA_PATH (str, optional): Filepath to save the inferred PFSA models to. Defaults to 'models/%s'.
      STOPLIST ([type], optional): List of phenotypes to exclude for inference. Defaults to [].
      POS_EPSILON (float, optional): Epsilon value for the inference of the positive PFSA models. Defaults to 0.5.
      NEG_EPSILON (float, optional): Epsilon value for the inference of the negative PFSA models. Defaults to 0.5.
      phenotype_folder (str, optional): Folder containing phenotypes used for encoding. Defaults to 'data/phenotypes'.
      GENESESS_PATH (str, optional): Path to genESeSS binary used for PFSA inference. Defaults to 'bin/genESeSS'.
      verbose (bool, optional): Verbosity switch. Defaults to False.
    """
    # Assert execution permission
    os.chmod(GENESESS_PATH, 0o777)
    
    # Get the phenotypes
    DX = [phn.split('.')[0] for phn in os.listdir(phenotype_folder) if phn.split("_")[0] == 'DX']
    phenotype_catalog = {
            'DX': DX,
            'TOTAL': DX}
            
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


        