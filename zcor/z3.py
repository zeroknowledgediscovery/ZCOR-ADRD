#!/usr/bin/env python

""" z3 Classification """

__author__ = "Ishanu Chattopadhyay"
__copyright__ = "Copyright 2018, zed@uchicago "
__credits__ = ["Dmytro Onishchenko", "Yi Huang"]
__license__ = "GPL"
__version__ = "0.314"
__maintainer__ = "Dmytro Onishchenko"
__email__ = "ishanu@uchicago.edu"
__status__ = "beta"

import pandas as pd
import numpy as np
import os.path
import subprocess
import os
import glob as glob


class Z3Classifier(object):
    """
    Class implementing z3 time-series classification
    The input streams are assumed to be short
    but many, and have non-trivial temporal patterns.
    We first learn PFSA models for each category, using
    all exemplars in the category simultaneously,
    using multistream GenESeSS. Then, we use the 
    PFSA likelihood computation to ascertain which 
    category each test stream belongs to.

    Author:
      zed.uchicago.edu 2018
    """

    def __init__(self,
                 genesess_path='bin/genESeSS',
                 llk_path='bin/llk',
                 trainposfile='trainpos.dat',
                 trainnegfile='trainneg.dat',
                 use_own_pfsa = False,
                 posmod="POS.pfsa",  
                 negmod="NEG.pfsa",
                 result_path='./Z3_temp', 
                 ):
        """Init
        
        Args:
          genesess_path (str, optional): path to genESeSS binary. Defaults to 'bin/genESeSS'.
          genesess_path (str, optional): path to llk binary. Defaults to 'bin/llk'.
          trainposfile (str, optional): name for the preprocessed sequences file for positive PFSA inference. Defaults to 'trainpos.dat'.
          trainnegfile (str, optional): name for the preprocessed sequences file for negative PFSA inference. Defaults to 'trainneg.dat'.
          use_own_pfsa (bool, optional): Indicate whether to use previously trained PFSA models for llk computation. Defaults to False.
          posmod (str, optional): Filepath to pre-trained positive PFSA model. Defaults to 'POS.pfsa'.
          negmod (str, optional): Filepath to pre-trained negative PFSA model. Defaults to 'NEG.pfsa'.
          result_path (str, optional): Path to save inferred PFSA models to. Defaults to './Z3_temp'.
        Returns:
           NA
        """

        try: 
            os.makedirs(result_path)
        except OSError:
            if not os.path.isdir(result_path):
                raise

        assert os.path.exists(genesess_path), "GenESeSS binary cannot be found." 
        assert os.path.exists(llk_path), "Pfsa likelihood estimator binary not found"

        self.GENESESS = genesess_path
        self.LLK = llk_path
        self.result_path = result_path
        self.use_own_pfsa = use_own_pfsa
        if not use_own_pfsa:
            self.neg_model = os.path.join(result_path,negmod)
            self.pos_model = os.path.join(result_path,posmod)
        else:
            self.neg_model = negmod
            self.pos_model = posmod
       
                 
    def produce_file(self, df, path):
        """Preprocess the contents of input dataframe for the PFSA inference, save to the provided filepath.
        
        Args:
          df ([type]): Input encodings dataframe.
          path (str): Filepath to save preprocessed data to.
        Returns:
           path (str): Filepath where preprocessed data is saved.
        """  
        path = os.path.join(self.result_path,path)
        with open(path, 'w') as f:
            if "record" in df.columns:
                for row in df.record:
                    f.write(row + '\n')
            else:
                for row in df.values:
                    f.write(' '.join([str(i) for i in row]) + '\n')
        return path
                 
    def fit(self,
            df,
            peps=0.25,
            neps=0.2,
            pos_file = 'trainpos.dat',
            neg_file = 'trainneg.dat',
            verbose=False):
        """Fit positive and negative PFSA models
        
        Args:
          df ([type]): Input encodings dataframe.
          peps (float, optional): epsilon parameter for positive PFSA model. Defaults to 0.25.
          neps (float, optional): epsilon parameter for negative PFSA model. Defaults to 0.2.
          posfile (str, optional): name for the preprocessed sequences file for positive PFSA inference. Defaults to 'trainpos.dat'.
          negfile (str, optional): name for the preprocessed sequences file for negative PFSA inference. Defaults to 'trainneg.dat'.
          verbose (bool, optional): Verbosity switch. Defaults to False.
          
        Returns:
           NA
        """           
        pos = df[df['target'] == 1]
        neg = df[df['target'] == 0]
        # Generate the Positive/Negative Datafiles out of input df
        train_pos = self.produce_file(pos.drop('target', 1), pos_file)
        train_neg = self.produce_file(neg.drop('target', 1), neg_file)
                 
        # NEG MODEL         
        sstr = self.GENESESS+' -f ' + train_neg + ' -D row -T symbolic -o '\
              +self.neg_model+' -F -t off -v 0 -e '+str(neps)
        res = subprocess.check_output(sstr, shell=True)
        if verbose:
            print(res)
                 
        # POS MODEL
        sstr = self.GENESESS+' -f '+train_pos+' -t off'\
              +' -F -v 0 -D row -T symbolic -o '+self.pos_model+' -e '+str(peps)
        res = subprocess.check_output(sstr, shell=True)
        if verbose:
            print(res)
            
        os.remove(train_pos)
        os.remove(train_neg)
        return
        
    def predictions(self, df):
        """Fit positive and negative PFSA models
        
        Args:
          df (str): filepath of preprocessed data to generate the loglikelihoods for 
          
        Returns:
           ([type]): list of numpy array with positive loglikelihoods and a numpy array with negative loglikelihoods.
        """   
        llpos=self.LLK+' -s ' + df + ' -f '+self.pos_model
        llneg=self.LLK+' -s ' + df + ' -f '+self.neg_model
        
        POS = np.array(subprocess.check_output(llpos,
                                               shell=True).split()).astype(float)
        NEG = np.array(subprocess.check_output(llneg,
                                               shell=True).split()).astype(float)
        return [POS, NEG]
    
    def predict_loglike(self, X_test):
        """Fit positive and negative PFSA models
        
        Args:
          df (str): dataframe with encodings to generate loglikelihoods for 
          
        Returns:
           ([type]): list of numpy array with positive loglikelihoods and a numpy array with negative loglikelihoods.
        """   
        test = self.produce_file(X_test, 'testfile.dat')
        return self.predictions(test)
    
