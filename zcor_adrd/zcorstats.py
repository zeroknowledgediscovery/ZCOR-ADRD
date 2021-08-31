import sklearn
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, classification_report, roc_auc_score, f1_score, auc, precision_recall_curve
from sklearn import metrics

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import scipy as sp

def closest_df(df, var, x):
    """
    Get the datafrme row with column value closest to the given one
    """
    return df.iloc[(df[var]-x).abs().argsort()[:1]] 

def classification_outcome(targets, preds):
    """
    Return classification outcomes based on true values and predicted values
    
    Args:
      targets ([type]): list of true values
      preds ([type]): list of predicted values
    
    Returns:
      list of classification outcomes
    """
    out = []
    for target, pred in zip(targets, preds):
        if target == 1:
            if target == pred:
                out.append('TP')
            else:
                out.append('FN')
        else:
            if target == pred:
                out.append('TN')
            else:
                out.append('FP')
    return out

def getPPV(row, RHO):
    """
    Get PPV metric
    """
    return row.tpr/(row.tpr + row.fpr * (1./RHO - 1))
                
def negative_predictive_value(outcomes):   
    """
    Get NPV metric
    """
    try:
        TN = len([i for i in outcomes if i == 'TN'])
        FN = len([i for i in outcomes if i == 'FN'])
        return TN/(TN+FN)
    except:
        return 0

                
                
def calc_U(y_true, y_score,cb=0.99):
    '''
    Calculate AUC and confidence bounds / pvalue 
    on AUC using U test correspondance
    '''    
    ZALPHA={0.9:1.645,
            0.95:1.96,
            .99:2.58,
            .999:3.27}
    n1 = np.sum(y_true == 1)
    n0 = len(y_score) - n1
    order = np.argsort(y_score)
    rank = np.argsort(order)
    rank += 1
    U1 = np.sum(rank[y_true == 1]) - n1*(n1+1)/2
    U0 = np.sum(rank[y_true == 0]) - n0*(n0+1)/2
    AUC1 = U1/(n1*n0)
    AUC0 = U0/(n1*n0)
    
    EU1=n0*n1*0.5
    s1=np.sqrt(n0 * n1 * (n0 + n1 + 1)/12.)
    U1_z= (U1-EU1)/s1
    U0_z= (U0-EU1)/s1
    p = sp.stats.norm.sf(abs(U1_z))*2 #twosided
    
    CF=(ZALPHA[cb] * s1)/(n1 * n0)
    
    if AUC1 > AUC0:
        return AUC1, p, U1, U1_z, CF
    return AUC0, p, U0, U0_z, CF  

#########################################################

def compute_performance(
                input_path,
                performance_feature,
                AUC_CONFIDENCE_BOUND = 0.99,
            ):
    """
    Generate ROC, PRC, performance stats;
    """
    performance_stats = {
        i:[] for i in [
            'fpr',
            'auc',
            'confidence',
            'cb',
            'p',
            'tpr',
            'ppv',
            'npv',
            'pos_LR',
            'neg_LR',
            'f1',
            'accuracy'
        ]
    }
    
    DF = pd.read_csv(input_path)
    
    # Filter out patients without valid entry for performance feature
    DF = DF[~DF[performance_feature].isnull()]

    fpr, tpr, thresholds = roc_curve(DF.target, DF[performance_feature])
    roc_auc = roc_auc_score(DF.target, DF[performance_feature])
    ROC = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'threshold': thresholds
    })        
    ROC.to_csv("../data/performance/ROC.csv", index = False)

    PRC = ROC.copy()
    PREVALENCE = DF.target.value_counts(normalize=True)[1]
    PRC['ppv'] = PRC.apply(lambda x: getPPV(x, PREVALENCE), axis = 1)
    PRC = PRC[['ppv', 'tpr']]
    PRC.columns = ['precision', 'recall']
    PRC.to_csv("../data/performance/PRC.csv", index = False)

    AUC, p, U, U_z, CF = calc_U(DF['target'], DF[performance_feature], cb=0.99)

    for specificity in [.99, .95, .90, .85, .80, .75]:
        FPR = round(1 - specificity, 3) 
        TPR = closest_df(ROC, 'fpr', FPR)['tpr'].iloc[0]
        threshold = closest_df(ROC, 'fpr', FPR)['threshold'].iloc[0]
        DF['binary_prediction'] = [int(i >= threshold) for i in DF[performance_feature]]
        DF['prediction_outcome'] = classification_outcome(DF['target'], DF['binary_prediction'])

        sensitivity = TPR

        precision = metrics.precision_score(DF['target'], DF['binary_prediction'])
        recall = metrics.recall_score(DF['target'], DF['binary_prediction'])
        accuracy = metrics.accuracy_score(DF['target'], DF['binary_prediction'])
        F1 = metrics.f1_score(DF['target'], DF['binary_prediction'])

        PPV = precision
        NPV = negative_predictive_value(DF['prediction_outcome'])

        pos_LR = sensitivity/(1 - specificity)
        neg_LR = (1 - sensitivity)/specificity

        performance_stats['auc'].append(AUC)
        performance_stats['confidence'].append(CF)
        performance_stats['cb'].append(AUC_CONFIDENCE_BOUND)
        performance_stats['p'].append(p)
        performance_stats['fpr'].append(FPR)
        performance_stats['tpr'].append(TPR)
        performance_stats['ppv'].append(PPV)
        performance_stats['npv'].append(NPV)
        performance_stats['pos_LR'].append(pos_LR)
        performance_stats['neg_LR'].append(neg_LR)
        performance_stats['f1'].append(F1)
        performance_stats['accuracy'].append(accuracy)

    performance_stats = pd.DataFrame(performance_stats)
    performance_stats.to_csv("../data/performance/performance_stats.csv", index = False)
    
    

def plot_curves(performance_stats,
                ROC,
                PRC):
    """
    Plot ROC, PRC;
    """
    # ROC
    plt.figure(figsize = (7,7))
    plt.minorticks_on()
    # Set grid to use minor tick locations. 
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    # Customize the minor grid
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    plt.plot(ROC['tpr'], ROC['fpr'], label='ROC curve (area = %0.3f)' % (performance_stats['auc'].iloc[0]), color = 'k')

    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR or (1 - Specifity)')
    plt.ylabel('TPR or (Sensitivity)')
    plt.legend(loc="lower right")

    # PRC
    plt.figure(figsize = (7,7))
    plt.minorticks_on()
    # Set grid to use minor tick locations. 
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    # Customize the minor grid
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    plt.plot(PRC['recall'], PRC['precision'], label='PRC curve', color = 'k')

    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR or (1 - Specifity)')
    plt.ylabel('TPR or (Sensitivity)')
    plt.legend(loc="lower right")