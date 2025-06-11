import tensorflow as tf
import numpy as np
import os
import argparse
import json 
import pandas as pd
from time import time
import multiprocessing

"""
Here we process the results, given a model discription, we first get the metrics for the validation data (such as threshodl dan f1 score)
After this we calulate the same but for the test data. 


"""



def main(config):

    output_shape=config['oo']
    task=config['t']
    input_size=config['in']
    print(f'\033[93mRunning Resnet {input_size} {output_shape} {task} on GPU \033[0m')
    ## Parameters
    fn_base = f'../Results/EWSN_ds_{input_size}/ResNet_{output_shape}/res_{task}/'

    ## Prepare data for validation check: 
    val_f1,val_pres,val_recall,val_th = get_parameters_validation(fn_base)

    ## Calculate Test
    f1,pres,recall = get_parameters_Testing(fn_base,val_th)
    result = {'val_f1':val_f1,
              'val_pres':val_pres,
              'val_recall':val_recall,
              'val_th':val_th,
              'f1':f1,
              'pres':pres,
              'recall':recall}
    with open(fn_base+'/Results/results.json', "w") as outfile:
            json.dump(result, outfile,indent=4)

def get_parameters_Testing(fn_base, th):

    embeddings = np.load(fn_base+'/Results/embeddings.npy',allow_pickle=True).item()
    idx_array = embeddings['idx']
    embedding_array = embeddings['embedding']
    embedding_by_idx = {index: embedding for index, embedding in zip(idx_array, embedding_array)}

    corr_file = '../Dataset/EWSN_disjunct/Okriftel_Testing_correlation.csv.gz'
    corr = pd.read_csv(corr_file, compression='gzip')
    # Get the embeddings for column 'i'
    embeddings_i = corr['i'].map(embedding_by_idx).tolist()
    embeddings_i_array = np.array(embeddings_i)

    # Get the embeddings for column 'j'
    embeddings_j = corr['j'].map(embedding_by_idx).tolist()
    embeddings_j_array = np.array(embeddings_j)

    # Calculate the Euclidean distance using vectorized NumPy operations
    corr['dist'] = np.linalg.norm(embeddings_i_array - embeddings_j_array, axis=1)
    
    # get th 

    df = corr

    TP = df[(df['bit_diff'] == 0) & (df['dist'] < th)].shape[0]
    # TN = df[(df['bit_diff'] != 0) & (df['dist'] >= th)].shape[0]
    FP = df[(df['bit_diff'] != 0) & (df['dist'] < th)].shape[0]
    FN = df[(df['bit_diff'] == 0) & (df['dist'] >= th)].shape[0]

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculate F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1,precision,recall




def get_parameters_validation(fn_base):

    embeddings = np.load(fn_base+'/Results/embeddings_val.npy',allow_pickle=True).item()
    idx_array = embeddings['idx']
    embedding_array = embeddings['embedding']
    embedding_by_idx = {index: embedding for index, embedding in zip(idx_array, embedding_array)}

    corr_file = '../Dataset/EWSN_disjunct/Okriftel_Validation_correlation.csv.gz'
    corr = pd.read_csv(corr_file, compression='gzip')
    # Get the embeddings for column 'i'
    embeddings_i = corr['i'].map(embedding_by_idx).tolist()
    embeddings_i_array = np.array(embeddings_i)

    # Get the embeddings for column 'j'
    embeddings_j = corr['j'].map(embedding_by_idx).tolist()
    embeddings_j_array = np.array(embeddings_j)

    # Calculate the Euclidean distance using vectorized NumPy operations
    corr['dist'] = np.linalg.norm(embeddings_i_array - embeddings_j_array, axis=1)
    
    # get th 
    t = time()
    ths =  np.arange(0,2,0.001)
    m_f1 = 0 
    m_precision = 0
    m_recall = 0
    m_th = 0 
    df = corr
    for th in ths:
        TP = df[(df['bit_diff'] == 0) & (df['dist'] < th)].shape[0]
        # TN = df[(df['bit_diff'] != 0) & (df['dist'] >= th)].shape[0]
        FP = df[(df['bit_diff'] != 0) & (df['dist'] < th)].shape[0]
        FN = df[(df['bit_diff'] == 0) & (df['dist'] >= th)].shape[0]

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        # Calculate F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if f1>m_f1:
            m_f1 = f1
            m_precision = precision
            m_recall = recall
            m_th = th
    
    print(time()-t, 'Time for search')
    return m_f1,m_precision,m_recall,m_th


    
if __name__ == "__main__":

    config = []
    sizes = [4,6,8,10,12,14,16,18,20,22,24]
    inputs = (16,32,64,128,256)

    
    for isize in inputs:
        for size in sizes:
            for task in range(4):
                config.append({'in':isize,'oo':size,'t':task})
        
    num_processes = 64  # Use all available cores
    print(f"Running {len(config)} models in parallel using {num_processes} processes.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(main, config)



