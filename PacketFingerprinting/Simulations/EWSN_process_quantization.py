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

class Quantizer:
    """ This quantizer is build around the train data and applied to the test_data"""
    def __init__(self,features:np.ndarray, n_bits=32):
        self.k = 2 ** n_bits
        self.n_bits = n_bits
        f_min = np.min(features, axis=0, keepdims=True)
        f_max = np.max(features, axis=0, keepdims=True)
    
        scale = (f_max - f_min) / (self.k - 1)
        scale[scale == 0] = 1e-16  # Prevent division by zero

        self.f_min=f_min
        self.scale = scale

    def quantize(self,seq):
        return np.round((seq - self.f_min) / self.scale)  # or smaller dtype depending on bits
    
    def dequantize(self,seq):
        return self.f_min + seq * self.scale   
    
    def sim(self,seq):
        return self.f_min + np.round((seq - self.f_min) / self.scale)* self.scale  




def main(config):
    output_shape=config['oo']
    input_size=config['in']
    # read best thing
    df_bm = pd.read_csv('../Analysis/res/best_models.csv', index_col=0)
    model = df_bm.loc[(df_bm['in_size']==input_size)&(df_bm['out_size']==output_shape)]
    in_size,out_size,res,downsample,val_f1,val_pres,val_recall,val_th,f1,pres,recall = model.values[0]
    n_bits = 8
    result = {}
    for n_bits in range(1,33):    
        ## Parameters
        fn_base = f'../Results/EWSN_ds_{int(in_size)}/ResNet_{int(out_size)}/res_{int(res)}'
        t = time()
        # ## Prepare data for validation check: 
        val_f1,val_pres,val_recall, Quanti = get_parameters_validation(fn_base, val_th,n_bits)
        print(f'\033[93mRunning Validation {input_size} {output_shape} {n_bits}:  {time()-t:.2f}s \033[0m')
        
        # ## Calculate Test
        t = time()
        f1,pres,recall = get_parameters_Testing(fn_base,val_th,Quanti)
        print(f'\033[92mRunning Testing {input_size} {output_shape} {n_bits}:  {time()-t:.2f}s \033[0m') 

        result[n_bits] = {
                'in_size':input_size,
                'out_size':out_size,
                'quant':n_bits,
                'tot_bits':n_bits*out_size,
                'tot_bytes': np.ceil(n_bits*out_size/8),
                'val_f1':val_f1,
                'val_pres':val_pres,
                'val_recall':val_recall,
                'val_th':val_th,
                'f1':f1,
                'pres':pres,
                'recall':recall}

    with open(fn_base+'/Results/quant.json', "w") as outfile:
            json.dump(result, outfile,indent=4)

def get_parameters_Testing(fn_base, th, quant: Quantizer):

    embeddings = np.load(fn_base+'/Results/embeddings.npy',allow_pickle=True).item()
    idx_array = embeddings['idx']
    embedding_array = embeddings['embedding']
    embedding_by_idx = {index: embedding for index, embedding in zip(idx_array, embedding_array)}

    corr_file = '../Dataset/EWSN_disjunct/Okriftel_Testing_correlation.csv.gz'
    corr = pd.read_csv(corr_file, compression='gzip')
    # Get the embeddings for column 'i'
    embeddings_i = corr['i'].map(embedding_by_idx).tolist()
    embeddings_i_array = quant.sim(np.array(embeddings_i))

    # Get the embeddings for column 'j'
    embeddings_j = corr['j'].map(embedding_by_idx).tolist()
    embeddings_j_array = quant.sim(np.array(embeddings_j))

    # Calculate the Euclidean distance using vectorized NumPy operations
    corr['dist'] = np.linalg.norm(embeddings_i_array - embeddings_j_array, axis=1)
    
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


def get_parameters_validation(fn_base,th,n_bits):

    embeddings = np.load(fn_base+'/Results/embeddings_val.npy',allow_pickle=True).item()
    idx_array = embeddings['idx']
    embedding_array = embeddings['embedding']
    embedding_by_idx = {index: embedding for index, embedding in zip(idx_array, embedding_array)}

    corr_file = '../Dataset/EWSN_disjunct/Okriftel_Validation_correlation.csv.gz'
    corr = pd.read_csv(corr_file, compression='gzip')

    # Get the embeddings for column 'i'
    embeddings_i = corr['i'].map(embedding_by_idx).tolist()
    embeddings_i_array = np.array(embeddings_i)
    
    #learning quantiser 
    quant = Quantizer(embeddings_i_array,n_bits)
    embeddings_i_array = quant.sim(embeddings_i_array)
    # Get the embeddings for column 'j'
    embeddings_j = corr['j'].map(embedding_by_idx).tolist()
    embeddings_j_array = quant.sim(np.array(embeddings_j))

    # Calculate the Euclidean distance using vectorized NumPy operations
    corr['dist'] = np.linalg.norm(embeddings_i_array - embeddings_j_array, axis=1)
    
   
    df = corr
    TP = df[(df['bit_diff'] == 0) & (df['dist'] < th)].shape[0]
    # TN = df[(df['bit_diff'] != 0) & (df['dist'] >= th)].shape[0]
    FP = df[(df['bit_diff'] != 0) & (df['dist'] < th)].shape[0]
    FN = df[(df['bit_diff'] == 0) & (df['dist'] >= th)].shape[0]

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculate F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1,precision,recall,quant


    
if __name__ == "__main__":

    config = []
    sizes = [4,6,8,10,12,14,16,18,20,22,24]
    inputs = (128,256)
    quanti = range(2,33)
    
    for isize in inputs:
        for size in sizes:
                config.append({'in':isize,'oo':size})
                
    num_processes =  multiprocessing.cpu_count()  # Use all available cores
    print(f"Running {len(config)} models in parallel using {num_processes} processes.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(main, config)

