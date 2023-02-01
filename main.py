#! /usr/bin/env python3
import torch
from classes.commons import check_downloads
from classes.baseline import run_baseline

from classes.bilstm import run_custom 
from classes.bilstm_cnn import run_paper

from classes.distil_bert import run_transformer

if __name__ == '__main__':
    check_downloads()
    # Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Baseline classifiers
    print(f"--- Running baseline classifiers ---")
    # run_baseline()
    
    ### Custom classifiers
    print(f"--- Running paper classifiers ---")
    # run_paper(device=device)    

    ### Custom classifiers
    print(f"--- Running custom classifiers ---")    
    # run_custom(device=device)

    ### Transformers
    print(f"--- Running transformers ---")
    run_transformer(device=device)
    
    


