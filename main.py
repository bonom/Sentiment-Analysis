#! /usr/bin/env python3
import torch
from classes.commons import check_downloads
from classes.baseline import run_baseline

from classes.custom import run_custom 
from classes.paper import run_paper

if __name__ == '__main__':
    check_downloads()
    # Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Baseline classifiers
    print(f"--- Running baseline classifiers ---")
    run_baseline()
    
    ### Custom classifiers
    print(f"--- Running paper classifiers ---")
    run_paper(device=device)    

    ### Custom classifiers
    print(f"--- Running custom classifiers ---")    
    run_custom(device=device)
    
    


