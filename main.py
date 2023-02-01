#! /usr/bin/env python3
import torch

from classes.commons import check_downloads, get_basic_logger
from classes.baseline import run_baseline

from classes.bilstm import run_custom 
from classes.bilstm_cnn import run_paper
from classes.distil_bert import run_transformer

if __name__ == '__main__':
    logger_main = get_basic_logger("Main")

    check_downloads()
    # Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Baseline classifiers
    logger_main.info("--- Running baseline classifiers ---")
    run_baseline()
    
    ### Paper classifiers
    logger_main.info("--- Running BiLSTM with CNN classifiers ---")
    run_paper(device=device)    

    ### Custom classifiers
    logger_main.info("--- Running BiLSTM classifiers ---") 
    run_custom(device=device)

    ### Transformers classifiers
    logger_main.info("--- Running DistilBERT classifiers ---")
    run_transformer(device=device)
    
    


