#! /usr/bin/env python3
import torch

from classes.baseline import run_baseline
from classes.bilstm_cnn import run_bilstm_cnn
from classes.bilstm import run_bilstm
from classes.distil_bert import run_distil_bert

from classes.commons import check_downloads, get_basic_logger

if __name__ == '__main__':
    logger_main = get_basic_logger("Main", log_path="Log.txt")

    check_downloads()
    # Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Baseline classifiers
    logger_main.info("--- Running baseline classifiers ---")
    run_baseline()
    
    ### Paper classifiers
    logger_main.info("--- Running BiLSTM with CNN classifiers ---")
    run_bilstm_cnn(device=device)    

    ### Custom classifiers
    logger_main.info("--- Running BiLSTM classifiers ---") 
    run_bilstm(device=device)

    ### Transformers classifiers
    logger_main.info("--- Running DistilBERT classifiers ---")
    run_distil_bert(device=device)
    
    


