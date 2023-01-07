#! /usr/bin/env python3
import torch
from classes.commons import check_downloads
from classes.baseline import main as baseline_main

from classes.improved import train_subjectivity_classification, train_polarity_classification

if __name__ == '__main__':
    check_downloads()
    
    ### Baseline classifiers
    print(f"--- Running baseline classifiers ---")
    # baseline_main()

    ### Custom classifiers
    print(f"--- Running custom classifiers ---")
    # Hyperparameters
    epochs = 40
    lr = 0.001
    weight_decay = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train subjectivity classifier
    subj_class = train_subjectivity_classification(epochs, lr, weight_decay, device)

    # Train polarity classifier
    pol_class = train_polarity_classification(epochs, lr, weight_decay, device)