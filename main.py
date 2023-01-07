import torch
from classes.commons import check_downloads
from classes.baseline import main as baseline_main

from classes.improved import train_subjectivity_classification

if __name__ == '__main__':
    check_downloads()
    
    ### Baseline classifiers
    print(f"--- Running baseline classifiers ---")
    # baseline_main()

    ### Custom classifiers
    print(f"--- Running custom classifiers ---")
    train_subjectivity_classification(100, 0.001, 0.0001, 'cpu' if not torch.cuda.is_available() else 'cuda:0')