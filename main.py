#! /usr/bin/env python3
import torch
from classes.commons import check_downloads
from classes.baseline import main as baseline_main

from classes.personal_implementation import train_subjectivity_classification, train_polarity_classification
from classes.paper_implementation import train_subjectivity_classification as train_subjectivity_classification_paper, train_polarity_classification as train_polarity_classification_paper

if __name__ == '__main__':
    check_downloads()
    
    ### Baseline classifiers
    print(f"--- Running baseline classifiers ---")
    # baseline_main()

    ### Custom classifiers
    print(f"--- Running custom classifiers ---")
    # Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train subjectivity classifier with custom implementation
    subj_class = train_subjectivity_classification(device=device)

    # Train polarity classifier with custom implementation
    pol_class = train_polarity_classification(device=device)

    # Train subjectivity classifier with custom implementation
    # subj_class = train_subjectivity_classification_paper(device=device)

    # Train polarity classifier with custom implementation
    # pol_class = train_polarity_classification_paper(device=device)

    