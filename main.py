#! /usr/bin/env python3
import torch
from classes.commons import check_downloads
from classes.baseline import main as baseline_main

from classes.personal_implementation import train_subjectivity_classification, train_polarity_classification, make_dirs
from classes.paper_implementation import paper_make_dirs, paper_train_subjectivity_classification, paper_train_polarity_classification

if __name__ == '__main__':
    try:
        check_downloads()
        make_dirs()
        paper_make_dirs()   
        # Hyperparameters
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ### Baseline classifiers
        print(f"--- Running baseline classifiers ---")
        # baseline_main()
        
        ### Custom classifiers
        print(f"--- Running paper classifiers ---")

        # Train subjectivity classifier with custom implementation
        subj_class = paper_train_subjectivity_classification(epochs=40, device=device)

        # Train polarity classifier with custom implementation
        pol_class = paper_train_polarity_classification(device=device)

        ### Custom classifiers
        print(f"--- Running custom classifiers ---")    

        # Train subjectivity classifier with custom implementation
        subj_class = train_subjectivity_classification(epochs=40, device=device)

        # Train polarity classifier with custom implementation
        pol_class = train_polarity_classification(device=device)
    except Exception as e:
        variables = list(locals().keys()) + list(globals().keys())
        for variable in variables:
            if variable != 'e' and variable != 'torch':
                if variable in locals().keys():
                    del locals()[variable]
                elif variable in globals().keys():
                    del globals()[variable]

        torch.cuda.empty_cache()
        raise e


