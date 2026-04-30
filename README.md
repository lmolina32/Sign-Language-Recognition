# Sign-Language-Recognition Project

**Team:** Leonardo Molina, Alphonsus Koong Bok Hui  
**Course:** CSE 40535 Computer Vision, Spring 2026  

# Repository Structure
```bash
sign-language-recognition/
├── docs/                    # Project assignments and answers 
├── results/                 # Results from our classifiers and models
├── sample_data/             # Sample Images
├── scripts/                 # Download Data from source and demos
├── src                      # src code for entire project
│   ├── classifer.py         # SVM and CNN classes live here 
│   ├── dataloader.py        # functions to load the data 
│   ├── eval.py              # evaluate SVM and CNN
│   ├── pipeline.py          # Full pipeline to transform imgs
│   └── train.py             # Train SVM and CNN 
└── tests/                   # Testing src code 
```

**NOTE: Project 3 update was moved to `docs/project03_update.md`** 

**NOTE: Project 4 update was moved to `docs/project04_update.md`**

**NOTE: Project 5 (final) update was moved to `docs/project05_update.md`**

# Presentation of project 
This presentation of the project can be found [here](https://drive.google.com/file/d/1DI7TsgrMnAGYE0NXD8tr-18etQMK0ZW-/view?usp=sharing). 

# Live Demos for CNN and SVM 
The live demos to run test images on the CNN and SVM are found in scripts. The usage is shown below.
```bash
python ./scripts/cnn_demo.py sample_data/test_data/P11_2_143.jpg --cnn-path results/cnn_results/final_cnn_model.pkl
python ./scripts/svm_demo.py sample_data/test_data/P11_2_143.jpg --svm-path results/svm_results/svm_model.pkl
```

# Random test sample (Project 5 deliverable)
To pick one random sample from the held-out test set and run both classifiers on it:
```bash
python scripts/random_test_demo.py
# or: python scripts/random_test_demo.py --seed 42   # for a reproducible pick
```
See `docs/project05_update.md` for the final report (test-set description, accuracy, analysis of the val→test drop, NN vs. classical comparison, and proposed improvements).
