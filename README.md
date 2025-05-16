# Data Analysis Project

This project is designed for personalized prediction mainly using PyTorch.

## Project Structure

```
data-analysis-project
├── src
│   ├── main.py                      # evaluation of personalized predictor
│   ├── main_sparse.py               # evaluation of sparse predictor
│   ├── main_baseline.py             # evaluation of other standard machine learning models
│   ├── config
│   │   ├── data                     # datasets information
│   │   └── model                    # personalized and sparse model hyperparameters for different datasets
│   ├── data                         # UCI data files and preprocessing methods
│   ├── models
│   │   ├── baseline_learner.py      # learning algorithms for logistic regression, SVM, XGBoost, random forest
│   │   ├── predictor_learner.py     # robust list learning algorithm for sparse linear classifiers
│   │   ├── selector_learner.py      # projected gradient descent for learning homogeneous halfspace subsets
│   │   └── personalized_learner.py  # personalized prediction scheme
│   └── utils
│       ├── data.py                  # data structures
│       └── simple_models.py         # custom linear predictive model
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd personalized-prediction-with-homogeneous-halfspaces
   ```

## Usage

To run the evaluation experiments, execute the following command:
```
python -m src.<experiment type>.py --data_name <dataset name> --num_exp <repeat number of the experiment>
```

experiment type: main, main_sparse, main_baseline

dataset name: haberman, diabetes, hepatitis, hypothyroid, wdbc


Note: for fast run, uncomment 
```config_file_path = "src/config/model/model_toy.yaml```
in ```src/main.py```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features.
