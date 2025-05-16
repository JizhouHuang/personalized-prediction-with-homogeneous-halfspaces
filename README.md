# Data Analysis Project

This project is designed for data analysis and manipulation using Python libraries NumPy and Pandas. It provides a structured approach to load, clean, analyze, and summarize datasets.

## Project Structure

```
data-analysis-project
├── src
│   ├── main.py          # Entry point for the application
│   └── utils
│       └── helpers.py   # Utility functions for data processing
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
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

experiment type: main, main_sparse, main-baseline

dataset name: haberman, diabetes, hepatitis, hypothyroid, wdbc


For fast-run uncomment 
``` config_file_path = "src/config/model/model_toy.yaml```
in ```src/main.py```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features.
