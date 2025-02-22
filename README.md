# Fintech Adoption Determinants – ANN Analysis

This repository contains a Python script that employs Artificial Neural Networks (ANN) to investigate the determinants of fintech adoption in Morocco. The analysis uses a 10-fold cross-validation approach to ensure robustness. 

## Overview

1. **Data Preparation**  
   - Loads a CSV file of survey responses.  
   - Creates composite variables (e.g., Performance Expectancy, Social Influence, etc.) by averaging survey items.

2. **ANN Models**  
   - **FullANN**: Uses all constructs (PE, SI, EE, FC, Big Five traits) to predict Intention to Use (IT).  
   - **SEMANN**: Includes only the significant variables identified by a prior SEM analysis (e.g., PE, SI, FC).

3. **Cross-Validation & Metrics**  
   - Utilizes 10-fold cross-validation to split the dataset.  
   - Evaluates performance using RMSE, R², and Adjusted R² for both training and validation sets.

## Requirements

- Python 3.7+  
- [NumPy](https://numpy.org/)  
- [Pandas](https://pandas.pydata.org/)  
- [scikit-learn](https://scikit-learn.org/)  
- [TensorFlow / Keras](https://www.tensorflow.org/)  

Install them (if not already) via:  
```bash
pip install numpy pandas scikit-learn tensorflow
```

## Usage

1. **Update File Paths:**  
   - Replace `"YOUR_DATA_FILE.csv"` in the script with your own data file path.  
   - Update `"YOUR_OUTPUT_FILE_FullANN.csv"` and `"YOUR_OUTPUT_FILE_SEMANN.csv"` to your preferred output filenames.

2. **Run the Script:**  
   ```bash
   python fintech_ann_analysis.py
   ```

3. **Interpret Results:**  
   - Two CSV files (for FullANN & SEMANN) containing the RMSE, R², and Adjusted R² metrics will be generated.  
   - Use these metrics to compare model performance and validate your findings.

## Project Structure

```
├── fintech_ann_analysis.py     # Main ANN script
├── data/
│   └── YOUR_DATA_FILE.csv      # Example data (not included)
└── README.md                   # This file
```

## License

Feel free to specify a suitable license (e.g., MIT, Apache 2.0) or remove this section if you prefer.

---

This short README provides a concise overview of how to run and interpret the ANN analysis for fintech adoption. Feel free to expand on any details or provide additional context to suit your project’s needs.
