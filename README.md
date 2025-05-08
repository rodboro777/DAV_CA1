# DAV_CA1 Environmental Performance Analysis Report CA

This project analyzes environmental performance indicators across different countries using various data analysis techniques including PCA, clustering, and composite scoring.

## Project Structure

The analysis is performed in `DAV_CA1.py`, which contains a `DataAnalyzer` class

### Results Directory Structure

The `/results` directory contains the output of various analyses:

```
results/
├── clustering/       # Contains clustering analysis outputs
│   ├── dendrogram plots
│   ├── kmeans results
│   └── DBSCAN results
├── composite/        # Composite indicator analysis
│   ├── composite_indicator.csv
│   ├── composite_indicator_ranked.csv
│   ├── top_10_countries.csv
│   └── bottom_10_countries.csv
├── correlation/      # Correlation analysis results
│   ├── correlation_matrix.csv
│   ├── correlation_matrix.png
│   └── p_values.csv
├── pca/             # Principal Component Analysis results
│   ├── explained_variance.csv
│   ├── feature_importance.csv
│   ├── pca_results.csv
│   └── variance_explained.csv
└── visualization/    # Visualization outputs
    └── composite_world_map.html/png

```


## Usage

To run the analysis:

```python
python DAV_CA1.py
```

The script will automatically:
1. Load and preprocess the data
2. Perform all analyses
3. Generate results in the respective directories

## Dependencies

- pandas
- scikit-learn
- matplotlib
- seaborn
- numpy
- scipy
- geopandas
- plotly
