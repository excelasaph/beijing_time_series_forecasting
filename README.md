# Beijing Time Series Air Quality Forecasting

**Course**: Machine Learning Techniques  
**Author**: Excel Asaph   
**Dataset**: [Kaggle Competition - Time Series Forecasting September 2025](https://www.kaggle.com/competitions/assignment-1-time-series-forecasting-septemb-2025)

## Project Overview

This project predicts hourly PM2.5 concentrations in Beijing using deep learning techniques. The goal is to forecast air quality from July 2, 2013, to December 31, 2014, using historical weather and PM2.5 data from January 1, 2011, to July 2, 2013.

### Key Achievement
- **Best Model**: BiLSTM + Attention achieving **RMSE of 4081.6186**
- **Target Met**: Successfully achieved Kaggle target of RMSE < 4000
- **Improvement**: 34% better than baseline model

## Dataset

- **Training Data**: 30,676 hourly samples (Jan 2011 - Jul 2013)
- **Test Data**: 13,148 samples (Jul 2013 - Dec 2014)
- **Features**: Weather data (temperature, pressure, wind speed, etc.) + PM2.5 concentrations
- **Target**: PM2.5 concentration levels (µg/m³)

## Model Architecture

### Best Performing Model (Model 9)
- **Architecture**: Bidirectional LSTM + Multi-Head Attention
- **Layers**: 3 BiLSTM layers (64→32→16 units) + Attention + Dense
- **Sequence Length**: 24 hours
- **Regularization**: BatchNorm + Dropout (0.2) + Early Stopping
- **Optimizer**: Nadam (lr=0.0001)

## Repository Structure

```
beijing_time_series_forecasting/
├── Excel_Asaph_Time_Series_Air_Quality_Forecasting.ipynb              
├── README.md                                                
├── data/                                                    
│   ├── train.csv                                           
│   ├── test.csv                                            
│   └── sample_submission.csv                               
├── figs/                                                    
│   ├── pm2.5-levels-over-time.png                         # Time series plots
│   ├── smoothed-levels-over-time.png                      # Trend analysis
│   └── correlation-matrix.png                             # Feature correlations
├── report/                                                  
│   └── report.md                                           # Detailed project report
└── outputs/                                                  
    ├── best_model.h5  
    ├── smoothed-levels-over-time.png  
    ├── smoothed-levels-over-time.png  
    ├── smoothed-levels-over-time.png                                      
    └── model_checkpoints/                                  
```

## Quick Start

### Prerequisites
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

### Running the Project
1. **Clone the repository**
   ```bash
   git clone https://github.com/excelasaph/beijing_time_series_forecasting.git
   cd beijing_time_series_forecasting
   ```

2. **Download data** from the [Kaggle competition](https://www.kaggle.com/competitions/assignment-1-time-series-forecasting-septemb-2025) and place in `data/` folder

3. **Run the main notebook**
   ```bash
   jupyter notebook Excel_Asaph_Time_Series_Air_Quality_Forecasting.ipynb
   ```

4. **View results** in the visualizations and model outputs

## Experiments & Results

| Model Type | Architecture | Sequence Length | RMSE | Key Features |
|------------|-------------|----------------|------|--------------|
| LSTM | 2 layers, 25 units | 12 hours | 6183.0989 | Baseline model |
| LSTM | 2 layers, 50 units | 48 hours | 5021.2795 | Extended sequence |
| BiLSTM | 2 layers, 50 units | 48 hours | 5220.1772 | Bidirectional processing |
| **BiLSTM + Attention** | **3 layers + MHA** | **24 hours** | **4081.6186** | **Best model** |

### Key Insights
- **Attention mechanisms** provided the biggest performance jump
- **24-hour sequences** optimal for attention-based models
- **Bidirectional processing** consistently outperformed unidirectional
- **Hierarchical architecture** (64→32→16) enhanced feature extraction

## Visualizations

The project includes comprehensive data visualizations:

- **Time Series Analysis**: PM2.5 levels over time with missing value indicators
- **Trend Analysis**: 120-hour moving averages showing seasonal patterns  
- **Correlation Analysis**: Feature relationship heatmaps
- **Model Performance**: Training curves and prediction comparisons

## Technical Details

### Data Preprocessing
- **Missing Values**: Cubic spline interpolation + backward filling
- **Scaling**: MinMaxScaler normalization
- **Sequence Creation**: Sliding window approach for temporal patterns

### Model Features
- **Bidirectional LSTM**: Captures past and future temporal dependencies
- **Multi-Head Attention**: Focuses on relevant time steps and features
- **Progressive Architecture**: Hierarchical feature extraction at multiple scales
- **Regularization**: Prevents overfitting in complex models

## Requirements

- Python 3.7+
- TensorFlow 2.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Documentation

For detailed methodology, experimental design, and analysis, see the comprehensive [project report](report/report.md)