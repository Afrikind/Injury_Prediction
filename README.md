# Elite Sports Injury Prediction System

## Overview
An advanced injury prediction system that utilizes deep learning to assess and predict injury risks for athletes. The system integrates real-time smartwatch data and provides personalized recommendations for injury prevention.

## Features
- **Neural Network-based Risk Assessment**
  - Multi-layer neural network architecture
  - Real-time prediction capabilities
  - Dynamic risk factor analysis
  - Personalized recommendations

- **Smartwatch Integration**
  - Real-time heart rate monitoring
  - Continuous data streaming
  - Automatic data processing
  - Support for multiple device types

- **User Management**
  - Role-based access control
  - Secure authentication
  - User profile management
  - Activity tracking

- **Data Visualization**
  - Interactive dashboards
  - Real-time performance metrics
  - Risk factor analysis
  - Training history visualization

## Technical Architecture

### Neural Network Architecture
```
Input Layer (64 neurons)
    ↓
Batch Normalization
    ↓
Dropout (0.3)
    ↓
Hidden Layer 1 (32 neurons, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (0.2)
    ↓
Hidden Layer 2 (16 neurons, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (0.1)
    ↓
Output Layer (1 neuron, Sigmoid)
```

### Key Components
- **Data Processing**
  - StandardScaler for feature normalization
  - SimpleImputer for missing value handling
  - SMOTE for class balancing

- **Model Training**
  - Early stopping to prevent overfitting
  - Batch normalization for training stability
  - Dropout layers for regularization
  - Cross-validation for robust performance

- **Real-time Processing**
  - WebSocket integration for continuous data streaming
  - Real-time heart rate monitoring
  - Dynamic risk assessment
  - Immediate feedback generation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Afrikind/Injury_Prediction.git
cd Injury_Prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run Injury_Prediction_app.py
```

## Dependencies
- streamlit==1.31.1
- numpy==1.24.3
- pandas==2.0.3
- scikit-learn==1.3.0
- plotly==5.18.0
- imbalanced-learn==0.11.0
- websocket-client==1.6.4
- requests==2.31.0
- openpyxl==3.1.2
- matplotlib==3.7.1
- seaborn==0.12.2
- tensorflow==2.15.0
- markdown-it-py==3.0.0
- mdurl==0.1.2
- Pygments==2.19.1
- rich==14.0.0

## Usage

1. **Login**
   - Access the system using your credentials
   - Choose your role (athlete, coach, or admin)

2. **Dashboard**
   - View performance metrics
   - Monitor real-time data
   - Access analysis reports

3. **Risk Assessment**
   - Input athlete data
   - Connect smartwatch (optional)
   - View risk predictions
   - Get personalized recommendations

4. **Reports**
   - Generate analysis reports
   - View historical data
   - Export data for further analysis

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


