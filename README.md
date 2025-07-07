# Human Activity Recognition using Smartphones 📱🤸

This project performs classification of human activities (e.g., walking, sitting, standing) based on smartphone sensor data using machine learning models. The dataset consists of preprocessed accelerometer and gyroscope readings from 30 participants.


## 📦 Dataset

- **Source**: [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- **Samples**: 10,299
- **Features**: 561 (time & frequency domain)
- **Activities**:
  - WALKING
  - WALKING_UPSTAIRS
  - WALKING_DOWNSTAIRS
  - SITTING
  - STANDING
  - LAYING


## 🔍 Key Phases

### Data Understanding & EDA
- Class and subject distribution
- Feature correlation, PCA visualization

### Preprocessing
- Label mapping
- Feature scaling (StandardScaler)
- SelectKBest feature selection
- PCA for dimensionality reduction (optional)

### Modeling
- Trained: `RandomForest`, `SVM (RBF)`, `KNN`, `MLP`
- Evaluated using accuracy and classification reports

### Hyperparameter Tuning
- `RandomizedSearchCV` with `StratifiedKFold` for:
  - RandomForest
  - MLPClassifier
  - SVM (RBF)

### Ensemble Techniques
- Soft Voting
- Stacking (meta-model: LogisticRegression)
- Blending (weighted average)

### Error Analysis
- Confusion matrix
- Misclassified samples
- Most confused activity pairs

### Final Output
- Accuracy: ~95–97% (ensemble models)
- Inference-ready prediction function using saved model, scaler, and selector
