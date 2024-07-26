
<details> <summary>Phase 1</summary>

- [X]  Linear Regression Model
    - [x]  Implement
    - [X]  Write DocStrings
- [X]  Logistic Regression Model
    - [X]  Implement
    - [X]  Write DocStrings
- [X]  Vanilla Neural Networks
    - [X]  Implement
    - [X]  Write DocStrings
</details>

<details> <summary> Phase 2</summary>


## 1. Validate or add Models
- [X] Linear Regression
  - Validate that it has similar accuracy to sklearn
- [X] Logistic Regression
  - Validate that it has similar accuracy to sklearn
- [X] Neural Network
  - Attempt to work with different datasets (CIFAR, MNIST variations)
- [X] K Nearest Neighbors
- [X] Support Vector Machines
- [X] Decision Trees
- [X] Refactor Examples once pre-built models are adjusted
- [ ] Majority Voting Implementation
  - [ ] Make every model compatible with the dimensions (samples, features)
    - [X] Linear Regression
  - [ ] Hard Voting
  - [ ] Soft Voting
  - [ ] Implementing / Computing total error via binomial distribution (irreducible or overall error? I'm thinking overall.)
  - [ ] Validate for all models.
- [ ] Bagging Implementation
  - [ ] Utility function (preprocessing) to draw samples from a uniform distribution.
- [ ] Random Forest Implementation
- [ ] Extra Random Forest Implementation
- [ ] Custom Neural Networks



## 2. Add Functionality for Custom Models
### 2.1 Initialization
- [ ] Add functionality for Xavier / He Initialization

### 2.2 Layers
- [ ] Add functionality for regular feed-forward layers
- [ ] Add functionality for Dropout layers
- [ ] Add functionality for BatchNorm layers

### 2.3 Regularization
- [ ] Add functionality for L1 Regularization
- [ ] Add functionality for L2 Regularization

### 2.4 Activation Functions
- [ ] Add functionality for different Activation Functions

### 2.5 Loss Functions & Metrics
- [ ] Add functionality for MSE
- [ ] Add functionality for MAE
- [ ] Add functionality for BCE
- [ ] Add functionality for CCE
- [ ] Add functionality for Smoothed CE

### 2.6 Optimizers
- [ ] Add functionality for Gradient Descent
- [ ] Add functionality for Momentum
- [ ] Add functionality for Nesterov Momentum
- [ ] Add functionality for RMSprop
- [ ] Add functionality for Adam
- [ ] Add functionality for AdaMax
- [ ] Add functionality for Nadam
- [ ] Add functionality for NadaMax

### 2.7 Learning Rate Scheduling
- [ ] Add functionality for Exponential Decay
- [ ] Add functionality for Halving
- [ ] Add functionality for Cyclical Learning Rate

## 3. Add Utilities
- [ ] Add MinMax Normalization
- [ ] Add Standardization (z-score)
- [ ] Add One-hot Encoding
- [ ] Add Mini-batching Data
- [ ] Saving Model Params
- [ ] Loading Bar While model is training ( like pytorch )
- [X] IO - CSV to Numpy
- [X] train_test_split
- [X] x_y_split

**MISC**

- A Logger? For Trainign Runs?

****
</details>