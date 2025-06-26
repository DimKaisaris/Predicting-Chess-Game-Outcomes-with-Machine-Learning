# Predicting-Chess-Game-Outcomes-with-Machine-Learning

## Project Overview  
In this project I investigate and predict the outcomes of chess games using machine learning techniques. Drawing from over 168,000 games on Lichess.org  the project applies extensive Exploratory Data Analysis (EDA), classification, and regression modeling to uncover patterns and build predictive models.

**Key Goals:**  
- Understand how player Elo, time formats, and openings affect outcomes  
- Predict game results using machine learning (classification & regression)  
- Evaluate feature importance and model performance across scenarios  
- Build benchmark dummy models based on theoretical expectations  

**Tools Used:** Python (Pandas, Scikit-learn, TensorFlow, Numpy, Matplotlib), Jupyter, Power Query

---

## Files & Code Structure  
Structure of the project: 
1. Exploratory Data Analysis   
2. Classification Models 
3. Regression Models
4. Sample Predictions

- 📄 Full PDF report: **[pdf](https://github.com/DimKaisaris/Predicting-Chess-Game-Outcomes-with-Machine-Learning/blob/main/images/Final_05.05.2025.pdf)**   
- 📁 Jupyter notebooks: **[here}()
- 📊 Data file: `data_ML.csv` (cleaned dataset)  
- 📘 Prediction results and dummy models included in `Part2_class_...ipynb` notebooks  
---

## Exploratory Data Analysis  
The full EDA was conducted in Jupyter using Matplotlib and Seaborn. Key findings:

1. **Outcome Distribution:** White wins ~49%, Black ~45%, Draw ~6%  
2. **Draw Rate Anomalies:** High draw rate below 1000 Elo (!), possibly due to stalemates and repetition  
3. **Opening Insights:** The Scandinavian Defense surprisingly outperforms for Black under 2000 Elo  
4. **Blitz Surprise:** Black outperforms White in Blitz in lower Elo ranges  
---

## Classification Models  
The following classification models were trained using Scikit-learn:

| Model                  | Accuracy |
|------------------------|----------|
| Random Forest          | 0.548    |
| AdaBoostClassifier     | 0.548    |
| Deep Neural Network    | 0.544    |
| Gradient Boosting      | 0.543    |
| SVM (RBF Kernel)       | 0.542    |
| Voting Classifier      | 0.541    |
| Logistic Regression    | 0.537    |
| Dummy Baseline         | 0.491    |

The highest-performing classifiers (Random Forest, AdaBoost) outperformed dummy models and showed Elo difference as the most important feature.

---

## Regression Models  
Framing the problem as a regression (Score = 1 for white win, 0.5 for draw, 0 for black win), we applied:

| Model                     | MSE     |
|---------------------------|---------|
| StackingRegressor         | 0.2261  |
| RandomForestRegressor     | 0.2263  |
| GradientBoostingRegressor | 0.2264  |
| Best_b_Regressor (custom) | 0.2269  |
| Dummy (mean) Regressor    | 0.235   |

The custom `Best_b_Regressor`, based on tuning the Elo expected outcome formula, ranked among the top 5.

---

## Sample Predictions  
We used `.predict_proba` from our best classifier to generate probabilistic outputs:

```python
Input: WhiteElo=2066 | BlackElo=2108 | Opening=Queen's Gambit | Format=Bullet
Output: Black Win: 50.5%, Draw: 4.1%, White Win: 45.5%
```

---


## Author  
**Dimitris Kaisaris**  
- Portfolio: [dimkaisaris.github.io/portfolio](https://dimkaisaris.github.io/portfolio)  
- Data science & chess enthusiast  
