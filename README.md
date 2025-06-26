# Predicting-Chess-Game-Outcomes-with-Machine-Learning

## Project Overview  
This project investigates and predicts the outcomes of chess games using machine learning techniques. Drawing from over 168,000 games on Lichess.org  the project applies extensive Exploratory Data Analysis (EDA), classification, and regression modeling to uncover patterns and build predictive models.

**Key Goals:**  
- Understand how player Elo, time formats, and openings affect outcomes  
- Predict game results using machine learning (classification & regression)  
- Evaluate feature importance and model performance across scenarios  
- Build benchmark dummy models based on theoretical expectations  

**Tools Used:** Python (Pandas, Scikit-learn, Seaborn, Matplotlib), Jupyter, Power Query

---

## Dataset Details  
Games were scraped from Lichess and included top-level players like Magnus Carlsen, with features such as:
- `Result` (White win, Black win, Draw)
- `WhiteElo`, `BlackElo`, and Elo difference
- `Time_format` (bullet, blitz, rapid, classical)
- `Opening_name` (grouped into 50 logical classes)
- `Termination` type (normal, timeout, abandoned)
- `Increment_binary` and custom columns like `Elo_Dif_Range` and `Score`

---

## Exploratory Data Analysis  
The full EDA was conducted in Jupyter using Matplotlib and Seaborn. Key findings:

1. **Outcome Distribution:** White wins ~49%, Black ~45%, Draw ~6%  
2. **Draw Rate Anomalies:** Highest draw rate below 1000 Elo (!), possibly due to stalemates and repetition  
3. **Opening Insights:** The Scandinavian Defense surprisingly outperforms for Black under 2000 Elo  
4. **Blitz Surprise:** Black outperforms White in Blitz in lower Elo ranges  
5. **Elo Gap Correlation:** Accuracy improves with wider Elo difference (up to 71% at 200+ Elo gap)

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

## Hypothesis Testing  
Chi-square tests were performed to validate statistically significant patterns. Example:
- Players with a 400+ Elo edge lose significantly more often in Blitz vs. other formats (p < 0.0001)
- Openings like the Sicilian did **not** show statistically higher loss rates

---

## Files & Code Structure  
- 📄 Full PDF report: **[Final_05.05.2025.pdf](./Final_05.05.2025.pdf)**  
- 📁 Jupyter notebooks: Classification, Regression, EDA, Custom Models  
- 📊 Data file: `data_ML.csv` (cleaned dataset)  
- 📘 Prediction results and dummy models included in `Part2_class_...ipynb` notebooks  

---

## Author  
**Dimitris Kaisaris**  
- Portfolio: [dimkaisaris.github.io/portfolio](https://dimkaisaris.github.io/portfolio)  
- Data science & chess enthusiast  
