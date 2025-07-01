# Predicting-Chess-Game-Outcomes-with-Machine-Learning 

## Project Overview  
In this project, I investigate and predict the outcomes of chess games using machine learning techniques. Drawing from over 168,000 games on Lichess.org  the project applies extensive Exploratory Data Analysis (EDA), classification, and regression modeling to uncover patterns and build predictive models.

**Key Goals:**  
- Understand how player Elo, time formats, and openings affect outcomes  
- Predict game results using machine learning (classification & regression)  
- Evaluate feature importance and model performance across scenarios  
- Build benchmark dummy models based on theoretical expectations  

**Tools Used:** Python (Pandas, Scikit-learn, TensorFlow, NumPy, Matplotlib), Jupyter, Power Query

---

## Files & Code Structure  
Structure of the project: 
1. Exploratory Data Analysis   
2. Classification Models 
3. Regression Models
4. Sample Predictions

- 📄 Full report (PDF): **[Final_05.05.2025.pdf](https://github.com/DimKaisaris/Predicting-Chess-Game-Outcomes-with-Machine-Learning/blob/main/images/Final_05.05.2025.pdf)**   
- 📁 Jupyter notebooks: **[here](https://github.com/DimKaisaris/Predicting-Chess-Game-Outcomes-with-Machine-Learning/tree/main/python%20code)**  
  
---
## Exploratory Data Analysis  
Using pandas, NumPy, and Matplotlib, I performed in-depth EDA to explore patterns and relationships in the dataset. Some key discoveries:
1. **Outcome Distribution:** White wins ~49%, Black ~45%, Draw ~6%  
2. **Draw Rate Anomalies:** High draw rate below 1000 Elo , possibly due to stalemates and repetition  
3. **Opening Insights:** The Scandinavian Defense surprisingly outperforms for Black under 2000 Elo  
4. **Blitz Surprise:** Black outperforms White in Blitz in lower Elo ranges 

---

## Classification Models  
I built preprocessing pipelines and trained several models using Scikit-learn, tuning them via GridSearchCV and RandomizedSearchCV. Below are the best performers:

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
I reframed the problem as a regression task by modeling the score as:
1 = White win, 0.5 = Draw, 0 = Black win.

Alongside standard regressors, I implemented a custom model inspired by the Elo expected outcome formula — and it performed admirably!

| Model                     | MSE     |
|---------------------------|---------|
| StackingRegressor         | 0.2261  |
| RandomForestRegressor     | 0.2263  |
| GradientBoostingRegressor | 0.2264  |
| Best_b_Regressor (custom) | 0.2269  |
| Dummy (mean) Regressor    | 0.235   |

The Best_b_Regressor cleverly tuned the Elo formula to boost predictive power — a tiny tweak, a nice payoff.

---

## Sample Predictions 
In the final part, I reflected on the outcomes and tested predictions using .predict_proba() from the best classifier:

```python
Input: WhiteElo=2066 | BlackElo=2108 | Opening=Queen's Gambit | Format=Bullet
Output: Black Win: 50.5%, Draw: 4.1%, White Win: 45.5%
```
These probability-based predictions provide valuable insights — especially for tight matchups.
Overall, this was a fun and successful machine learning journey through the world of chess!
---


## Author  
**Dimitris Kaisaris**    
