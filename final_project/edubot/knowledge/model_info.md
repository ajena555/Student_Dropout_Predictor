## Machine Learning Model

We use a Random Forest Classifier trained on student academic and behavioral data.

### Key Features:
- Grades (G1, G2, G3)
- Number of absences
- Study time per week
- Parental education levels
- Alcohol consumption on weekdays/weekends
- Participation in extra-curricular activities

The model was trained on a dataset balanced using SMOTE to handle class imbalance. We used feature selection to keep only the most relevant predictors, and the model achieved high precision and recall on the test set.
