
from dataset.DataFrame import x_train, y_train, x_test,y_test, le

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV ,GridSearchCV, StratifiedKFold, cross_validate, learning_curve
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay
import pickle

with open("../clean_df.pkl", "rb") as f:
    df_clean = pickle.load(f)


#rf = RandomForestClassifier()

rf_params = {
    "n_estimators": [100, 200, 400, 800, 1200],
    "max_depth": [None, 10, 20, 30, 50],
    "min_samples_split": [2, 5, 8, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ['sqrt', 'log2', None],
    "bootstrap": [True, False]
}

cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

#rf_tuner=GridSearchCV(estimator=rf,param_distributions=rf_params, n_iter=30, cv=cv, n_jobs=-1, scoring='accuracy')
#rf_tuner.fit(x_train,y_train)

#best_params = svc_tuner.best_params_
#print("Best Parameters:", best_params)
#best parameters: {'n_estimators': 50, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 2}
#rf = rf_tuner.best_estimator_ #andrà sostituito con i best parameters
rf = RandomForestClassifier(n_estimators=50, max_depth= None, min_samples_split= 2, min_samples_leaf= 2)


scorings ={
    'accuracy':'accuracy',
    'precision':'precision_macro',
    'recall':'recall_macro',
    'f1_macro':'f1_macro',
    'f1_weighted':'f1_weighted',
    'precision_weighted':'precision_weighted',
    'recall_weighted':'recall_weighted',
}


cv_metrics_results = {}
for name, scoring in scorings.items():
    scores = cross_validate(rf, x_train, y_train, cv=cv, n_jobs=-1, scoring=scoring)
    mean = float(np.mean(scores['test_score']))
    std = float(np.std(scores['test_score'], ddof=0))
    var = float(np.var(scores['test_score'], ddof=0))
    cv_metrics_results[name] = {"scores": scores['test_score'].tolist(), "mean": mean, "std": std, "var": var}
    print(f"- {name}: mean={mean:.4f}, std={std:.4f}, var={var:.6f}")


#learning curves
train_sizes, train_scores, valid_scores = learning_curve(
    rf, x_train, y_train, cv=cv, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

#accuracy
plt.figure(figsize=(8,6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1)
plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1)

plt.plot(train_sizes, train_scores_mean,'o-' ,label='Training score', color= 'r')
plt.plot(train_sizes, valid_scores_mean,'o-', label='Validation score', color='g')

plt.title('Learning Curve (Random Forest)')
plt.xlabel('Training set size')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()


# Error Plot
plt.figure(figsize=(8,6))
plt.plot(train_sizes, 1 - train_scores_mean, label='Training error')
plt.plot(train_sizes, 1 - valid_scores_mean, label='Validation error')
plt.title('Error Curve')
plt.xlabel('Training set size')
plt.ylabel('Error')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

rf.fit(x_train,y_train)

#classification report
cls_report = classification_report(y_test, rf.predict(x_test), target_names=le.classes_)
print("\nClassification report (test):\n", cls_report)

#confusion matrix
cm = confusion_matrix(y_test, rf.predict(x_test))

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - (RandomForest)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

with open('../Saved_models/rm_model.pkl', 'wb') as file:
  pickle.dump(rf,file)