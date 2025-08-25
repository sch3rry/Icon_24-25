from dataset.DataFrame import x_train, y_train, x_test, y_test, le
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, learning_curve
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay
import pickle

with open("../clean_df.pkl", "rb") as f:
    df_clean = pickle.load(f)


# Definizione modello base
#svc = SVC()

svc_params={
    'kernel':['rbf','linear', 'poly', 'sigmoid'],
    'C':[0.1, 2, 5, 10],
    'gamma':[0.001, 0.01, 0.1, 1.0],
    'degree':[2,3,4,5]

}


#Best Parameters: {'C': 5, 'degree': 2, 'gamma': 0.1, 'kernel': 'rbf'}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Hyperparameter tuning
#svc_tuner=RandomizedSearchCV(estimator=svc,param_distributions=svc_params,cv=cv,n_jobs=-1, scoring='accuracy', random_state=42)
#svc_tuner.fit(x_train, y_train)

#best_params = svc_tuner.best_params_
#svc = svc_tuner.best_estimator_
#print("Best Parameters:", best_params)

svc=SVC(C=5, kernel='rbf', gamma=0.1, degree=2)


# Cross-validation con metriche multiple
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
    scores = cross_validate(svc, x_train, y_train, cv=cv, n_jobs=-1, scoring=scoring)
    mean = float(np.mean(scores['test_score']))
    std = float(np.std(scores['test_score'], ddof=0))
    var = float(np.var(scores['test_score'], ddof=0))
    cv_metrics_results[name] = {"scores": scores['test_score'].tolist(), "mean": mean, "std": std, "var": var}
    print(f"- {name}: mean={mean:.4f}, std={std:.4f}, var={var:.6f}")

# learning curves
train_sizes, train_scores, valid_scores = learning_curve(
    svc, x_train, y_train, cv=cv, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

# accuracy
plt.figure(figsize=(8, 6))

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1)
plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1)

plt.plot(train_sizes, train_scores_mean,'o-' ,label='Training score', color= 'r')
plt.plot(train_sizes, valid_scores_mean,'o-', label='Validation score', color='g')

plt.title('Learning Curve (SVC)')
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

svc.fit(x_train, y_train)

#classification report
cls_report = classification_report(y_test, svc.predict(x_test), target_names=le.classes_)
print("\nClassification report (test):\n", cls_report)

#confusion matrix
cm = confusion_matrix(y_test, svc.predict(x_test))
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - SVC")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

with open('../Saved_models/svc_model.pkl', 'wb') as file:
  pickle.dump(svc,file)