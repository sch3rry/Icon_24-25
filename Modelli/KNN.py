from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold, learning_curve
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



from dataset.DataFrame import x_train, y_train, x_test, y_test, le

with open("../clean_df.pkl", "rb") as f:
    df_clean = pickle.load(f)


Knn_params = {
    'n_neighbors': list(range(1, 11)),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [16, 32, 64],
    "metric": ["euclidean", "manhattan", "minkowski"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


knn = KNeighborsClassifier(n_neighbors=6, weights='distance', algorithm='auto', leaf_size=16, metric='manhattan')
knn.fit(x_train, y_train)
#grid_search = GridSearchCV(knn, Knn_params, cv=cv, n_jobs=-1)
#grid_search.fit(x_train,y_train)
#print(grid_search.best_params_)

#knn = KNeighborsClassifier(best parameters)

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
    scores = cross_validate(knn, x_train, y_train, cv=cv, n_jobs=-1, scoring=scoring)
    mean = float(np.mean(scores['test_score']))
    std = float(np.std(scores['test_score'], ddof=0))
    var = float(np.var(scores['test_score'], ddof=0))
    cv_metrics_results[name] = {"scores": scores['test_score'].tolist(), "mean": mean, "std": std, "var": var}
    print(f"- {name}: mean={mean:.4f}, std={std:.4f}, var={var:.6f}")

# learning curves
train_sizes, train_scores, valid_scores = learning_curve(
    knn, x_train, y_train, cv=cv, scoring='accuracy',
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

plt.title('Learning Curve (KNN)')
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

#classification report
cls_report = classification_report(y_test, knn.predict(x_test), target_names=le.classes_)
print("\nClassification report (test):\n", cls_report)

#confusion matrix
cm = confusion_matrix(y_test, knn.predict(x_test))
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - KNN")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()


with open("../Saved_models/knn_model.pkl", "wb") as f:
        pickle.dump(knn,f)