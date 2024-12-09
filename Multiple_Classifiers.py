# Импорт дополнительных библиотек
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score

# Шаг 6: Обучение и сравнение нескольких классификаторов
def evaluate_model(model, X_train, X_test, y_train, y_test, threshold=0):
    model.fit(X_train, y_train)
    
    # Предсказание вероятностей, если поддерживается, иначе предсказание меток напрямую
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
        y_preds = (y_probs >= threshold).astype(int)
    else:
        y_preds = model.predict(X_test)
    
    recall = recall_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds, zero_division=0)
    false_positives = sum((y_preds == 1) & (y_test == 0))
    
    return recall, precision, false_positives

# Определение классификаторов
classifiers = {
    "Logistic Regression": LogisticRegressionCV(cv=5, penalty="l2", max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

# Оценка каждого классификатора
results = []
for name, model in classifiers.items():
    recall, precision, false_positives = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
    results.append({
        "Model": name,
        "Recall": recall,
        "Precision": precision,
        "False Positives": false_positives
    })
    
# Отображение результатов
results_df = pd.DataFrame(results).sort_values(by="Recall", ascending=False)
print("\nРезультаты сравнения моделей:")
print(results_df)
# Display Results
results_df = pd.DataFrame(results).sort_values(by="Recall", ascending=False)
print("\nModel Comparison Results:")
print(results_df)
