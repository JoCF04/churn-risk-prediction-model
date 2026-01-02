import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1 CARGA DE DATOS
try:
    df = pd.read_csv("customer_churn.csv")
    print("Archivo cargado exitosamente.")
except FileNotFoundError:
    print("Error: No se encontró 'customer_churn.csv'")
    exit()

# 2 LIMPIEZA 
# Eliminamos la columna de índice innecesaria
if 'Unnamed: 0' in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

features = df.drop("Churn", axis=1).copy()
target_variable = df["Churn"].copy()

features = pd.get_dummies(features, columns=["ContractRenewal"], dtype=int)

# 3. ANÁLISIS EXPLORATORIO (Gráficos)
print("Generando visualizaciones...")

# Gráfico de barras: Renovación de contrato vs Fuga
churn_counts = pd.crosstab(df["ContractRenewal"], df["Churn"])
churn_counts.plot(kind='bar', stacked=True)
plt.title('Contract Renewal vs. Churn')
plt.xlabel('Contract Renewal')
plt.ylabel('Count')
plt.show() 

# Histograma: Distribución de antigüedad
plt.hist([df[df["Churn"] == 1]["AccountWeeks"], df[df["Churn"] == 0]["AccountWeeks"]], 
         label=["Churned", "Non-Churned"], bins=20)
plt.title('Tenure Distribution by Churn Status')
plt.xlabel('Account Weeks')
plt.ylabel('Frequency')
plt.legend()
plt.show()

#MACHINE LEARNING
X_train, X_test, y_train, y_test = train_test_split(
    features, target_variable, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. EVALUACIÓN DE RESULTADOS
y_pred = model.predict(X_test)

print("\n--- Métricas del Modelo de Riesgo ---")
print(f"Accuracy (Exactitud): {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")

# por si no funciona, colocar en terminal "pip install pandas matplotlib scikit-learn"