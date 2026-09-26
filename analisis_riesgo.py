import os
import sys

import matplotlib
matplotlib.use("Agg")  # guarda los gráficos como imagen en vez de abrir ventanas
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AZUL, NARANJA = "#2a78d6", "#eb6834"   # se queda / se va
os.makedirs("docs", exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})

# 1. CARGA DE DATOS
try:
    df = pd.read_csv("customer_churn.csv")
    print("Archivo cargado exitosamente.")
except FileNotFoundError:
    print("Error: No se encontró 'customer_churn.csv'")
    sys.exit(1)

# 2. LIMPIEZA
# Eliminamos la columna de índice innecesaria
if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

features = df.drop("Churn", axis=1).copy()
target_variable = df["Churn"].copy()

# ContractRenewal viene como Yes/No; drop_first deja una sola columna (ContractRenewal_Yes)
features = pd.get_dummies(features, columns=["ContractRenewal"], dtype=int, drop_first=True)

print(f"Clientes: {len(df)} | Se fueron: {target_variable.sum()} ({target_variable.mean():.1%})")

# 3. ANÁLISIS EXPLORATORIO (gráficos en docs/)
print("Generando visualizaciones...")

# Renovación de contrato vs fuga (en %, para comparar grupos de distinto tamaño)
tasa = pd.crosstab(df["ContractRenewal"], df["Churn"], normalize="index") * 100
ax = tasa.plot(kind="bar", stacked=True, color=[AZUL, NARANJA], width=0.6, edgecolor="white", figsize=(6, 4))
ax.set_title("Renovación de contrato vs. fuga")
ax.set_xlabel("¿Renovó contrato?")
ax.set_ylabel("% de clientes")
ax.set_xticklabels(["No", "Sí"], rotation=0)
ax.legend(["Se queda", "Se va"], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
for i, v in enumerate(tasa[1]):
    ax.text(i, 100 - v / 2, f"{v:.0f}%", ha="center", va="center", color="white", fontweight="bold")
plt.tight_layout()
plt.savefig("docs/renovacion_vs_fuga.png")
plt.close()

# Llamadas a servicio al cliente vs fuga
llamadas = df.assign(Llamadas=df["CustServCalls"].clip(upper=5)).groupby("Llamadas")["Churn"].mean() * 100
ax = llamadas.plot(kind="bar", color=NARANJA, width=0.6, figsize=(6, 4))
ax.set_title("Tasa de fuga según llamadas a servicio al cliente")
ax.set_xlabel("Llamadas a servicio al cliente")
ax.set_ylabel("% que se fue")
ax.set_xticklabels([str(i) if i < 5 else "5+" for i in llamadas.index], rotation=0)
plt.tight_layout()
plt.savefig("docs/llamadas_vs_fuga.png")
plt.close()

# Distribución de antigüedad
plt.figure(figsize=(6, 4))
plt.hist([df[df["Churn"] == 0]["AccountWeeks"], df[df["Churn"] == 1]["AccountWeeks"]],
         label=["Se queda", "Se va"], bins=20, color=[AZUL, NARANJA])
plt.title("Antigüedad de la cuenta según estado")
plt.xlabel("Semanas como cliente")
plt.ylabel("Clientes")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("docs/antiguedad.png")
plt.close()

# 4. MACHINE LEARNING
# stratify mantiene la misma proporción de fugas en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    features, target_variable, test_size=0.3, random_state=42, stratify=target_variable
)

modelos = {
    # El modelo original del curso
    "Original": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    # Mismo modelo, pero le da más peso a los clientes que se van (son solo el 14.5%)
    "Balanceado": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
}

# 5. EVALUACIÓN DE RESULTADOS
baseline = 1 - y_test.mean()
print(f"\nBaseline (decir que nadie se va): accuracy {baseline:.3f}")

filas = []
fig_cm, axes = plt.subplots(1, 2, figsize=(9, 4))
fig_roc, ax_roc = plt.subplots(figsize=(5, 5))

for (nombre, modelo), ax_cm, color in zip(modelos.items(), axes, [AZUL, NARANJA]):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    filas.append({
        "Modelo": nombre,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    })

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Se queda", "Se va"],
                                            cmap="Blues", colorbar=False, ax=ax_cm)
    ax_cm.set_title(nombre)
    ax_cm.set_xlabel("Predicción")
    ax_cm.set_ylabel("Real")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax_roc.plot(fpr, tpr, color=color, linewidth=2, label=f"{nombre} (AUC {filas[-1]['ROC-AUC']:.2f})")

fig_cm.suptitle("Matriz de confusión (datos de prueba)")
fig_cm.tight_layout()
fig_cm.savefig("docs/matriz_confusion.png")

ax_roc.plot([0, 1], [0, 1], "--", color="#9a9a9a", label="Al azar")
ax_roc.set_title("Curva ROC")
ax_roc.set_xlabel("Falsos positivos (tasa)")
ax_roc.set_ylabel("Verdaderos positivos (tasa)")
ax_roc.legend(frameon=False)
fig_roc.tight_layout()
fig_roc.savefig("docs/curva_roc.png")
plt.close("all")

print("\n--- Métricas del Modelo de Riesgo ---")
print(pd.DataFrame(filas).set_index("Modelo").round(3).to_string())

# ¿Qué variables pesan más? (coeficientes del modelo balanceado, con datos estandarizados)
coef = pd.Series(modelos["Balanceado"][-1].coef_[0], index=features.columns).sort_values()
nombres = {
    "CustServCalls": "Llamadas a servicio al cliente", "DayMins": "Minutos de día",
    "MonthlyCharge": "Cargo mensual", "OverageFee": "Cargo por exceso", "RoamMins": "Minutos en roaming",
    "AccountWeeks": "Antigüedad", "DayCalls": "Llamadas de día", "DataUsage": "Uso de datos",
    "ContractRenewal_Yes": "Renovó contrato", "DataPlan": "Tiene plan de datos",
}
plt.figure(figsize=(7, 4.5))
plt.barh([nombres.get(c, c) for c in coef.index], coef.values,
         color=[NARANJA if v > 0 else AZUL for v in coef.values], height=0.6)
plt.axvline(0, color="#9a9a9a", linewidth=1)
plt.title("Qué variables pesan más en el modelo")
plt.text(0.99, 0.02, "naranja: sube el riesgo · azul: lo baja", transform=plt.gca().transAxes, ha="right", color="#52514e", fontsize=9)
plt.xlabel("Peso en el modelo")
plt.tight_layout()
plt.savefig("docs/variables.png")
plt.close()

print("\nGráficos guardados en la carpeta docs/")
