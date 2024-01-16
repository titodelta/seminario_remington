import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Configuración para reproducibilidad
np.random.seed(42)

# Generación de datos simulados
num_estudiantes = 500
horas_de_estudio = 20 * np.random.rand(num_estudiantes)
puntaje_real = 5 * horas_de_estudio / 20 + 0.5 * np.random.randn(num_estudiantes)

# Asegurarse de que los puntajes reales estén en el rango [0, 5]
puntaje_real = np.clip(puntaje_real, 0, 5)

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(horas_de_estudio, puntaje_real, test_size=0.2, random_state=42)

# Grados de polinomio a probar
grados = np.arange(1, 9)
errores_entrenamiento = []
errores_prueba = []

plt.figure(figsize=(15, 10))

for grado in grados:
    # Ajuste polinómico
    modelo = make_pipeline(PolynomialFeatures(grado), LinearRegression())
    modelo.fit(X_train.reshape(-1, 1), y_train)

    # Predicciones
    y_pred_train = modelo.predict(X_train.reshape(-1, 1))
    y_pred_test = modelo.predict(X_test.reshape(-1, 1))

    # Errores
    error_train = mean_absolute_error(y_train, y_pred_train)
    error_test = mean_absolute_error(y_test, y_pred_test)

    errores_entrenamiento.append(error_train)
    errores_prueba.append(error_test)

    # Subplot para cada grado de polinomio
    plt.subplot(3, 4, grado)
    plt.scatter(X_test, y_test, label='Datos reales (prueba)')
    plt.plot(X_test, y_pred_test, label=f'Ajuste polinómico (grado {grado})', color='red')
    plt.title(f'Grado {grado} - Error prueba: {error_test:.2f}')
    plt.xlabel('Horas de Estudio (prueba)')
    plt.ylabel('Puntaje Real')
    plt.legend()

# Mostrar el gráfico global
plt.tight_layout()
plt.show()


# Predecir calificaciones para 10 escenarios desde 0 hasta 20 horas de estudio
horas_estudio_escenarios = np.linspace(0, 20, 10).reshape(-1, 1)
calificaciones_escenarios = modelo.predict(horas_estudio_escenarios)

# Escala original: puntajes en el rango de 0 a 5
calificaciones_escenarios_original = np.clip(calificaciones_escenarios, 0, 5)

# Mostrar resultados
print("\nPredicciones para 10 escenarios desde 0 hasta 20 horas de estudio:")
for i, horas in enumerate(horas_estudio_escenarios.flatten()):
    print(f"Horas de estudio: {horas:.2f} | Calificación estimada: {calificaciones_escenarios_original[i]:.2f}")


# Encuentra el grado de polinomio con el menor error de prueba
mejor_grado = grados[np.argmin(errores_prueba)]
mejor_error = min(errores_prueba)

print(f"\nResumen:")
print(f"El mejor ajuste polinómico fue de grado {mejor_grado} con un error de prueba mínimo de {mejor_error:.2f}")
