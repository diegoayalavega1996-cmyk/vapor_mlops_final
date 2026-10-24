# Comprobaciones de ejecución

El proyecto se verificó con Python 3.12.14 y el entorno instalado mediante `uv sync --locked`.

| Comprobación | Resultado |
|---|---|
| Instalación de dependencias y paquete | Correcta |
| Comando `preparar-datos` | Ambos CSV validados, sin sobrescritura |
| Comando `entrenar` | Cuatro divisiones temporales y modelo final generados |
| Pruebas de datos, modelo, API y Streamlit | 24 aprobadas |
| Formulario de Streamlit | 14 entradas y predicción en ambas unidades |
| Inicio del servidor Streamlit | Respuesta HTTP 200 en la página y el estado de salud |
| Dependencias de `app/requirements.txt` | Resolución correcta con Python 3.12 |
| Sintaxis de CI/CD | YAML y scripts de shell válidos |
| Simulación del script de CD | Primera publicación y actualización posterior correctas en un remoto Git local |
| Construcción de Docker | Pendiente: Docker no disponible en el entorno de comprobación |
| Publicación en GitHub y Streamlit Community Cloud | Pendiente de conectar el repositorio y la aplicación |

La simulación de CD verificó que la rama `deploy` contiene el código validado, los CSV y los tres artefactos del entrenamiento, sin volver a ejecutar los workflows desde esa rama. No sustituye una ejecución autenticada en GitHub Actions.

## Resultados con la separación cronológica del paper

| Métrica | Resultado |
|---|---:|
| Filas de TRAIN (`df_operacion.csv`) | 82 146 |
| Filas de TEST (`df_operacion2.csv`) | 13 261 |
| MAE de validación temporal, promedio (t/h) | 0.5485 |
| MAE TEST (t/h) | 0.7803 |
| RMSE TEST (t/h) | 1.0725 |
| R² TEST | 0.8722 |

El modelo se ajustó con los registros del 1 de junio al 18 de agosto de 2026 y se evaluó en el periodo posterior, del 28 de agosto al 6 de septiembre de 2026. La validación temporal se realizó exclusivamente dentro de TRAIN.

Starlette emite un aviso de obsolescencia sobre el cliente de pruebas `httpx`. Las consultas funcionan con la versión fijada y las 24 pruebas terminan correctamente.
