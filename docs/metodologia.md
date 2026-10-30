# Modelo y tratamiento de los datos


## Hiperparámetros

| Parámetro | Valor | Procedencia |
|---|---:|---|
| `n_estimators` | 200 | Paper, IV.A |
| `max_depth` | 20 | Paper, IV.A |
| `min_samples_split` | 2 | Paper, IV.A |
| `min_samples_leaf` | 2 | Paper, IV.A |
| `max_features` | 0.7 | Paper, IV.A |
| `random_state` | 42 | Notebook, celda de optimización |
| `n_jobs` | 1 | Configuración de ejecución con menor paralelismo |

Se emplea `RandomForestRegressor` de scikit-learn. No se vuelve a ejecutar una búsqueda de hiperparámetros ni se cambian los valores del modelo para reducir su tamaño. Se mantienen los valores predeterminados de scikit-learn para los demás parámetros. El modelo se comprime al guardarlo y se reutiliza en memoria durante la inferencia.

## Separación de archivos

| Uso actual | Archivo | Inicio | Fin | Filas |
|---|---|---|---|---:|
| Entrenamiento | `df_operacion.csv` | 2026-06-01 00:00 | 2026-08-18 23:59 | 82 146 |
| Prueba | `df_operacion2.csv` | 2026-08-28 17:00 | 2026-09-06 22:00 | 13 261 |

Esta asignación coincide con la del paper: el modelo aprende con el periodo histórico y se evalúa en un periodo posterior. Los hiperparámetros publicados se seleccionaron utilizando TRAIN. TEST se mantiene fuera del ajuste de los árboles y de la validación interna.

La validación interna utiliza `TimeSeriesSplit(n_splits=4)` exclusivamente sobre TRAIN, ordenado por `DateTime`. No se mezclan los CSV y no se realiza una división aleatoria. Las medianas, los rangos de entrada y la importancia MDI se calculan únicamente con TRAIN. TEST se emplea para evaluar el modelo final y crear el histórico de predicciones.

Los CSV ya contienen registros `IN_SERVICE`. El notebook define este estado con una velocidad de turbina de al menos 5000 rpm, un caudal de aire mayor de 50 000 kg/h y la exclusión de los primeros 60 minutos tras un arranque real. El proyecto conserva el filtrado existente; no reconstruye los arranques con los archivos ya filtrados.

## Calidad y unidades

Se verifican las columnas, los valores numéricos finitos, la unicidad de fechas, la coherencia entre kg/h y t/h, el estado operativo y la ausencia de fechas compartidas entre TRAIN y TEST. Los CSV no se descargan, reemplazan ni modifican durante la preparación.

`df_operacion.csv` contiene un balance de vapor negativo: −3595.981567 kg/h el 26 de junio de 2026 a las 16:27. Se conserva para mantener el conjunto original y se registra en `negative_balance_rows` dentro de las métricas. No se interpreta como un consumo físico negativo ni se recorta la serie para mejorar los resultados.

La variable objetivo es `Turbine_Steam_Consumption_kg_h`. Se excluyen como predictores el objetivo en ambas unidades y las corrientes de vapor que intervienen directamente en su balance.

## Predictores

| Columna | Unidad |
|---|---|
| `Condenser_Absolute_Pressure_kg_cm2A` | kgf/cm² A |
| `Blower_Air_Flow_kg_h` | kg/h |
| `Blower_Discharge_Temperature_C` | °C |
| `Turbine_Speed_rpm` | rpm |
| `Turbine_Inlet_Steam_Pressure_kg_cm2` | kgf/cm² |
| `Turbine_Inlet_Steam_Temperature_C` | °C |
| `Seawater_Flow_m3_h` | m³/h |
| `Seawater_Inlet_Temperature_C` | °C |
| `Seawater_DeltaT_C` | °C |
| `Seawater_Inlet_Pressure_kg_cm2` | kgf/cm² |
| `Regenerator_Pressure_kg_cm2` | kgf/cm² |
| `Antisurge_Valve_Opening_pct` | % |
| `HS_Ejector_Steam_Pressure_kg_cm2` | kgf/cm² |
| `HS_Ejector_Steam_Temperature_C` | °C |

## Archivos generados

- `models/vapor_model.joblib`: bosque ajustado exclusivamente con TRAIN.
- `models/metrics.json`: métricas reales, parámetros, versiones, rangos, periodos y huellas SHA-256 de datos y artefactos.
- `models/test_predictions.csv`: consumo real y predicho en cada fecha de TEST.

La tabla II del paper publica MAE = 0.786 t/h, RMSE = 1.077 t/h y R² = 0.8710 para su asignación original. Estos valores se guardan como referencia en una sección distinta de las métricas calculadas. La validación de parámetros fijos de este proyecto no repite la búsqueda aleatoria de 20 combinaciones del estudio.

La importancia mostrada en la aplicación es MDI, calculada por el bosque. No se presenta como SHAP ni se interpreta como causalidad. El notebook original contiene el análisis SHAP del estudio, que no se ejecuta durante el inicio de la aplicación.

## Dependencias y despliegue

Se utiliza Python 3.12 y se fija la versión de scikit-learn para cargar el modelo con la misma versión empleada al entrenarlo. `uv.lock` fija el entorno de desarrollo y CI. El archivo `app/requirements.txt` remite a las dependencias fijadas en la raíz y es el punto de entrada de dependencias para Streamlit Community Cloud.

CI guarda el modelo y las predicciones del mismo entrenamiento que pasa las pruebas. CD recupera ese artefacto, vinculado a la ejecución concreta de CI, y publica una instantánea del código correspondiente en `deploy` mediante un commit normal. No entrena en el servidor web ni utiliza hooks de Render.

Documentación de ejecución y despliegue:

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource
- https://github.com/actions/upload-artifact
- https://github.com/actions/download-artifact
