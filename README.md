# potencial — pipeline de clasificación de potencial por vertical

Reemplaza a los seis notebooks (`classification_pot_*`) por un paquete con un módulo por
etapa, una config por vertical y dos notebooks orquestadores.

    configs/verticales.yaml    regla de decisión, umbrales, features, hiperparámetros LGBM,
                               modo de Feature mean y export — uno por vertical
    configs/diccionario_encuesta.csv

    src/config.py     carga y valida verticales.yaml
    src/data.py       la query de BigQuery, parametrizada por vertical (grupo de precio,
                      filtro de segmento) — verificada carácter por carácter contra los 6 notebooks
    src/features.py   coordenadas, scores de 1 km, cuadrante, distancia a Mass
    src/equifax.py    cruce con Equifax: urbanicidad, % de hogares, puntos de interés
    src/model.py      LightGBM por clase, predicción cruzada, SHAP (sin MLflow)
    src/decision.py   reglas umbral_2clases y acumulada (cada vertical conserva la suya)
    src/assemble.py   estimaciones + datos del cliente, con columnas limpias (sin _x/_y)
    src/export.py     Feature mean (modo features / shap) y escritura del Excel
    src/pm_mercado.py mercado_id (maestro + respaldo de clustering), agregación por mercado
                      y las filas de referencia que solo lleva el Excel de PM
    src/metrics_log.py registro histórico (CSV) de las métricas de validación de cada
                      corrida — el "modelo de referencia" para comparar corridas futuras
    src/pdm.py         paso final de PM: familias top compradas, métricas de venta y la
                      clasificación No Potencial/Low/Medium/High por cliente
    src/segmentacion_gold.py  tabla final de segmentación (transaccional + estratégica +
                      potencial) para la capa gold de BigQuery, en modo completo o
                      incremental, con MERGE (nunca duplica cod_interlocutor)
    src/salidas.py     encuentra el Excel más reciente de cada vertical por nombre de
                      archivo, para no escribirlos a mano en segmentacion_gold.ipynb

    notebooks/orquestador.ipynb     las 5 verticales de cliente (lavanderías, hoteles,
                                    panadería, gastro, bodegas)
    notebooks/orquestador_pm.ipynb  PM a nivel mercado (flujo distinto: agrega antes de
                                    modelar, y el Excel lleva filas de referencia)
    notebooks/correr_todo.ipynb     las 6 verticales (5 de cliente + PM) en una sola
                                    ejecución — PM ya incluye el paso PDM al final
    notebooks/pdm_final.ipynb       paso final de PM por separado (para correrlo solo,
                                    después de orquestador_pm.ipynb)
    notebooks/segmentacion_gold.ipynb  arma la tabla de segmentación gold (transaccional +
                                    estratégica + potencial), modo completo o incremental,
                                    y hace el MERGE — corre después de las 6 verticales

    tests/            133 tests: unitarios, contra los 6 Excel de referencia (2026-08-31) y
                      de equivalencia con el código original (tests/legacy/)

## Qué está validado y cómo

- **Decisión, ensamblado, export, Feature mean**: exacto contra los 6 Excel (157,873 filas).
- **Features geográficas y Equifax**: exacto contra los Excel; validado además con los
  parquets de clientes reales (score_comercial, score_competencia, cuadrante, que se
  normalizan sobre toda la base y no se pueden validar solo con el Excel).
- **Modelo**: idéntico a `potential_cls.py` con datos sintéticos y semilla fija (diff 0.0
  en probabilidades y SHAP).
- **`data.py`**: la query generada coincide con la de cada notebook, comparada sin espacios.
- **PM (`mercado_id`, agregación, filas de referencia)**: `mercado_id` exacto contra la hoja
  "IDs mercado" (100%); la agregación, con tolerancia (~98-99%, ver más abajo).
- **Los 5 orquestadores de cliente + PM corrieron de punta a punta** con datos reales
  (parquets de BigQuery del 2026-09-28 + equifax.csv + puntos comerciales + locales.csv),
  entrenando el modelo real y escribiendo el Excel final.

## Por qué no da 100% exacto contra la línea base en todo

Los Excel de referencia son del 2026-08-31. Los parquets de clientes que se usaron para
validar `features.py`, `equifax.py` y los smoke tests de punta a punta son del 2026-09-28:
la base de clientes ya cambió (clientes nuevos, otros que perdieron coordenadas válidas,
etc.), así que columnas que dependen de TODA la base (`score_comercial`,
`score_competencia`, la agregación por mercado de PM) coinciden entre 96% y 99.8%, no
100%. Lo que no depende de la base completa (decisión, Equifax por cliente, `mercado_id`)
sí da 100%.

## Hallazgos durante la migración (bugs del código anterior, corregidos acá)

1. `PROBA_COLUMNS` con la clave `'4 Bronce'` cuando la clase se llama `'2 Bronce'`: dejaba
   mal `proba_pred_segment` y `prediction_conf` en las filas Bronce de 4 de las 6 verticales.
2. `format_potential_df_2` no agregaba la fila de referencia de la clase alta a "Feature
   mean": salía vacía en 5 de las 6 verticales.
3. Columnas `X`, `Y`, `des_ciudad` y `des_segmento_estrategico` duplicadas en la hoja
   principal de lavandería y bodegas (`src/assemble.py` las deja una sola vez).
4. Hoteles no tiene grupo de precio propio: la query real usa `cod_grupo_precio_dex='LV'`
   (el de lavanderías) filtrado por `des_segmento_estrategica='Hotel/Casino'`.
5. Panadería no filtra por segmento estratégico; lo agrupa con un `CASE WHEN` en
   `'Estadio 1'`..`'Estadio 4'`/`'?'`.
6. `des_ciudad` del mercado, en PM, no sale de "DePara Zona ventas" como parecía en el
   notebook: ese `merge` cruza por columnas compartidas (incluida `des_ciudad`), así que en
   la práctica no reemplazaba nada. `pm_mercado.aggregate_mercado` usa directamente la
   `des_ciudad` propia de los clientes del mercado (validado 100% contra la línea base).
7. BigQuery devuelve las columnas `NUMERIC` (`pct_venta_diaadia_insuma`, `pct_descuento`,
   `val_score_urbanicidad`) como `Decimal`; LightGBM las rechaza. `data.fetch_clientes` las
   convierte a `float64`, y `model.py` lo hace también de forma defensiva.
8. Coordenadas de cliente invertidas (lat/lng al revés): entre 0.05% y 0.25% de los
   clientes según la vertical. `features.agregar_features_geo(coords=...)` deja elegir si
   `X`, `Y` y `DistanciaTiendaMass` usan las coordenadas crudas (igual que antes) o las
   corregidas.

## Tabla de segmentación gold (`segmentacion_gold.py`)

Reemplaza a `notebook.ipynb` (el que unía las 3 segmentaciones — transaccional, estratégica
y potencial — y cargaba a la capa gold). Dos cambios deliberados frente al original:

- **`MERGE` en vez de `INSERT INTO`**: el notebook original solo insertaba filas, sin
  revisar si el cliente ya existía — esa es la causa más probable de los duplicados en la
  tabla gold. El `MERGE` empareja por `cod_interlocutor` y nunca puede duplicar.
- **Sin el parche puntual** que el notebook original tenía para un cliente específico
  (`drop_duplicates` hardcodeado a un id): ya no hace falta, el `MERGE` lo hace innecesario.

Dos modos:
- **`completo`**: recalcula y reemplaza las 3 segmentaciones para todo el universo.
- **`incremental`**: agrega clientes nuevos y completa, en los que ya existen, la
  segmentación que les faltaba — nunca toca `cod_boca_salida` ni reemplaza un dato que ya
  existía, aunque el recálculo dé otro valor. `periodo`/`fec_proceso` se pisan en TODA la
  tabla en cada corrida (incluidos los clientes sin ningún dato nuevo), así que cada corrida
  incremental sube al `MERGE` la tabla completa, no solo la diferencia. La tabla solo crece
  entre corridas incrementales; una corrida `completo` es la única que puede "resetear" el
  universo de clientes.

En los dos modos, `periodo` y `fec_proceso` (el último día del mes de `periodo`) se
escriben en toda la tabla resultante, representando la fecha de la corrida más reciente.

## Métricas de cada corrida (modelo de referencia)

Cada vez que un notebook entrena un modelo, agrega sus métricas de validación (accuracy,
f1_macro, auc del modelo y de los tres baselines dummy) a
`data_potential/output/metricas_modelos.csv`, con fecha y vertical. Nunca borra corridas
anteriores, así que sirve como historial para comparar una corrida nueva contra las
pasadas (`src.metrics_log.ultima_referencia("bodegas")`, por ejemplo).

## Pendiente

- **Correr las 6 verticales contra BigQuery real** (acá solo se pudo con los parquets que
  se subieron, de una fecha fija) y comparar el Excel resultante contra la línea base con
  tolerancia, no solo con datos parciales.
- **PM: paso final (PDM)** — ya migrado (`src/pdm.py`, `notebooks/pdm_final.ipynb`). Reusa la
  `des_segmento_estrategico` del propio pipeline de PM (Confitero -> Impulso) en vez de la
  consulta aparte del notebook original. Encontrado y replicado a propósito: la clasificación
  de "Impulso" usa el umbral de distrito calculado con los clientes "Lavandero", no uno
  propio — así estaba en el notebook original.
- **`hoteles`, `panaderia`, `gastro`**: no tienen smoke test de punta a punta con modelo
  entrenado (sí `lavanderias`, `bodegas` y `pm_mercado`); deberían salir bien porque
  comparten el mismo código, pero no se corrieron.
- **Compatibilidad de entorno**: probado con Python 3.12, pandas 2.3.3/3.0.2, numpy 2.3.5,
  scikit-learn 1.8.0, lightgbm 4.6.0, shap 0.50.0, optuna 5.0.0.

## Tests

    pip install pytest pyyaml xlsxwriter openpyxl pandas numpy scikit-learn scipy lightgbm shap optuna
    pytest tests -q --baseline-dir "C:/ruta/a/los/excels" --insumos-dir "C:/ruta/a/los/insumos"

`--insumos-dir` (opcional, por defecto la misma carpeta que `--baseline-dir`) espera
`locales.csv`, `puntos_comerciales*.csv` (o `resultados_*.csv`), `equifax.csv`,
`urbanicidad.csv`, `Maestro*mercado*.xlsx` y `cluster_mercado_id*.csv`. Sin esas carpetas
corren solo los tests con datos sintéticos. La primera corrida con equifax.csv (~260 MB) y
la de bodegas (Excel >100 MB) tardan más; después usan caché en `.cache_tests/`.
