-- Validación incremental de la segmentación ESTRATÉGICA: agosto 2026 vs julio 2026.
-- Para otro mes, cambia solo las dos fechas de los CTE `anterior` y `actual`.
--
-- Reglas del modo incremental que valida:
--   * solo cambia el segmento de quien estaba "Por definir" (el que ya tenía segmento no cambia)
--   * los clientes nuevos se agregan; ningún cliente sale; la boca de salida no cambia
-- `cambio_esperado = FALSE` marca lo que rompe esas reglas y sale primero en el resultado.
WITH anterior AS (
  SELECT cod_interlocutor, cod_boca_salida, des_segmento_estrategica
  FROM `acpe-prod-dlk-shrd-gld.delivery_cliente_gtm.cr_segmentacion`
  WHERE periodo = DATE '2026-07-01'
),
actual AS (
  SELECT cod_interlocutor, cod_boca_salida, des_segmento_estrategica
  FROM `acpe-prod-dlk-shrd-gld.delivery_cliente_gtm.cr_segmentacion`
  WHERE periodo = DATE '2026-08-01'
),
cruce AS (
  SELECT
    COALESCE(a.cod_interlocutor, b.cod_interlocutor) AS cod_interlocutor,
    a.cod_boca_salida                                AS boca_anterior,
    b.cod_boca_salida                                AS boca_actual,
    a.des_segmento_estrategica                       AS seg_estrategica_anterior,
    b.des_segmento_estrategica                       AS seg_estrategica_actual,
    a.cod_interlocutor IS NULL                       AS es_nuevo,
    b.cod_interlocutor IS NULL                       AS salio,
    -- "sin segmento" = NULL, vacío, "Por definir" o "Sin asignar"
    IFNULL(LOWER(TRIM(a.des_segmento_estrategica)), '') IN ('', 'por definir', 'sin asignar') AS ant_por_definir,
    IFNULL(LOWER(TRIM(b.des_segmento_estrategica)), '') IN ('', 'por definir', 'sin asignar') AS act_por_definir
  FROM anterior a
  FULL OUTER JOIN actual b ON a.cod_interlocutor = b.cod_interlocutor
)
SELECT
  cod_interlocutor,
  boca_anterior,
  boca_actual,
  seg_estrategica_anterior,
  seg_estrategica_actual,
  CASE
    WHEN es_nuevo THEN 'CLIENTE NUEVO'
    WHEN salio THEN 'CLIENTE SALIÓ'                                   -- no debería ocurrir
    WHEN boca_anterior IS DISTINCT FROM boca_actual THEN 'CAMBIO DE BOCA'  -- no debería ocurrir
    WHEN ant_por_definir AND NOT act_por_definir THEN 'POR DEFINIR A SEGMENTO'
    WHEN NOT ant_por_definir AND act_por_definir THEN 'REGRESÓ A POR DEFINIR'  -- no debería ocurrir
    WHEN ant_por_definir AND act_por_definir THEN 'CAMBIO ENTRE VACÍOS'
    ELSE 'CAMBIO DE SEGMENTO'                                         -- no debería ocurrir
  END AS tipo_cambio,
  (es_nuevo
   OR (NOT salio
       AND boca_anterior IS NOT DISTINCT FROM boca_actual
       AND ant_por_definir AND NOT act_por_definir)) AS cambio_esperado
FROM cruce
WHERE
  es_nuevo
  OR salio
  OR boca_anterior IS DISTINCT FROM boca_actual
  OR seg_estrategica_anterior IS DISTINCT FROM seg_estrategica_actual
ORDER BY cambio_esperado, tipo_cambio, cod_interlocutor
