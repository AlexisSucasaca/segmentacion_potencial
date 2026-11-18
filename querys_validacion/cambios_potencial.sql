-- Validación incremental del POTENCIAL (flg_potential + des_potencial_predic): agosto 2026 vs julio 2026.
-- Para otro mes, cambia solo las dos fechas de los CTE `anterior` y `actual`.
--
-- Reglas del modo incremental que valida:
--   * el potencial solo cambia para quien estaba "Por definir" y ahora tiene segmento
--     (antes: transaccional o estratégica sin segmento)
--   * quien ya tenía segmento conserva su potencial, incluido NP
--   * un flg_potential vacío (NULL) es NP: se muestra como NP y no cuenta como cambio frente a FALSE
--   * los clientes nuevos se agregan; ningún cliente sale; la boca de salida no cambia
-- `cambio_esperado = FALSE` marca lo que rompe esas reglas y sale primero en el resultado.
WITH anterior AS (
  SELECT cod_interlocutor, cod_boca_salida, des_segmento_transaccional,
         des_segmento_estrategica, flg_potential, des_potencial_predic
  FROM `acpe-prod-dlk-shrd-gld.delivery_cliente_gtm.cr_segmentacion`
  WHERE periodo = DATE '2026-07-01'
),
actual AS (
  SELECT cod_interlocutor, cod_boca_salida, des_segmento_transaccional,
         des_segmento_estrategica, flg_potential, des_potencial_predic
  FROM `acpe-prod-dlk-shrd-gld.delivery_cliente_gtm.cr_segmentacion`
  WHERE periodo = DATE '2026-08-01'
),
cruce AS (
  SELECT
    COALESCE(a.cod_interlocutor, b.cod_interlocutor) AS cod_interlocutor,
    a.cod_boca_salida                                AS boca_anterior,
    b.cod_boca_salida                                AS boca_actual,
    -- vacío = NP; queda NULL solo si el cliente no existe en ese periodo
    IF(a.cod_interlocutor IS NULL, NULL, IFNULL(a.flg_potential, FALSE)) AS flg_ant,
    IF(b.cod_interlocutor IS NULL, NULL, IFNULL(b.flg_potential, FALSE)) AS flg_act,
    NULLIF(TRIM(a.des_potencial_predic), '')         AS pot_ant,
    NULLIF(TRIM(b.des_potencial_predic), '')         AS pot_act,
    a.cod_interlocutor IS NULL                       AS es_nuevo,
    b.cod_interlocutor IS NULL                       AS salio,
    -- estaba "Por definir": transaccional o estratégica sin segmento
    (IFNULL(LOWER(TRIM(a.des_segmento_transaccional)), '') IN ('', 'por definir', 'sin asignar')
     OR IFNULL(LOWER(TRIM(a.des_segmento_estrategica)), '') IN ('', 'por definir', 'sin asignar')) AS ant_por_definir,
    -- hoy sigue "Por definir" si su transaccional no tiene segmento
    IFNULL(LOWER(TRIM(b.des_segmento_transaccional)), '') IN ('', 'por definir', 'sin asignar') AS act_por_definir
  FROM anterior a
  FULL OUTER JOIN actual b ON a.cod_interlocutor = b.cod_interlocutor
)
SELECT
  cod_interlocutor,
  boca_anterior,
  boca_actual,
  CASE WHEN flg_ant IS TRUE  THEN 'P'
       WHEN flg_ant IS FALSE THEN 'NP'
       ELSE NULL
  END AS flg_potential_anterior,
  CASE WHEN flg_act IS TRUE  THEN 'P'
       WHEN flg_act IS FALSE THEN 'NP'
       ELSE NULL
  END AS flg_potential_actual,
  pot_ant AS des_potencial_anterior,
  pot_act AS des_potencial_actual,
  CASE
    WHEN es_nuevo THEN 'CLIENTE NUEVO'
    WHEN salio THEN 'CLIENTE SALIÓ'                                   -- no debería ocurrir
    WHEN boca_anterior IS DISTINCT FROM boca_actual THEN 'CAMBIO DE BOCA'  -- no debería ocurrir
    WHEN ant_por_definir AND NOT act_por_definir THEN 'POR DEFINIR A SEGMENTO'
    WHEN ant_por_definir THEN 'CAMBIO SIN SALIR DE POR DEFINIR'      -- no debería ocurrir
    ELSE 'CAMBIO DE POTENCIAL CON SEGMENTO PREVIO'                    -- no debería ocurrir
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
  OR flg_ant IS DISTINCT FROM flg_act
  OR pot_ant IS DISTINCT FROM pot_act
ORDER BY cambio_esperado, tipo_cambio, cod_interlocutor
