"""Copia LITERAL de `query_clientes` de cada notebook original, solo para test_data.py."""

QUERIES_ORIGINALES = {
    'lavanderias': """
WITH clientes AS (
    SELECT DISTINCT  
        a.cod_cliente_dex,
        a.tip_documento,
        a.des_provincia,
        a.des_ciudad,
        a.des_distrito,
        a.val_coordenada_x,
        a.val_coordenada_y, 
        b.des_segmento_estrategica as des_segmento_estrategico,
        b.des_segmento_transaccional AS segmento_trx,
        e.val_score_urbanicidad,
        g.cod_cliente_alicorp_actual
    FROM `acpe-prod-dlk-shrd-gld.delivery_modelo_dimensional.cr_cliente_dex` a
    LEFT JOIN 
    (select distinct * from `acpe-dev-uc-sandbox-aa.dev.cpm_segmentacion_unificada_diamante`
where periodo='2026-07-01') b
    on a.cod_cliente_dex=cod_interlocutor
    LEFT JOIN `acpe-dev-uc-ml.dev.equifax_urbanicidad_cliente` e
        ON a.cod_cliente_dex = LPAD(CAST(e.id_interlocutor_origen AS STRING), 10, '0')
    LEFT JOIN `acpe-dev-uc-ml.dev.equifax_encuesta_cliente` f
        ON a.cod_cliente_dex = LPAD(CAST(f.id_interlocutor_origen AS STRING), 10, '0')
    LEFT JOIN `acpe-dev-uc-sandbox-aa.dev.ultimos_valores` g
        ON a.cod_cliente_dex = g.cod_cliente_dex
    WHERE
        a.cod_grupo_precio_dex = 'LV'
        AND b.des_segmento_transaccional IN ('1 Oro+','2 Oro','3 Plata','4 Bronce')
        AND b.des_segmento_estrategica IN ('Lavanderia')
),

ventas AS (
    SELECT
        cod_cliente_dex,
        ROUND(ABS(
            SUM(CASE WHEN des_autoventa IN ('Dia a Dia', 'Insuma') THEN mnt_total ELSE 0 END)
            / NULLIF(SUM(CASE WHEN cod_clase_posicion = 'TNR' THEN mnt_total ELSE 0 END), 0) * 100
        ), 2) AS pct_venta_diaadia_insuma,
        ROUND(ABS(
            SUM(CASE WHEN cod_clase_posicion != 'TNR' THEN mnt_total ELSE 0 END)
            / NULLIF(SUM(CASE WHEN cod_clase_posicion = 'TNR' THEN mnt_total ELSE 0 END), 0) * 100
        ), 2) AS pct_descuento
    FROM `acpe-prod-dlk-shrd-gld.delivery_modelo_dimensional.cr_venta_cliente_dex_diario`
    WHERE periodo >= '2026-01-01'
      AND periodo <= '2026-07-01'
    GROUP BY cod_cliente_dex
)

SELECT
    c.*,
    v.pct_venta_diaadia_insuma,
    v.pct_descuento
FROM clientes c
LEFT JOIN ventas v
    ON c.cod_cliente_dex = v.cod_cliente_dex;
""",
    'hoteles': """
WITH clientes AS (
    SELECT DISTINCT  
        a.cod_cliente_dex,
        a.tip_documento,
        a.des_provincia,
        a.des_ciudad,
        a.des_distrito,
        a.val_coordenada_x,
        a.val_coordenada_y, 
        b.des_segmento_estrategica as des_segmento_estrategico,
        b.des_segmento_transaccional AS segmento_trx,
        e.val_score_urbanicidad,
        g.cod_cliente_alicorp_actual
    FROM `acpe-prod-dlk-shrd-gld.delivery_modelo_dimensional.cr_cliente_dex` a
    LEFT JOIN 
    (select distinct * from `acpe-dev-uc-sandbox-aa.dev.cpm_segmentacion_unificada_diamante`
where periodo='2026-07-01') b
    on a.cod_cliente_dex=cod_interlocutor
    LEFT JOIN `acpe-dev-uc-ml.dev.equifax_urbanicidad_cliente` e
        ON a.cod_cliente_dex = LPAD(CAST(e.id_interlocutor_origen AS STRING), 10, '0')
    LEFT JOIN `acpe-dev-uc-ml.dev.equifax_encuesta_cliente` f
        ON a.cod_cliente_dex = LPAD(CAST(f.id_interlocutor_origen AS STRING), 10, '0')
    LEFT JOIN `acpe-dev-uc-sandbox-aa.dev.ultimos_valores` g
        ON a.cod_cliente_dex = g.cod_cliente_dex
    WHERE
        a.cod_grupo_precio_dex = 'LV'
        AND b.des_segmento_transaccional IN ('1 Oro+','2 Oro','3 Plata','4 Bronce')
        AND b.des_segmento_estrategica IN ('Hotel/Casino')
),

ventas AS (
    SELECT
        cod_cliente_dex,
        ROUND(ABS(
            SUM(CASE WHEN des_autoventa IN ('Dia a Dia', 'Insuma') THEN mnt_total ELSE 0 END)
            / NULLIF(SUM(CASE WHEN cod_clase_posicion = 'TNR' THEN mnt_total ELSE 0 END), 0) * 100
        ), 2) AS pct_venta_diaadia_insuma,
        ROUND(ABS(
            SUM(CASE WHEN cod_clase_posicion != 'TNR' THEN mnt_total ELSE 0 END)
            / NULLIF(SUM(CASE WHEN cod_clase_posicion = 'TNR' THEN mnt_total ELSE 0 END), 0) * 100
        ), 2) AS pct_descuento
    FROM `acpe-prod-dlk-shrd-gld.delivery_modelo_dimensional.cr_venta_cliente_dex_diario`
    WHERE periodo >= '2026-01-01'
      AND periodo <= '2026-07-01'
    GROUP BY cod_cliente_dex
)

SELECT
    c.*,
    v.pct_venta_diaadia_insuma,
    v.pct_descuento
FROM clientes c
LEFT JOIN ventas v
    ON c.cod_cliente_dex = v.cod_cliente_dex;
""",
    'panaderia': """
WITH clientes AS (
select distinct  
    a.cod_cliente_dex, a.tip_documento, a.des_provincia, a.des_ciudad,
    a.des_distrito, a.val_coordenada_x, a.val_coordenada_y, 
    case when b.des_segmento_estrategica in ('Estadio 1', 'Estadio 2', 'Estadio 3','Estadio 4') then b.des_segmento_estrategica
    else '?' end as des_segmento_estrategico
    , b.des_segmento_transaccional as segmento_trx , 
      e.val_score_urbanicidad, g.cod_cliente_alicorp_actual
from `acpe-prod-dlk-shrd-gld.delivery_modelo_dimensional.cr_cliente_dex` a
left join 
(select distinct * from `acpe-dev-uc-sandbox-aa.dev.cpm_segmentacion_unificada_diamante`
where periodo='2026-07-01') b
    on a.cod_cliente_dex=cod_interlocutor
left join `acpe-dev-uc-ml.dev.equifax_urbanicidad_cliente` e
    on a.cod_cliente_dex = LPAD(CAST(e.id_interlocutor_origen AS string), 10, '0')
left join `acpe-dev-uc-ml.dev.equifax_encuesta_cliente` f
    on a.cod_cliente_dex = LPAD(CAST(f.id_interlocutor_origen AS string), 10, '0')
left join `acpe-dev-uc-sandbox-aa.dev.ultimos_valores` g
    on a.cod_cliente_dex=g.cod_cliente_dex
where 
a.cod_grupo_precio_dex='PA' 
   and b.des_segmento_transaccional in ('1 Oro+','2 Oro','3 Plata','4 Bronce')
),
ventas AS (
    SELECT
        cod_cliente_dex,
        ROUND(ABS(
            SUM(CASE WHEN des_autoventa IN ('Dia a Dia', 'Insuma') THEN mnt_total ELSE 0 END)
            / NULLIF(SUM(CASE WHEN cod_clase_posicion = 'TNR' THEN mnt_total ELSE 0 END), 0) * 100
        ), 2) AS pct_venta_diaadia_insuma,
        ROUND(ABS(
            SUM(CASE WHEN cod_clase_posicion != 'TNR' THEN mnt_total ELSE 0 END)
            / NULLIF(SUM(CASE WHEN cod_clase_posicion = 'TNR' THEN mnt_total ELSE 0 END), 0) * 100
        ), 2) AS pct_descuento
    FROM `acpe-prod-dlk-shrd-gld.delivery_modelo_dimensional.cr_venta_cliente_dex_diario`
    WHERE periodo >= '2026-01-01'
      AND periodo <= '2026-07-01'
    GROUP BY cod_cliente_dex
)

SELECT
    c.*,
    v.pct_venta_diaadia_insuma,
    v.pct_descuento
FROM clientes c
LEFT JOIN ventas v
    ON c.cod_cliente_dex = v.cod_cliente_dex
""",
    'gastro': """
WITH clientes AS (
    SELECT DISTINCT  
        a.cod_cliente_dex,
        a.tip_documento,
        a.des_provincia,
        a.des_ciudad,
        a.des_distrito,
        a.val_coordenada_x,
        a.val_coordenada_y, 
        b.des_segmento_estrategica as des_segmento_estrategico,
        b.des_segmento_transaccional AS segmento_trx,
        e.val_score_urbanicidad,
        g.cod_cliente_alicorp_actual
    FROM `acpe-prod-dlk-shrd-gld.delivery_modelo_dimensional.cr_cliente_dex` a
    LEFT JOIN 
    (select distinct * from `acpe-dev-uc-sandbox-aa.dev.cpm_segmentacion_unificada_diamante`
where periodo='2026-07-01') b
    on a.cod_cliente_dex=b.cod_interlocutor
    LEFT JOIN `acpe-dev-uc-ml.dev.equifax_urbanicidad_cliente` e
        ON a.cod_cliente_dex = LPAD(CAST(e.id_interlocutor_origen AS STRING), 10, '0')
    LEFT JOIN `acpe-dev-uc-ml.dev.equifax_encuesta_cliente` f
        ON a.cod_cliente_dex = LPAD(CAST(f.id_interlocutor_origen AS STRING), 10, '0')
    LEFT JOIN `acpe-dev-uc-sandbox-aa.dev.ultimos_valores` g
        ON a.cod_cliente_dex = g.cod_cliente_dex
    WHERE
        a.cod_grupo_precio_dex = 'GA'
        AND b.des_segmento_transaccional IN ('1 Oro+','2 Oro','3 Plata','4 Bronce')
        AND b.des_segmento_estrategica IN ('Comida Casera Criolla','Sanguchería','Chifa','Cebichería','Pollería','Otros Gastronomía')
),

ventas AS (
    SELECT
        cod_cliente_dex,
        ROUND(ABS(
            SUM(CASE WHEN des_autoventa IN ('Dia a Dia', 'Insuma') THEN mnt_total ELSE 0 END)
            / NULLIF(SUM(CASE WHEN cod_clase_posicion = 'TNR' THEN mnt_total ELSE 0 END), 0) * 100
        ), 2) AS pct_venta_diaadia_insuma,
        ROUND(ABS(
            SUM(CASE WHEN cod_clase_posicion != 'TNR' THEN mnt_total ELSE 0 END)
            / NULLIF(SUM(CASE WHEN cod_clase_posicion = 'TNR' THEN mnt_total ELSE 0 END), 0) * 100
        ), 2) AS pct_descuento
    FROM `acpe-prod-dlk-shrd-gld.delivery_modelo_dimensional.cr_venta_cliente_dex_diario`
    WHERE periodo >= '2026-01-01'
      AND periodo <= '2026-07-01'
    GROUP BY cod_cliente_dex
)

SELECT
    c.*,
    v.pct_venta_diaadia_insuma,
    v.pct_descuento
FROM clientes c
LEFT JOIN ventas v
    ON c.cod_cliente_dex = v.cod_cliente_dex;
""",
    'bodegas': """
WITH clientes AS (
    SELECT DISTINCT  
        a.cod_cliente_dex,
        a.tip_documento,
        a.des_provincia,
        a.des_ciudad,
        a.des_distrito,
        a.val_coordenada_x,
        a.val_coordenada_y, 
        b.des_segmento_estrategica as des_segmento_estrategico,
        b.des_segmento_transaccional AS segmento_trx,
        e.val_score_urbanicidad,
        g.cod_cliente_alicorp_actual, 
        a.flg_cliente_perfecto
    FROM `acpe-prod-dlk-shrd-gld.delivery_modelo_dimensional.cr_cliente_dex` a
    LEFT JOIN 
    (select distinct * from `acpe-dev-uc-sandbox-aa.dev.cpm_segmentacion_unificada_diamante`
where periodo='2026-07-01') b
    on a.cod_cliente_dex=b.cod_interlocutor
    LEFT JOIN `acpe-dev-uc-ml.dev.equifax_urbanicidad_cliente` e
        ON a.cod_cliente_dex = LPAD(CAST(e.id_interlocutor_origen AS STRING), 10, '0')
    LEFT JOIN `acpe-dev-uc-ml.dev.equifax_encuesta_cliente` f
        ON a.cod_cliente_dex = LPAD(CAST(f.id_interlocutor_origen AS STRING), 10, '0')
    LEFT JOIN `acpe-dev-uc-sandbox-aa.dev.ultimos_valores` g
        ON a.cod_cliente_dex = g.cod_cliente_dex
    WHERE
        a.cod_grupo_precio_dex = 'BO'
        AND b.des_segmento_transaccional IN ('1 Oro+','2 Oro','3 Plata','4 Bronce')
        AND b.des_segmento_estrategica IN ('Golosinera','Alejada','Estrella','Clásica')
),

ventas AS (
    SELECT
        cod_cliente_dex,
        ROUND(ABS(
            SUM(CASE WHEN des_autoventa IN ('Dia a Dia', 'Insuma') THEN mnt_total ELSE 0 END)
            / NULLIF(SUM(CASE WHEN cod_clase_posicion = 'TNR' THEN mnt_total ELSE 0 END), 0) * 100
        ), 2) AS pct_venta_diaadia_insuma,
        ROUND(ABS(
            SUM(CASE WHEN cod_clase_posicion != 'TNR' THEN mnt_total ELSE 0 END)
            / NULLIF(SUM(CASE WHEN cod_clase_posicion = 'TNR' THEN mnt_total ELSE 0 END), 0) * 100
        ), 2) AS pct_descuento
    FROM `acpe-prod-dlk-shrd-gld.delivery_modelo_dimensional.cr_venta_cliente_dex_diario`
    WHERE periodo >= '2026-01-01'
      AND periodo <= '2026-07-01'
    GROUP BY cod_cliente_dex
)

SELECT
    c.*,
    v.pct_venta_diaadia_insuma,
    v.pct_descuento
FROM clientes c
LEFT JOIN ventas v
    ON c.cod_cliente_dex = v.cod_cliente_dex;
""",
    'pm_mercado': """
WITH clientes AS (
    SELECT DISTINCT  
        a.cod_cliente_dex,
        a.tip_documento,
        a.des_provincia,
        a.des_ciudad,
        a.des_distrito,
        a.val_coordenada_x,
        a.val_coordenada_y, 
        b.des_segmento_estrategica as des_segmento_estrategico,
        b.des_segmento_transaccional AS segmento_trx,
        e.val_score_urbanicidad,
        g.cod_cliente_alicorp_actual, 
        a.flg_cliente_perfecto
    FROM `acpe-prod-dlk-shrd-gld.delivery_modelo_dimensional.cr_cliente_dex` a
    LEFT JOIN 
    (select distinct * from `acpe-dev-uc-sandbox-aa.dev.cpm_segmentacion_unificada_diamante`
where periodo='2026-07-01') b
    on a.cod_cliente_dex=b.cod_interlocutor
    LEFT JOIN `acpe-dev-uc-ml.dev.equifax_urbanicidad_cliente` e
        ON a.cod_cliente_dex = LPAD(CAST(e.id_interlocutor_origen AS STRING), 10, '0')
    LEFT JOIN `acpe-dev-uc-ml.dev.equifax_encuesta_cliente` f
        ON a.cod_cliente_dex = LPAD(CAST(f.id_interlocutor_origen AS STRING), 10, '0')
    LEFT JOIN `acpe-dev-uc-sandbox-aa.dev.ultimos_valores` g
        ON a.cod_cliente_dex = g.cod_cliente_dex
    WHERE
        a.cod_grupo_precio_dex = 'PM'
        AND b.des_segmento_transaccional IN ('1 Oro+','2 Oro','3 Plata','4 Bronce')
        AND b.des_segmento_estrategica IN ('Abarrotero','Lavandero','Confitero')
),

ventas AS (
    SELECT
        cod_cliente_dex,
        ROUND(ABS(
            SUM(CASE WHEN des_autoventa IN ('Dia a Dia', 'Insuma') THEN mnt_total ELSE 0 END)
            / NULLIF(SUM(CASE WHEN cod_clase_posicion = 'TNR' THEN mnt_total ELSE 0 END), 0) * 100
        ), 2) AS pct_venta_diaadia_insuma,
        ROUND(ABS(
            SUM(CASE WHEN cod_clase_posicion != 'TNR' THEN mnt_total ELSE 0 END)
            / NULLIF(SUM(CASE WHEN cod_clase_posicion = 'TNR' THEN mnt_total ELSE 0 END), 0) * 100
        ), 2) AS pct_descuento
    FROM `acpe-prod-dlk-shrd-gld.delivery_modelo_dimensional.cr_venta_cliente_dex_diario`
    WHERE periodo >= '2026-01-01'
      AND periodo <= '2026-07-01'
    GROUP BY cod_cliente_dex
)

SELECT
    c.*,
    v.pct_venta_diaadia_insuma,
    v.pct_descuento
FROM clientes c
LEFT JOIN ventas v
    ON c.cod_cliente_dex = v.cod_cliente_dex;
""",
}
