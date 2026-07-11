# Metadata, diccionario semantico y rol del LLM

La pantalla de Metadata muestra una lectura funcional de las columnas del dataset antes de que
lleguen al Dashboard Conversacional.

## Regla principal

El LLM no valida, no activa y no inventa variables.

Metadata se apoya en:

- perfil calculado por backend;
- columnas reales del dataset;
- reglas de exclusion;
- diccionario semantico por proyecto/contexto;
- estados de graficabilidad y uso recomendado.

El LLM puede explicar esos metadatos ya calculados, pero no debe convertir una variable inexistente,
inactiva o poco interpretable en una variable valida.

## Estados visibles en UI

La UI puede mostrar indicadores como:

- variable util para analisis;
- variable no recomendada;
- activa o inactiva en diccionario semantico;
- graficable o no graficable;
- disponible para explicacion asistida por LLM;
- muchos nulos;
- cardinalidad alta;
- no usar como metrica;
- no usar como dimension.

## Relacion con Dashboard Conversacional

El Dashboard Conversacional no deberia trabajar con columnas libres. Sus recomendaciones y
visualizaciones deben usar variables presentes en el dataset y aceptadas por el diccionario o por
validaciones del backend.

Si una variable sugerida por el LLM no existe, esta inactiva o no es interpretable, el backend debe
rechazarla, ajustarla o mostrar una explicacion clara.

## Evidencia real vs interpretacion asistida

Metadata solo describe variables y calidad de columnas. Una variable queda asociada a evidencia real
cuando el dashboard o el drill-down calculan tickets/evidencias reales sobre DuckDB. Hasta entonces,
la explicacion del LLM debe tratarse como interpretacion asistida, no como evidencia comprobada.
