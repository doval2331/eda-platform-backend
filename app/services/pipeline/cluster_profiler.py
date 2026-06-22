"""
Cluster profiler genérico — calcula estadísticas operativas
por cluster comparadas con el dataset global.
No asume nada sobre el dominio de los datos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# ESTADÍSTICAS GLOBALES
# ─────────────────────────────────────────────────────────────

def calcular_stats_globales(
    df: pd.DataFrame,
    exclude_cols: list[str] | None = None,
) -> dict:
    """
    Calcula estadísticas globales del dataset completo.
    Sirve como referencia para comparar cada cluster.
    """
    exclude = set(exclude_cols or [])
    exclude.update([
        "incident_id", "client_id", "descripcion_corta",
        "synthetic_segment", "segment", "description",
        "descripcion_larga", "texto", "text",
    ])

    bool_cols = [c for c in df.select_dtypes(include=["bool"]).columns
                 if c not in exclude]
    num_cols  = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in exclude]
    cat_cols  = [c for c in df.select_dtypes(include=["object"]).columns
                 if c not in exclude]
    bin_cols  = [c for c in num_cols
                 if df[c].dropna().isin([0, 1]).all()] + bool_cols
    pure_num  = [c for c in num_cols if c not in bin_cols]

    stats: dict = {
        "numeric":     {},
        "binary":      {},
        "categorical": {},
        "n_total":     len(df),
    }

    for col in pure_num:
        vals = df[col].dropna()
        if len(vals) > 0:
            stats["numeric"][col] = {
                "mean":   round(float(vals.mean()), 3),
                "median": round(float(vals.median()), 3),
                "std":    round(float(vals.std()), 3),
                "min":    round(float(vals.min()), 3),
                "max":    round(float(vals.max()), 3),
            }

    for col in bin_cols:
        vals = df[col].dropna()
        if len(vals) > 0:
            stats["binary"][col] = {
                "pct": round(float(vals.astype(float).mean() * 100), 2)
            }

    for col in cat_cols:
        vals = df[col].dropna().astype(str)
        if len(vals) > 0:
        
        
            dist = vals.value_counts(normalize=True).head(6).round(4).to_dict()
            stats["categorical"][col] = {
                "mode":         vals.mode()[0],
                "pct_mode":     round(
                    vals.value_counts(normalize=True).iloc[0] * 100, 1
                ),
                "distribution": {k: round(v * 100, 1) for k, v in dist.items()},
                "n_unique":     int(vals.nunique()),
            }

    return stats


# ─────────────────────────────────────────────────────────────
# INTERPRETACIÓN GENÉRICA
# ─────────────────────────────────────────────────────────────

def _interpretar_generico(
    signals_high: list[dict],
    signals_low:  list[dict],
    signals_cat:  list[dict],
) -> str:
    """
    Genera interpretación comparando señales detectadas.
    Completamente genérico — no depende del dominio.
    """
    partes = []

    if signals_high:
        tops = signals_high[:2]
        desc = " | ".join(
            f"{s['variable']} alto: {s['valor_cluster']} "
            f"vs {s['valor_global']} global"
            + (f" (z={s['z']:.1f})" if s.get("z") else "")
            for s in tops
        )
        partes.append(f"↑ {desc}")

    if signals_low:
        tops = signals_low[:2]
        desc = " | ".join(
            f"{s['variable']} bajo: {s['valor_cluster']} "
            f"vs {s['valor_global']} global"
            + (f" (z={s['z']:.1f})" if s.get("z") else "")
            for s in tops
        )
        partes.append(f"↓ {desc}")

    if signals_cat:
        tops = signals_cat[:2]
        desc = " | ".join(
            f"{s['variable']} concentrado en '{s['mode']}' "
            f"({s['pct_mode']:.0f}% vs {s['pct_global']:.0f}% global)"
            for s in tops
        )
        partes.append(f"◆ {desc}")

    if not partes:
        return "Perfil similar al promedio global — sin características distintivas"
    return " | ".join(partes)


# ─────────────────────────────────────────────────────────────
# NIVEL DE DESVIACIÓN — genérico
# ─────────────────────────────────────────────────────────────

def _nivel_desviacion(
    signals_high: list[dict],
    signals_low:  list[dict],
) -> str:
    """
    Clasifica cuánto se desvía el cluster del promedio global.
    Basado en z-scores — completamente genérico.
    """
    z_scores = [
        abs(s["z"]) for s in signals_high + signals_low
        if s.get("z") is not None
    ]
    if not z_scores:
        return "tipico"
    z_max  = max(z_scores)
    z_mean = float(np.mean(z_scores))
    if z_max > 4 or z_mean > 2.5:
        return "muy_atipico"
    elif z_max > 2 or z_mean > 1.5:
        return "atipico"
    return "tipico"


def _nivel_homogeneidad(signals_cat: list[dict]) -> str:
    """Qué tan homogéneo es el cluster en variables categóricas."""
    if not signals_cat:
        return "mixto"
    media = float(np.mean([s["pct_mode"] for s in signals_cat]))
    if media > 90:
        return "muy_homogeneo"
    elif media > 70:
        return "homogeneo"
    return "mixto"


def _nivel_tamaño(pct_total: float) -> str:
    """Tamaño relativo del cluster respecto al dataset."""
    if pct_total > 20:
        return "dominante"
    elif pct_total > 10:
        return "significativo"
    elif pct_total > 5:
        return "menor"
    return "pequeño"


# ─────────────────────────────────────────────────────────────
# PROFILER PRINCIPAL
# ─────────────────────────────────────────────────────────────

def cluster_profiler(
    df: pd.DataFrame,
    labels: np.ndarray,
    stats_globales: dict | None = None,
    exclude_cols: list[str] | None = None,
) -> list[dict]:
    """
    Calcula el perfil operativo de cada cluster.
    Detecta automáticamente el tipo de cada variable.
    Compara con estadísticas globales del dataset.
    Genérico — funciona con cualquier dataset tabular.
    """
    exclude = set(exclude_cols or [])
    exclude.update([
        "incident_id", "client_id", "descripcion_corta",
        "synthetic_segment", "segment", "description",
        "descripcion_larga", "texto", "text",
    ])

    # Calcular estadísticas globales si no se pasan
    if stats_globales is None:
        stats_globales = calcular_stats_globales(df, exclude_cols)

    bool_cols = [c for c in df.select_dtypes(include=["bool"]).columns
                 if c not in exclude]
    num_cols  = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in exclude]
    cat_cols  = [c for c in df.select_dtypes(include=["object"]).columns
                 if c not in exclude]
    bin_cols  = [c for c in num_cols
                 if df[c].dropna().isin([0, 1]).all()] + bool_cols
    pure_num  = [c for c in num_cols if c not in bin_cols]

    clusters = sorted(set(int(x) for x in labels))
    perfiles = []

    for cl in clusters:
        mask     = labels == cl
        subset   = df[mask]
        n        = int(mask.sum())
        es_ruido = cl == -1
        pct      = round(n / len(df) * 100, 1)

        perfil: dict = {
            "cluster_id":        cl,
            "label":             "Ruido" if es_ruido else f"Cluster {cl}",
            "size":              n,
            "pct_total":         pct,
            "es_ruido":          es_ruido,
            "numeric_stats":     {},
            "binary_stats":      {},
            "categorical_stats": {},
            "signals_high":      [],
            "signals_low":       [],
            "signals_cat":       [],
            "interpretation":    "",
            "nivel_desviacion":  "",
            "nivel_homogeneidad":"",
            "nivel_tamaño":      "",
        }

        # ── Numéricas continuas ───────────────────────────────
        for col in pure_num:
            vals = subset[col].dropna()
            if len(vals) == 0:
                continue
            g      = stats_globales["numeric"].get(col, {})
            mean_g = g.get("mean", 0)
            std_g  = g.get("std", 1) or 1
            mean_cl = float(vals.mean())
            z       = (mean_cl - mean_g) / std_g

            perfil["numeric_stats"][col] = {
                "mean":        round(mean_cl, 2),
                "median":      round(float(vals.median()), 2),
                "std":         round(float(vals.std()), 2),
                "min":         round(float(vals.min()), 2),
                "max":         round(float(vals.max()), 2),
                "z_score":     round(z, 2),
                "vs_global":   round(mean_cl - mean_g, 2),
                "mean_global": mean_g,
            }

            signal = {
                "variable":       col,
                "valor_cluster":  round(mean_cl, 2),
                "valor_global":   mean_g,
                "z":              round(z, 2),
            }
            if z > 1.5:
                perfil["signals_high"].append(signal)
            elif z < -1.5:
                perfil["signals_low"].append(signal)

        # Ordenar señales por magnitud
        perfil["signals_high"].sort(key=lambda s: abs(s["z"]), reverse=True)
        perfil["signals_low"].sort(key=lambda s: abs(s["z"]), reverse=True)

        # ── Binarias ──────────────────────────────────────────
        for col in bin_cols:
            vals       = subset[col].dropna()
            pct_global = stats_globales["binary"].get(col, {}).get("pct", 50)
            if len(vals) == 0:
                continue
            pct_cl = round(float(vals.astype(float).mean() * 100), 1)
            ratio  = pct_cl / pct_global if pct_global > 0 else 0

            perfil["binary_stats"][col] = {
                "pct":        pct_cl,
                "pct_global": pct_global,
                "ratio":      round(ratio, 2),
            }

            signal = {
                "variable":      col,
                "valor_cluster": pct_cl,
                "valor_global":  pct_global,
                "z":             None,
            }
            if ratio > 1.5:
                perfil["signals_high"].append(signal)
            elif ratio < 0.5:
                perfil["signals_low"].append(signal)

        # ── Categóricas ───────────────────────────────────────
        for col in cat_cols:
            vals = subset[col].dropna().astype(str)
            if len(vals) == 0:
                continue
            moda     = vals.mode()[0]
            pct_moda = round(
                vals.value_counts(normalize=True).iloc[0] * 100, 1
            )
            dist     = vals.value_counts(normalize=True).head(6).round(4).to_dict()
            g_dist   = stats_globales["categorical"].get(col, {}).get(
                "distribution", {}
            )
            g_pct_mode = g_dist.get(moda, 0)

            perfil["categorical_stats"][col] = {
                "mode":            moda,
                "pct_mode":        pct_moda,
                "pct_mode_global": g_pct_mode,
                "concentration":   round(pct_moda / g_pct_mode, 2)
                                   if g_pct_mode > 0 else None,
                "distribution":    {
                    k: round(v * 100, 1) for k, v in dist.items()
                },
                "n_unique": int(vals.nunique()),
            }

            if pct_moda > 80:
                perfil["signals_cat"].append({
                    "variable":  col,
                    "mode":      moda,
                    "pct_mode":  pct_moda,
                    "pct_global": g_pct_mode,
                })

        # ── Clasificación genérica ────────────────────────────
        if es_ruido:
            perfil["interpretation"]   = (
                "Casos atípicos — combinaciones inusuales de variables"
            )
            perfil["nivel_desviacion"]  = "anomalia"
            perfil["nivel_homogeneidad"]= "mixto"
            perfil["nivel_tamaño"]      = _nivel_tamaño(pct)
        else:
            perfil["interpretation"]    = _interpretar_generico(
                perfil["signals_high"],
                perfil["signals_low"],
                perfil["signals_cat"],
            )
            perfil["nivel_desviacion"]  = _nivel_desviacion(
                perfil["signals_high"],
                perfil["signals_low"],
            )
            perfil["nivel_homogeneidad"] = _nivel_homogeneidad(
                perfil["signals_cat"]
            )
            perfil["nivel_tamaño"]      = _nivel_tamaño(pct)

        perfiles.append(perfil)

    return perfiles


# ─────────────────────────────────────────────────────────────
# MODO DE VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────

def modo_visualizacion(n_clusters: int) -> str:
    """
    Determina el modo de visualización según el número de clusters.
    El frontend usa esto para decidir qué componente renderizar.
    """
    if n_clusters <= 4:
        return "detallado"
    elif n_clusters <= 15:
        return "mapa_tabla"
    return "mapa_paginado"