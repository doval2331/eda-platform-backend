"""Generación de evidencias sintéticas en alta dimensión (prototipo sin embeddings pesados)."""

from __future__ import annotations

import random
from typing import Literal

import numpy as np

Modality = Literal["texto", "imagen", "multimodal"]

PREVIEW_TEXTO: dict[int, list[str]] = {
    0: [
        "Ticket: falla de login por credenciales inválidas",
        "Error 401 al autenticar usuarios",
        "Bloqueo de usuario tras varios intentos fallidos",
    ],
    1: [
        "Incidencia en verificación de identidad (KYC)",
        "Error en validación de documentos de cliente",
        "Proceso KYC incompleto o rechazado",
    ],
    2: [
        "Caída de servicio en API de pedidos",
        "Timeouts en endpoint /orders",
        "Latencia elevada en integración externa",
    ],
    3: [
        "Permisos insuficientes para operación crítica",
        "Error 403 al aprobar transacción",
        "Fallo de autorización en módulo de pagos",
    ],
    4: [
        "Picos de tráfico y saturación de colas",
        "Retrasos en procesamiento por alta concurrencia",
        "Degradación de rendimiento en horario punta",
    ],
    5: [
        "Errores 500 en servicios backend",
        "Excepciones no controladas en generación de reportes",
        "Fallo general en proceso batch nocturno",
    ],
}

PREVIEW_IMAGEN: dict[int, list[str]] = {
    0: ["Captura de pantalla de error de login", "Gráfico de intentos de acceso fallidos"],
    1: ["Documento de identidad borroso", "Formulario de alta de cliente incompleto"],
    2: ["Dashboard con caída de disponibilidad", "Gráfica de latencia elevada en API"],
    3: ["Pantalla con mensaje de permiso denegado", "Diagrama de roles restringidos"],
    4: ["Gráfico de picos de tráfico", "Monitorización de colas saturadas"],
    5: ["Captura de error 500 en consola", "Log de excepción en servicio backend"],
}


def seed_for_modality(modality: Modality, base_seed: int) -> int:
    offsets = {"texto": 0, "imagen": 57, "multimodal": 113}
    return base_seed + offsets.get(modality, 0)


def _preview(modality: Modality, cluster_id: int, rng: random.Random) -> str:
    if modality == "imagen":
        return rng.choice(PREVIEW_IMAGEN.get(cluster_id, PREVIEW_IMAGEN[0]))
    if modality == "multimodal":
        t = rng.choice(PREVIEW_TEXTO.get(cluster_id, PREVIEW_TEXTO[0]))
        img = rng.choice(PREVIEW_IMAGEN.get(cluster_id, PREVIEW_IMAGEN[0]))
        return f"Texto: {t} | Imagen: {img}"
    return rng.choice(PREVIEW_TEXTO.get(cluster_id, PREVIEW_TEXTO[0]))


def generate_high_dim_features(
    n_samples: int,
    n_features: int,
    n_clusters: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Matriz de características en R^n_features con estructura de clusters gaussianos.
    Devuelve X, etiquetas verdaderas (solo para metadata) y metadatos por punto.
    """
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    sizes = [44, 36, 48, 30, 38, 24]
    while sum(sizes) < n_samples:
        sizes.append(max(8, n_samples // (n_clusters + 2)))
    sizes = sizes[:n_clusters]
    total = sum(sizes)
    if total > n_samples:
        excess = total - n_samples
        for i in range(len(sizes) - 1, -1, -1):
            if excess <= 0:
                break
            drop = min(sizes[i] - 5, excess)
            sizes[i] -= drop
            excess -= drop
    while sum(sizes) < n_samples:
        sizes[0] += 1

    centroids = rng.standard_normal((n_clusters, n_features)) * 2.5
    X_parts: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict] = []

    idx = 0
    for c in range(n_clusters):
        n = sizes[c] if c < len(sizes) else 0
        if n <= 0:
            continue
        cov_scale = 0.35 + 0.08 * c
        block = centroids[c] + rng.standard_normal((n, n_features)) * cov_scale
        X_parts.append(block)
        labels.extend([c] * n)
        for _ in range(n):
            idx += 1
            metadata.append(
                {
                    "id": f"e{idx:03d}",
                    "preview": _preview("texto", c, py_rng),
                    "source": "texto",
                }
            )

    X = np.vstack(X_parts).astype(np.float64)
    true_labels = np.array(labels, dtype=np.int32)

    if len(X) > n_samples:
        X = X[:n_samples]
        true_labels = true_labels[:n_samples]
        metadata = metadata[:n_samples]

    return X, true_labels, metadata


def apply_modality_to_metadata(
    metadata: list[dict],
    true_labels: np.ndarray,
    modality: Modality,
    seed: int,
) -> list[dict]:
    py_rng = random.Random(seed + 7)
    out: list[dict] = []
    for i, meta in enumerate(metadata):
        cluster_id = int(true_labels[i]) if i < len(true_labels) else 0
        out.append(
            {
                "id": meta["id"],
                "preview": _preview(modality, cluster_id, py_rng),
                "source": modality,
            }
        )
    return out
