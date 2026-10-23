"""Lo que se publica en el repositorio no lleva identificadores sin anonimizar ni el pipeline con las tablas de
entidades (las mismas reglas que verifica ``publicar_repo.sh`` antes de copiar)."""
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
PROHIBIDAS = {"oc_id", "proveedor", "material", "sp_creador", "creador_oc", "ultimo_liberador", "imp_ceco"}
CSV = [p for p in (RAIZ / "data").rglob("*.csv") if not p.name.startswith("otif_depurado")]
PARQUET = list((RAIZ / "data").rglob("*.parquet"))


@pytest.mark.parametrize("ruta", CSV + PARQUET, ids=lambda p: str(p.relative_to(RAIZ)))
def test_data_sin_identificadores(ruta):
    if ruta.name == "diccionario.csv":     # describe las variables; no trae filas de datos
        return
    cols = pd.read_csv(ruta, nrows=0).columns if ruta.suffix == ".csv" else pd.read_parquet(ruta).columns
    assert not PROHIBIDAS & {c.strip().lower() for c in cols}


def test_el_pipeline_con_tablas_de_entidades_esta_ignorado():
    assert "artifacts/logistica.joblib" in (RAIZ / ".gitignore").read_text(encoding="utf-8")
