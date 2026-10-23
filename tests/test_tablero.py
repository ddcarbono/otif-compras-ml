"""El tablero arranca con lo que trae ``data/`` y responde en sus tres páginas. Usa el cliente de prueba de Flask, sin
navegador."""
import pytest

import app as tablero


@pytest.fixture(scope="module")
def cliente():
    return tablero.server.test_client()


def test_registra_las_tres_paginas():
    import dash
    assert [p["name"] for p in dash.page_registry.values()] == ["Contexto", "EDA", "Modelos"]


@pytest.mark.parametrize("ruta", ["/", "/eda", "/modelos"])
def test_cada_pagina_responde(cliente, ruta):
    r = cliente.get(ruta)
    assert r.status_code == 200


def test_layouts_se_construyen():
    import dash
    for p in dash.page_registry.values():
        lay = p["layout"]() if callable(p["layout"]) else p["layout"]
        assert lay is not None
