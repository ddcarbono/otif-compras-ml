"""Quita el </div> de más que deja la bibliografía de sphinxcontrib-bibtex 2.5 con docutils 0.18.

La lista de referencias cierra su contenedor dos veces. En una página con {bibliography} (la portada), ese cierre de más
saca el pie del artículo («Siguiente» y el autor) fuera de su columna, y el tema lo pinta arriba a la derecha. Se
corrige en el HTML de la página, antes de escribirlo, solo si sobran cierres después de la última referencia.
"""
import re


def _arreglar(app, pagename, templatename, context, doctree):
    body = context.get("body", "")
    if "doc-biblioentry" not in body:
        return
    exceso = body.count("</div>") - len(re.findall(r"<div[\s>]", body))
    if exceso <= 0:
        return
    i = body.rfind('role="doc-biblioentry"')
    j = body.find("</div>", i) + len("</div>")      # cierra la última referencia
    k = body.find("</div>", j) + len("</div>")      # cierra el contenedor de la lista
    context["body"] = body[:k] + body[k:].replace("</div>", "", exceso)


def setup(app):
    app.connect("html-page-context", _arreglar)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
