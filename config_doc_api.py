"""
generate_docs.py
----------------
Run this script once from the root of the project to automatically
generate all RST files needed for the Sphinx documentation.

Usage:
    uv run python generate_docs.py
"""

from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
DOCS_DIR = Path("docs")

MODULES = {
    "data": [
        ("eurostat",        "Eurostat API Client",          "morta_nuts2.data.eurostat"),
        ("mortality_table", "Mortality Table",              "morta_nuts2.data.mortality_table"),
    ],
    "model": [
        ("bsplines",    "B-Splines",                        "morta_nuts2.model.Bsplines.Bsplines"),
        ("lcp",         "Lee-Carter Parametric Model",      "morta_nuts2.model.LC_p.lcp_class"),
        ("llp",         "Lee & Li Parametric Model",        "morta_nuts2.model.LL_p.llp_class"),
        ("param_init",  "Parameter Initialization",         "morta_nuts2.model.parameters_init.param_init"),
        ("projection",  "Mortality Projection Engine",      "morta_nuts2.model.projection.project_class"),
    ],
    "visualisation": [
        ("plot_class",         "Plot Class",         "morta_nuts2.graph.plot_class"),
        ("mortality_plotter",  "Mortality Plotter",  "morta_nuts2.visualisation.MortalityPlotter"),
    ],
}

RST_TEMPLATE = """{title}
{underline}

.. automodule:: {module}
   :members:
   :undoc-members:
   :show-inheritance:
"""

# ── Generation ─────────────────────────────────────────────────────────────
def generate():
    for section, entries in MODULES.items():
        folder = DOCS_DIR / "api" / section
        folder.mkdir(parents=True, exist_ok=True)

        for filename, title, module in entries:
            rst_path = folder / f"{filename}.rst"
            content = RST_TEMPLATE.format(
                title=title,
                underline="=" * len(title),
                module=module,
            )
            rst_path.write_text(content, encoding="utf-8")
            print(f"  ✔  {rst_path}")

    print("\nAll RST files generated successfully.")
    print("Run  'uv run make html'  inside the docs/ folder to build the site.")


if __name__ == "__main__":
    generate()