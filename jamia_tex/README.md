# JAMIA TeX Package

This folder contains a submission-oriented TeX package for the SA-AKI conformal selective-triage manuscript, formatted against the official Oxford University Press authoring template files.

## Contents

- `main.tex`: primary JAMIA manuscript source
- `refs.bib`: paper-specific bibliography
- `oup-authoring-template.cls`, `oup-plain.bst`, `oup-abbrvnat.bst`: official OUP template assets
- `figures/`: numbered main-text and appendix figures
- `tables/`: generated TeX tabular snippets
- `upstream/official/`: retained copy of the downloaded official template inputs

## Regeneration

From the repository root:

```bash
/opt/ml-venv/bin/python saaki/build_jamia_tex.py
```

This script:

1. Copies the official OUP class and bibliography files into `jamia_tex/`.
2. Copies artifact figures and re-renders the workflow diagrams as static PNG/PDF files.
3. Regenerates all table snippets from the canonical artifact bundle.
4. Writes `refs.bib`, `main.tex`, and this README.

## Suggested Build Sequence

If a local TeX engine is available:

```bash
cd jamia_tex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

`latexmk -pdf main.tex` is also reasonable if installed.

## Notes

- The author, affiliation, and correspondence fields in `main.tex` are placeholders and should be replaced with submission metadata.
- The TeX package is built from the chaptered markdown manuscript in `saaki/jamia_manuscript/` plus the canonical artifacts in `local_outputs/artifacts/`.
- Local PDF compilation was not validated in this environment because a TeX engine was not available.
