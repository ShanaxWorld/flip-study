"""Builds the two revision-run notebooks from their cell files."""
import os, nbformat as nbf
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("NB_OUT", os.path.join(HERE, "..", ".."))
META = {"kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
def build(cdir, seq, out):
    nb = nbf.v4.new_notebook()
    for kind, x in seq:
        if kind == "h":
            nb.cells.append(nbf.v4.new_markdown_cell(x))
        elif kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(open(os.path.join(cdir, x)).read().strip()))
        else:
            nb.cells.append(nbf.v4.new_code_cell(open(os.path.join(cdir, x)).read().rstrip()))
    nb.metadata = META
    nbf.write(nb, out); print(out, len(nb.cells), "cells")
build(os.path.join(HERE, "ov", "cells"), [
    ("md", "00_intro.md"), ("code", "01_config.py"),
    ("h", "### 0 · Reference answers of the main study (for the determinism check)"), ("code", "01c_reference.py"),
    ("h", "### 1 · Setup"), ("code", "02_setup.py"),
    ("h", "### 1b · Run record (+ resume from an earlier session)"), ("code", "02b_record.py"),
    ("h", "### 2 · Imports and GPU check"), ("code", "03_imports.py"),
    ("h", "### 3 · Benchmarks: tables, the main study's sampling, downloads, frames"), ("code", "04_data.py"),
    ("h", "### 4 · Model, T4 memory fix, our decoder loop, prefix caching"), ("code", "05_model.py"),
    ("h", "### 4b · Safety checks (run on the first video)"), ("code", "05b_check.py"),
    ("h", "### 5 · V-CAST (+ self-test)"), ("code", "06_compressors.py"),
    ("h", "### 6 · Revision run"), ("code", "07_run.py"),
    ("h", "### 7 · Summary and determinism check"), ("code", "08_analysis.py"),
    ("md", "09_end.md")], os.path.join(OUT, "flips_ext_llavaov.ipynb"))
build(os.path.join(HERE, "qw", "cells"), [
    ("md", "00_intro_qwen3vl.md"), ("code", "01_config.py"), ("code", "01b_manifest.py"),
    ("h", "### 0 · Reference answers of the replication run (for the determinism check)"), ("code", "01c_reference.py"),
    ("h", "### 1 · Setup"), ("code", "02_setup.py"),
    ("h", "### 1b · Run record (+ resume from an earlier session)"), ("code", "02b_record.py"),
    ("h", "### 2 · Imports and GPU check"), ("code", "03_imports.py"),
    ("h", "### 3 · Benchmarks: tables, the replication sample, downloads, frames"), ("code", "04_data.py"),
    ("h", "### 4 · Model, T4 memory fix, our decoder loop, prefix caching"), ("code", "05_model_qwen3vl.py"),
    ("h", "### 4b · Safety checks (run on the first video)"), ("code", "05b_checks_qwen3vl.py"),
    ("h", "### 5 · V-CAST with the official Qwen3-VL code (+ self-test)"), ("code", "06b_compress_qwen3vl.py"),
    ("h", "### 6 · Revision run"), ("code", "07_run.py"),
    ("h", "### 7 · Summary and determinism check"), ("code", "08_analysis.py"),
    ("md", "09_end.md")], os.path.join(OUT, "flips_ext_qwen3vl_vcast.ipynb"))
