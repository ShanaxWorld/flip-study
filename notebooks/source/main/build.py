import nbformat as nbf
import os
HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, "cells") + "/"
NB = os.path.join(HERE, "..", "..")
seq = [("md", "00_intro.md"), ("code", "01_config.py"),
       ("h", "### 1 · Setup"), ("code", "02_setup.py"),
       ("h", "### 1b · Run record (+ resume from an earlier session)"), ("code", "02b_record.py"),
       ("h", "### 2 · Imports and GPU check"), ("code", "03_imports.py"),
       ("h", "### 3 · Benchmarks: tables, pre-registered sampling, downloads, frames"), ("code", "04_data.py"),
       ("h", "### 4 · Model, T4 memory fix, our decoder loop, prefix caching"), ("code", "05_model.py"),
       ("h", "### 4b · Safety checks (run on the first video)"), ("code", "05b_check.py"),
       ("h", "### 5 · Compressors: all eight methods (+ self-test)"), ("code", "06_compressors.py"),
       ("h", "### 6 · Main run"), ("code", "07_run.py"),
       ("h", "### 7 · Per-benchmark analysis"), ("code", "08_analysis.py"),
       ("md", "09_end.md")]
nb = nbf.v4.new_notebook()
for kind, x in seq:
    if kind == "h":
        nb.cells.append(nbf.v4.new_markdown_cell(x))
    elif kind == "md":
        nb.cells.append(nbf.v4.new_markdown_cell(open(C + x).read().strip()))
    else:
        nb.cells.append(nbf.v4.new_code_cell(open(C + x).read().rstrip()))
nb.metadata = {"kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True},
               "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, os.path.join(NB, "flips_main_llavaov.ipynb"))
print(len(nb.cells), "cells")
