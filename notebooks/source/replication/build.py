import nbformat as nbf
import os
HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, "cells") + "/"
NB = os.path.join(HERE, "..", "..")
def build(model, out):
    seq = [("md", f"00_intro_{model}.md"), ("code", "01_config.py"), ("code", "01b_manifest.py"),
           ("h", "### 1 · Setup"), ("code", "02_setup.py"),
           ("h", "### 1b · Run record (+ resume from an earlier session)"), ("code", "02b_record.py"),
           ("h", "### 2 · Imports and GPU check"), ("code", "03_imports.py"),
           ("h", "### 3 · Benchmarks: tables, the replication sample, downloads, frames"), ("code", "04_data.py"),
           ("h", "### 4 · Model, T4 memory fix, our decoder loop, prefix caching"), ("code", f"05_model_{model}.py"),
           ("h", "### 4b · Safety checks (run on the first video)"), ("code", f"05b_checks_{model}.py"),
           ("h", "### 5 · Method code (FlashVID, HoliTom, VisionZip — same ports as the main study)"), ("code", "06a_methods.py"),
           ("h", "### 5b · Compressors for this model (+ self-test)"), ("code", f"06b_compress_{model}.py"),
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
            src = open(C + x).read().rstrip()
            if x == "01_config.py":
                src = src.replace("__MODEL__", model)
            nb.cells.append(nbf.v4.new_code_cell(src))
    nb.metadata = {"kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True},
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    nbf.write(nb, out)
    print(out, len(nb.cells), "cells")
build("llava_video", os.path.join(NB, "flips_rep_llava_video.ipynb"))
build("qwen3vl", os.path.join(NB, "flips_rep_qwen3vl.ipynb"))
