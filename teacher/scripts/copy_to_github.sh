#!/usr/bin/env bash

# Sync what needs to be synced to another folder containing git/master branch
# usage bash scripts/copy_to_github.sh from rootdir

# Build and copy the doc to isae-practical-deep-learning/gh-pages folder which contains gh-pages branch
base_path=$(pwd)
cd docs
python build_doc.py
python execute_notebooks.py
cd ${base_path}

mkdocs build
rsync -rd site/ ../isae-practical-deep-learning/gh-pages/site/

# Copy the code to isae-practical-deep-learning which contains master

# src/
rsync -rd src/khumeia/ ../isae-practical-deep-learning/src/khumeia/
rsync -rd src/tests/ ../isae-practical-deep-learning/src/tests/
rsync -rd src/MANIFEST.in ../isae-practical-deep-learning/src/MANIFEST.in
rsync -rd src/README.md ../isae-practical-deep-learning/src/README.md
rsync -rd src/requirements* ../isae-practical-deep-learning/src/
rsync -rd src/setup* ../isae-practical-deep-learning/src/

# notebooks
rsync -rd notebooks/*.ipynb ../isae-practical-deep-learning/notebooks/

# gcp
rsync -rd gcp/ ../isae-practical-deep-learning/gcp/

# gitignore
rsync -rd .gitignore ../isae-practical-deep-learning/
