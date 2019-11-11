#!/usr/bin/env bash

# I use this script with "watch -n 60 'bash docs/build_docs.sh' and mkdocs serve"
# from rootdir to periodically rebuild and serve the documentation
base_path=$(pwd)
cd docs
python build_doc.py
python execute_notebooks.py
cd ${base_path}

