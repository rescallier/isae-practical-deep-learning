# Development scripts
 
 (do not copy them to the public github)

## `copy_to_public_github.sh`

I use this in conjunction with watch (watch -n 60 'bash scripts/copy_to_public_github.sh') to transfer codefrom this repository to the public repo. The private datalab repository is the dev reference.

## `get_data.sh` and `process_raw_to_input_format.py`

This script traces the steps taken to generate the dataset that is avaiable through `khumeia`.

## `mkdocs_serve.sh`

I use this script with "watch -n 60 'bash docs/build_docs.sh' and mkdocs serve" from rootdir to periodically rebuild and serve the documentation 

## `build_to_gh_pages.sh`

I use this script with "watch -n 60 'bash scripts/build_to_gh_pages.sh'" for root dir to periodically rebuild and copy to another folder (which contains the github/gh-pages branch)