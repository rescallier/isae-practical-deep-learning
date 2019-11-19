# script that builds a notebook, converts
jupytext --to notebook 1_session_1_hands_on.py -o tmp.ipynb
nb-filter-cells -i tmp.ipynb -t exercise -o ../1_session_1_hands_on.ipynb
rm -r tmp.ipynb

jupytext --to notebook 2_session_2_skorch_class_imbalance.py -o tmp.ipynb
nb-filter-cells -i tmp.ipynb -t exercise -o ../2_session_2_skorch_class_imbalance.ipynb
rm -r tmp.ipynb

jupytext --to notebook 3_session_2_sliding_window.py -o tmp.ipynb
nb-filter-cells -i tmp.ipynb -t exercise -o ../3_session_2_sliding_window.ipynb
rm -r tmp.ipynb
