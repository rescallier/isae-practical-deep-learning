#!/usr/bin/env bash
pip install jupyter_contrib_nbextensions yapf
set -x \
    && jupyter contrib nbextension install --user \
    && jupyter nbextension enable --py --sys-prefix widgetsnbextension \
    && jupyter nbextensions_configurator enable  \
    && jupyter nbextension enable addbefore/main  \
    && jupyter nbextension enable autoscroll/main  \
    && jupyter nbextension enable contrib_nbextensions_help_item/main  \
    && jupyter nbextension enable code_prettify/code_prettify  \
    && jupyter nbextension enable datestamper/main  \
    && jupyter nbextension enable dragdrop/main  \
    && jupyter nbextension enable execute_time/ExecuteTime  \
    && jupyter nbextension enable help_panel/help_panel  \
    && jupyter nbextension enable hide_input/main  \
    && jupyter nbextension enable highlighter/highlighter  \
    && jupyter nbextension enable init_cell/main  \
    && jupyter nbextension enable move_selected_cells/main  \
    && jupyter nbextension enable notify/notify  \
    && jupyter nbextension enable rubberband/main  \
    && jupyter nbextension enable scroll_down/main  \
    && jupyter nbextension enable search-replace/main  \
    && jupyter nbextension enable table_beautifier/main  \
    && jupyter nbextension enable toc2/main  \
    && jupyter nbextension enable toggle_all_line_numbers/main ;
