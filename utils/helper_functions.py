#!/usr/bin/python
# -*- coding: utf-8 -*-

# @uthor: Makram Jandar
#   |    |  |  |    \|  |  |      | /  _]    \ 
#   |__  |  |  |  o  )  |  |      |/  [_|  D  )
#   __|  |  |  |   _/|  ~  |_|  |_|    _]    / 
#  /  |  |  :  |  |  |___, | |  | |   [_|    \ 
#  \  `  |     |  |  |     | |  | |     |  .  \
#   \____j\__,_|__|  |____/  |__| |_____|__|\_|
#             © Jupyter Helper Functions & more

""" Several helper functions for interactive use. """
import time, sys
from IPython.core.display import HTML

""" Reloading Jupyter from cell """
def reloadJupyter():
	return HTML("<script>Jupyter.notebook.kernel.restart()</script>")

""" Progress Bar Generator """
def updateProgress(progress):
	# update_progress() : Displays or updates a console progress bar
	# Accepts a float between 0 and 1. Any int will be converted to a float.
	# A value under 0 represents a 'halt'.
	# A value at 1 or bigger represents 100%
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()