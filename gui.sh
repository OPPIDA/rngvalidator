#!/bin/bash
# Author : Florian Picca <florian.picca@oppida.fr>
# Date : January 2020

# set working directory
cd "$(dirname "$0")"

# launch main program
python3 -m src.gui.App "$@"