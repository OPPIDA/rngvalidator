#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Florian Picca <florian.picca@oppida.fr>
# Date : September 2022

import PySimpleGUI as psg
from .windows.mainWindow import mainWindow

psg.theme("LightGrey1")

if __name__ == "__main__":

    mainview = mainWindow()

    while True:
        event, values = mainview.window.read()

        if event == psg.WINDOW_CLOSED:
            break
        mainview.handleEvents(event, values)

    mainview.window.close()
