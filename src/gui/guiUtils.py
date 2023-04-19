import PySimpleGUI as psg


def horizontalCenter(list):
    res = [psg.Push()]
    for e in list:
        res += [e, psg.Push()]
    return res


def toggleableCombo(*args, **kwargs):
    """
    Make a Combo not move after setting it visible/invisible.
    """
    return psg.Col([[psg.Combo(*args, **kwargs)]], pad=(0, 0))


def toggleableFileBrowse(*args, **kwargs):
    """
    Make a Combo not move after setting it visible/invisible.
    """
    return psg.Col([[psg.FileBrowse(*args, **kwargs)]], pad=(0, 0))
