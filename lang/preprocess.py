# -*- coding: utf-8 -*-
"""Implements various helpers for preprocessing text."""


from nltk.tokenize import casual


def replace_html_entities(txt):
    """Replace the html entities in text with corresponding unicode entities.

    Uses UTF-8 encoding.

    Args:
        txt (str): input string
    """
    return casual._replace_html_entities(txt)
