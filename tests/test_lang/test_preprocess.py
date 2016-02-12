# -*- coding: utf-8 -*-
from gears.lang import preprocess


def test_remove_html_entities():
    """Test if it removes the entities."""
    test_cases = {
        "bloggin&#x27;": "bloggin'",
        "Tumblr &lt;3s": "Tumblr <3s",
        "Facebook -&gt; Twitter": "Facebook -> Twitter"
    }
    for txt in test_cases:
        assert preprocess.replace_html_entities(txt) == test_cases[txt]
