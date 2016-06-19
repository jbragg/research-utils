"Module for various sequence tagging related task." ""

import os
from nltk.tag import stanford

ner_tagger = None


def get_stanford_ner(token_list):
    """Return the NER tags."""
    global ner_tagger
    if not ner_tagger:
        print
        ner_tagger = stanford.StanfordNERTagger(
            os.getenv("STANFORD_NER_MODEL"), os.getenv("STANFORD_NER_JAR"))
    return ner_tagger.tag(token_list)
