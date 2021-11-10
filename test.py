"""
Testing the trained model
"""

import spacy

text = "Hillary Clinton said that India will not just become a regional power, but it will become a global power by 2025."

nlp = spacy.load("Output")
doc = nlp(text)

print("\n\n ---------------------- \n\n", doc)

for ent in doc.ents:
    print("\n\n --->>>>", ent, ent.label_)
