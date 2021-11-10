from __future__ import unicode_literals, print_function

import pickle
import random
import spacy

from pathlib import Path
from spacy.training import Example
from spacy.util import minibatch, compounding

"""
geo = Geographical Entity
org = Organization
per = Person
gpe = Geopolitical Entity
tim = Time indicator
art = Artifact
eve = Event
nat = Natural Phenomenon
"""


def test_model(model_dir, text):
    """
    Test the trained model
    """
    nlp2 = spacy.load(model_dir)
    doc2 = nlp2(text)
    for ent in doc2.ents:
        print(ent.label_, ent.text)


def train_custom_ner(train_data, labels, model, new_model_name, output_dir, n_iter):
    """
    Setting up the pipeline and entity recognizer, and training the new entity.
    """
    # Step-1: Load the model or create a blank model
    if model is not None:
        nlp = spacy.load(model)  # load existing spacy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # Step-2: If Blank model is being used then add 'ner' to the pipeline else get the 'ner' pipe
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    else:
        ner = nlp.get_pipe('ner')

    # Step-3: Add new entity labels to the named entity recognizer (ner)
    for i in labels:
        ner.add_label(i)   # Add new entity labels to entity recognizer

    # Initializing Optimizer
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.create_optimizer()

    # Step-4: Get names of other pipes to disable them during training to train only NER
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER

        # Step-5: Loop over the examples
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)

                example = []
                # Update the model with iterating each text

                for i in range(len(texts)):
                    doc = nlp.make_doc(texts[i])
                    example.append(Example.from_dict(doc, annotations[i]))

                nlp.update(example, sgd=optimizer, drop=0.35, losses=losses)
            print('Losses', losses)

    # Test the trained model
    test_text = 'India is the fastest growing economy in the world.'
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # Save model
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # Test the saved model
        test_model(output_dir, test_text)


if __name__ == '__main__':
    LABEL = ['I-geo', 'B-geo', 'I-art', 'B-art', 'B-tim', 'B-nat', 'B-eve', 'O', 'I-per', 'I-tim', 'I-nat', 'I-eve',
             'B-per', 'I-org', 'B-gpe', 'B-org', 'I-gpe']

    # Load training data
    with open('Data/ner_corpus_260', 'rb') as fp:
        TRAIN_DATA = pickle.load(fp)

    train_custom_ner(TRAIN_DATA, LABEL, 'en_core_web_sm', 'new_model', 'trained_model/', 10)
