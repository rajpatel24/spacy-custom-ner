import logging
import json
import pickle


def json_to_spacy(input_file, output_file):
    """
    Convert json file to spaCy format.
    """
    try:
        training_data = []
        lines=[]
        with open(input_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                point = annotation['points'][0]
                labels = annotation['label']
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    entities.append((point['start'], point['end'] + 1 ,label))

            training_data.append((text, {"entities" : entities}))

        print(training_data)

        with open(output_file, 'wb') as fp:
            pickle.dump(training_data, fp)

    except Exception as e:
        logging.exception("Unable to process " + input_file + "\n" + "error = " + str(e))
        return None


if __name__ == '__main__':
    input_file = "Data/ner_corpus_260.json"
    output_file = "Data/ner_corpus_260"
    json_to_spacy(input_file, output_file)
