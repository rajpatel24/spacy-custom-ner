# Custom Named Entity Recognition using SpaCy

***

The purpose of this project is to demonstrate the Custom Named Entity Recognition using SpaCy.

***

I have used following dataset for this project:

`https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus`




## Installation Guidelines


- Clone the repository


- Create virtual environment:

    ```
    pip install virtualenv
    ```
    
    ```
    virtualenv -p python3.6 venv
    ```

- Activate the virtual environment:
  
    ```
    source venv/bin/activate
    ```

- Activate the virtual environment:

    ```
    source mybot/venv/bin/activate
    ```
-  Install dependencies:
    
    ```
    pip install -r requirements.txt
    ```

## Project Guidelines


1. Convert your dataset .csv file to .tsv file


2. Convert .tsv file to .json using "tsv_to_json.py"


3. Convert .json to .spacy (SpaCy supported format) using "json_to_spacy.py"


Now, you are all set to do further processing on data.
***

Inspired by:

`https://towardsdatascience.com/custom-named-entity-recognition-using-spacy-7140ebbb3718`
