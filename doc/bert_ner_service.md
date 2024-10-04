# Bert Named Entity Recognition Service

## Base Path
The NER service path will be set by the `SERVICE` environment variable, but will default to `/bert_ner_service`

## Enable Span Tagging
To get full entity spans instead of individual tokens, set the environment variable `NER_SPANS=1`. The `launch-in-docker` script also accepts a parameter `ner_spans` which toggles this environment variable when launching the service container.

## API
### `GET /ping`
Simple server (not model) healthcheck
- **Response**: 
    - **Content-type**: `text/plain`
    - **Statuses**:{200}
    - **Schema**: Plain text: `Ready!!`
### `POST /predict`
The ner endpoint
- **Request**: 
    - **Content-type**: `application/json`
    - **Schema**:
        ```json
        {
            "crid":u32
            "sentences":[String]
        }
        ```
    - **Schema Params**:
        - **crid**: (Current Request ID) Unsigned int. Not used at all by service aside from being passed through to the response, purely used (or misused, we don't care) to allow async callers to track. 
        - **sentences**: A batch of text sequences (sentences) to extract named entities from
- **Response**: 
    - **Content-type**: `application/json`
    - **Statuses**:{200,500}
    - **Schema**:
        ```json
        {
            "crid":u32
            "entities":[
                {
                    "word":String,
                    "score":f64,
                    "label":String,
                    "offset": {
                        "begin":u32,
                        "end":u32
                    },
                }
            ]
        }
        ```
    - **Schema Params**:
        - **crid**: (Current Request ID) Unsigned int. Not used at all by service aside from being passed through to the response, purely used (or misused, we don't care) to allow async callers to track
        - **entities**: The extraxted entities as individual tokens