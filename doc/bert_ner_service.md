# Bert Named Entity Recognition Service

## Base Path
The NER service path will be set by the `SERVICE` environment variable, but will defailt to `/bert_ner_service`

## API
### `GET /ping`
Simple server (not model) healthcheck
- **Response**: 
    - **Content-type**: `text/plain`
    - **Statuses**:{200}
    - **Schema**: Plain text: `Ready!!`
### `POST /predict`
The embedding endpoint
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
        - **embeddings**: The extraxted entities as individual tokens