# Bert Named Entity Recognition Service

## Base Path
The NER service path will be set by the `SERVICE` environment variable, but will default to `/bert_ner_service`

## Enable BIO span tagging
Models for NER commonly use `B-{label}` `I-{label}` `O-{label}` prefixes on their tags to help group tokens together into "spans" that all have the same `label`. The default behavior of the NER service is to just tag all sequences at the token level. You can anable span tagging using BIO bu setting the environment variable `BIO_ENABLED`, which can either be `0`(disabled) or `1`(enabled). The `launch-in-docker` script has an arg `bio_enabled` which defaults to `0`.

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