# Bert Sequence Classification Service

## Base Path
The sequence classification service path will be set by the `SERVICE` environment variable, but will default to `/bert_sequence_classification_service`

## API
### `GET /ping`
Simple server (not model) healthcheck
- **Response**: 
    - **Content-type**: `text/plain`
    - **Statuses**:{200}
    - **Schema**: Plain text: `Ready!!`
### `POST /predict`
The classifier endpoint
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
        - **sentences**: A batch of text sequences (sentences) to get labels for
- **Response**: 
    - **Content-type**: `application/json`
    - **Statuses**:{200,500}
    - **Schema**:
        ```json
        {
            "crid":u32
            "labels":[
                { 
                    "text": String,
                    "score": f64,
                    "id": u64,
                    "sentence": u64 }
                {
            ]
        }
        ```
    - **Schema Params**:
        - **crid**: (Current Request ID) Unsigned int. Not used at all by service aside from being passed through to the response, purely used (or misused, we don't care) to allow async callers to track
        - **labels**: The labels for each sentence in the input list