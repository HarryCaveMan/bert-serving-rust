# Bert Embedding Service

## Base Path
The embedding service path will be set by the `SERVICE` environment variable, but will default to `/bert_embedding_service`

## API
### `GET /ping`
Simple server (not model) healthcheck
- **Response**: 
    - **Content-type**: `text/plain`
    - **Statuses**:{200}
    - **Schema**: Plain text: `Ready!!`
### `POST /encode`
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
        - **sentences**: A batch of text sequences (sentences) to encode into vectors (max len 512 tokens)
- **Response**: 
    - **Content-type**: `application/json`
    - **Statuses**:{200,500}
    - **Schema**:
        ```json
        {
            "crid":u32
            "embeddings":[Vec<f32>]
        }
        ```
    - **Schema Params**:
        - **crid**: (Current Request ID) Unsigned int. Not used at all by service aside from being passed through to the response, purely used (or misused, we don't care) to allow async callers to track
        - **embeddings**: The embedding vectors for the sentences