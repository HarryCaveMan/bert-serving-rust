# Bert Raranking Service

## Base Path
The rearnking service path will be set by the `SERVICE` environment variable, but will default to `/bert_reranking_service`

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
            "crid":u32,
            "queries":[String],
            "results_set":[[String]],
            "logit_index_thresh": i64
        }
        ```
    - **Schema Params**:
        - **crid**: (Current Request ID) Unsigned int. Not used at all by service aside from being passed through to the response, purely used (or misused, we don't care) to allow async callers to track. 
        - **queries**: A batch of text sequences (queries)
        - **results_set**: Array containing results array for each query
        - **logit_index_thresh**: The score of a result will be the sum(softmax(x)) of the logit for all labels below (exclusive) this threshold: (IE if thresh: 2, labels: {0:perfect,1:okay,2:not-good,3:bad}, score would be softmax probability of okay or better)
- **Response**: 
    - **Content-type**: `application/json`
    - **Statuses**:{200,500}
    - **Schema**:
        ```json
        {
            "crid":u32,
            "results":{ 
                    "queries": [String],
                    "results_set": [
                        {
                            "text": String,
                            "score": f64,
                            "rank": u64
                        }
                    ]
            {
        }
        ```
    - **Schema Params**:
        - **crid**: (Current Request ID) Unsigned int. Not used at all by service aside from being passed through to the response, purely used (or misused, we don't care) to allow async callers to track
        - **results**: The ResultSet
            - **queries**: The input queries (same as request)
            - **results_set**: List of results for the queries with the following metadata:
                - **text**: The result text
                - **score**: The softmax probability that the result was above the threshold label
                - **rank**: The rank of this retult among its peers