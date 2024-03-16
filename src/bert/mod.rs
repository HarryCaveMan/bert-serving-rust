
pub mod bert_embedding_model;
pub mod bert_ner;
pub mod bert_sequence_classification;
pub mod bert_reranking;

#[cfg(test)]
mod tests {
    use rust_bert::pipelines::sentence_embeddings::{Embedding};
    use rust_bert::pipelines::ner::{Entity};
    use rust_bert::pipelines::sequence_classification::{Label};
    use bert_embedding_model::{BertEmbeddingModel};
    use bert_ner::model::{BertNERModel}; 
    use bert_sequence_classification::model::{BertSequenceClassificationModel};
    use bert_reranking::model::{BertRerankingModel,RankedResults};
    use super::*;

    #[test]
    fn test_embedding_from_file() {
        println!("Testing BERT embedding model file constructor...");
        let model_path: &str = "notebooks/models/all-MiniLM-L12-v2";
        let embedding_model: BertEmbeddingModel = BertEmbeddingModel::new_from_file(model_path).unwrap();
        let sentences: Vec<String> = vec![
            String::from("Hi I'm me"),
            String::from("Hi you're you"),
            String::from("Hi they're them"),
            String::from("Hi we're us"),
            String::from("abcdefghijklmnopqrstuvwxyz now I know my abc's, next time won't you sing with me? 0123456789")
        ];        
        let embeddings: Vec<Embedding> = embedding_model.encode(&sentences).unwrap();
        println!("Embedding test success! vector dimension: [{:?},{:?}]",embeddings.len(),embeddings[0].len());
        assert_eq!(embeddings.len(),5);
        assert_eq!(embeddings[0].len(),384);
    }

    #[test]
    fn test_ner_from_file() {
        println!("Testing BERT NER model file constructor...");
        let model_path: &str = "notebooks/models/bert-large-cased-finetuned-conll03-english";
        let ner_model: BertNERModel = BertNERModel::new_from_file(model_path).unwrap();
        let sentences: Vec<String> = vec![
            String::from("Hi I'm HarryCaveMan From The United States"),
            String::from("Hi I'm Billy Williams from WilliamsBurg Virginia"),
            String::from("President Barack Obama lived in Washington D.C."),
            String::from("abcdefghijklmnopqrstuvwxyz now I know my abc's, next time won't you sing with me? 0123456789")
        ];
        let extractions: Vec<Vec<Entity>> = ner_model.predict(&sentences);
        println!("NER test success! Extractions\n{:?}",extractions);
    }

    #[test]
    fn test_sequence_classification_from_file() {
        println!("Testing BERT NER model file constructor...");
        let model_path: &str = "notebooks/models/finbert";
        let sequence_classification_model: BertSequenceClassificationModel = BertSequenceClassificationModel::new_from_file(model_path).unwrap();
        let sentences: Vec<String> = vec![
            String::from("Quarterly earnings"),
            String::from("Quarterly earnings fell"),
            String::from("Quarterly earnings rose"),
            String::from("Growth outpaced inflation"),
            String::from("Inflation outpaced growth"),
            String::from("abcdefghijklmnopqrstuvwxyz now I know my abc's, next time won't you sing with me? 0123456789")
        ];
        let labels: Vec<Label> = sequence_classification_model.predict(&sentences);
        println!("Sequence Classifier test success! Labels\n{:?}",labels);
    }

    #[test]
    fn test_reranking_from_file() {
        let model_path: &str = "notebooks/models/amazon-query-product-ranking";
        let reranking_model: BertRerankingModel = BertRerankingModel::new_from_file(model_path).unwrap();
        let logit_index_thresh: i64 = 1;
        let queries: Vec<String> = vec![
            String::from("4x100 wheel 15in")
        ];
        let results_set: Vec<Vec<String>> = vec![
            vec![
                String::from("15x7 universal wheel"),
                String::from("17 inch wleel"),
                String::from("4x100 15x7 wheel"),
                String::from("wheel"),
                String::from("Washington D.C."),
                String::from("mouse trap"),
                String::from("model plane"),
                String::from("cheeseburgers might be unhealthy, but they sure are tasty!")
            ]
        ];
        let ranked_results = reranking_model.predict(&queries,&results_set,logit_index_thresh);
        println!("Reranking test success! Rankings:\n{:?}",ranked_results)
    }
}