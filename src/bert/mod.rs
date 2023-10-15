
pub mod bert_embedding_model;
pub mod bert_ner;

#[cfg(test)]
mod tests {
    use rust_bert::pipelines::sentence_embeddings::{Embedding};
    use rust_bert::pipelines::ner::{Entity};
    use rust_bert::{RustBertError};
    use bert_embedding_model::{BertEmbeddingModel};
    use bert_ner::model::{BertNERModel};   
    use super::*;

    #[test]
    fn test_embedding_from_file() {
        println!("Testing BERT embedding model file constructor...")
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
        println!("Testing BERT NER model file constructor...")
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
}