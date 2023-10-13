
pub mod bert_embedding_model;

#[cfg(test)]
mod tests {
    use rust_bert::pipelines::sentence_embeddings::Embedding;
    use rust_bert::RustBertError;
    use bert_embedding_model::BertEmbeddingModel;
    use tch::Device;
    use super::*;
    #[test]
    fn check_device() {
        let device: Device = Device::cuda_if_available();
        println!("Device is CUDA?\n{:?}",device.is_cuda());
    }

    #[test]
    fn test_embedding_model() {
        let model_path: &str = "notebooks/models/all-MiniLM-L12-v2";
        let embedding_model: Result<BertEmbeddingModel,RustBertError> = BertEmbeddingModel::new_from_file(model_path);
        let sentences: Vec<String> = vec![
            String::from("Hi I'm me"),
            String::from("Hi you're you"),
            String::from("Hi they're them"),
            String::from("Hi we're us"),
            String::from("abcdefghijklmnopqrstuvwxyz now I know my abc's, next time won't you sing with me? 0123456789")
        ];
        match embedding_model{
            Ok(model) => {
                let emebeddings_result: Result<Vec<Embedding>, RustBertError> = model.get_embeddings(&sentences);
                match emebeddings_result {
                    Ok(embeddings) => println!("Success, vector dimension:\n{:?}",embeddings[0].len()),
                    Err(err) => println!("Failed:\n{:?}",err)
                }
            },
            Err(err) => println!("Failed:\n{:?}",err)
        };
    }
}