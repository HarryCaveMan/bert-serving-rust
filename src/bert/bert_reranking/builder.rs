// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
// Copyright 2019-2020 Guillaume Becquin
// Copyright 2020 Maarten van Gompel
// Copyright 2024 Harris Joseph
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//
//
//
// # query/result reranking (via n-label paired sequence classification)
// More generic sequence classification pipeline, works with multiple models (Bert, Roberta)
// Private module used mostly by the bert_reranking::model module. Builder for bert_reranking_model::RerankingModel

use std::path::{PathBuf};
use serde::{Deserialize};
use tch::Device;
use rust_bert::resources::{LocalResource};
use rust_bert::pipelines::common::{ModelType,ModelResource};
use rust_bert::pipelines::sequence_classification::{SequenceClassificationConfig};
use rust_bert::{Config,RustBertError};
use super::bert_reranking_model::{RerankingModel};

/// # SequenceClassification Model Builder
///
/// Allows the user to build a model from standard Sentence-Transformer files
/// (configuration and weights).
pub struct RerankingBuilder<T> {
    device: Device,
    inner: T,
}

impl<T> RerankingBuilder<T> {
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
}

pub struct Local {
    model_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    model_type: ModelType,
}

impl Config for ModelConfig {}

impl RerankingBuilder<Local> {
    pub fn local<P: Into<PathBuf>>(model_dir: P) -> Self {
        Self {
            device: Device::cuda_if_available(),
            inner: Local {
                model_dir: model_dir.into(),
            },
        }
    }

    pub fn create_model(self) -> Result<RerankingModel, RustBertError> {
        let model_dir = self.inner.model_dir;
        let config_resource = model_dir.join("config.json");
        let transformer_type = ModelConfig::from_file(&config_resource).model_type;
        let local_resource = LocalResource{local_path:model_dir.join("rust_model.ot")};
        let (tokenizer_vocab, tokenizer_merges) = match transformer_type {
            ModelType::Bert | ModelType::DistilBert => (model_dir.join("vocab.txt"), None),
            ModelType::Roberta => (
                model_dir.join("vocab.json"),
                Some(model_dir.join("merges.txt")),
            ),
            ModelType::Albert => (model_dir.join("spiece.model"), None),
            ModelType::T5 => (model_dir.join("spiece.model"), None),
            _ => {
                return Err(RustBertError::InvalidConfigurationError(format!(
                    "Unsupported transformer model {transformer_type:?} for Sentence Embeddings",
                )));
            }
        };
        let config: SequenceClassificationConfig = SequenceClassificationConfig {
            model_type: transformer_type,
            model_resource: ModelResource::Torch(Box::new(local_resource)),
            config_resource: config_resource.into(),
            vocab_resource: tokenizer_vocab.into(),
            merges_resource: tokenizer_merges.map(|r| r.into()),
            lower_case: false,
            strip_accents: None,
            add_prefix_space: None,
            device: self.device,
            kind: None
        };
        RerankingModel::new(config)
    }
}