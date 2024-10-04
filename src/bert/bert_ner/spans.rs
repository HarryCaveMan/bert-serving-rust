use rust_bert::pipelines::ner::{Entity};
use rust_tokenizers::{Offset,OffsetSize};
use rayon::prelude::*;

pub fn tag_spans_bio(input_passages: &Vec<String>,token_tags: &Vec<Vec<Entity>>) -> Vec<Vec<Entity>> {
    token_tags.par_iter()
    .enumerate()
    .map(|(index,passage_token_tags)| {
        let mut begin: OffsetSize = 0;
        let mut end: OffsetSize = 0;
        let mut scores_total: f64 = 0.0;
        let mut tokens_in_span: u32 = 0;
        let mut current_tag: Option<String> = None;
        let mut passage_tagged_spans: Vec<Entity> = Vec::new();
        for entity in passage_token_tags {
            if entity.label.starts_with("B-") {
                if let Some(tag) = current_tag.take() {
                    passage_tagged_spans.push(
                        Entity {
                            word: input_passages[index][begin as usize..end as usize].to_string(),
                            score: scores_total/tokens_in_span as f64,
                            label: tag,
                            offset: Offset { begin: begin, end: end }
                        }
                    );
                }
                tokens_in_span=1;
                scores_total = entity.score;
                begin = entity.offset.begin;
                end = entity.offset.end;
                current_tag = Some(entity.label[2..].to_string());
            }
            else if entity.label.starts_with("I-") {
                // hack for models that just tag every token as I-{tag}, still joins groups of same `tag`
                // treats a new I-{tag} (not equal to I-{current_tag}) as if it were a B-{tag} and starts a new span.
                if !(current_tag.as_deref() == Some(&entity.label[2..])) {
                    if let Some(tag) = current_tag.take() {
                        passage_tagged_spans.push(
                            Entity {
                                word: input_passages[index][begin as usize..end as usize].to_string(),
                                score: scores_total/tokens_in_span as f64,
                                label: tag,
                                offset: Offset { begin: begin, end: end }
                            }
                        );
                    }
                    tokens_in_span=1;
                    scores_total = entity.score;
                    begin = entity.offset.begin;
                    end = entity.offset.end;
                    current_tag = Some(entity.label[2..].to_string());
                }
                tokens_in_span+=1;
                scores_total+=entity.score;
                end = entity.offset.end;
            }
        }
        // Push the last entity
        if let Some(tag) = current_tag.take() {
            passage_tagged_spans.push(
                Entity {
                    word: input_passages[index][begin as usize..end as usize].to_string(),
                    score: scores_total / tokens_in_span as f64,
                    label: tag,
                    offset: Offset { begin: begin, end: end }
                }
            );
        }
        passage_tagged_spans
    })
    .collect::<Vec<Vec<Entity>>>()
}