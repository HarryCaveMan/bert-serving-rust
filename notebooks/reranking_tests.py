import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from json import dumps as to_json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('LiYuan/amazon-query-product-ranking')
model = AutoModelForSequenceClassification.from_pretrained('LiYuan/amazon-query-product-ranking').to(DEVICE)

passages = ["15x7 universal wheel","17 inch wleel","4x100 15x7 wheel","wheel","Washington D.C.","mouse trap","model plane","cheeseburgers might be unhealthy, but they sure are tasty!"]
query = ["4x100 wheel 15in"]*len(passages)
inputs = tokenizer(query, passages, padding=True, truncation=True, return_tensors='pt', max_length=512).to(DEVICE)

with torch.no_grad():
    logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    probs = torch.sum(probs[:,:logit_index_threshold],1)
rankings = torch.argsort(probs,dim=0,descending=True).tolist()
ranked_metadata = [{"score":float(probs[i]),"text":passages[i]} for i in rankings]

print(f"""RANKINGS PASSAGE INDEX
All Probs:
  {probs}
Rankings:
  {rankings}
""")

ranked_metadata = [{"score":float(probs[i]),"text":passages[i]} for i in rankings]

print(f"""RANKINGS FULL METADATA

{to_json(ranked_metadata,indent=2)}