from transformers import pipeline


classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")
results = classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)
print(results)