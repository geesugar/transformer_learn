from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
result = translator("你好，我是一个中国人。")
print(result)
