from transformers import pipeline

generator = pipeline("text-generation")
results = generator("In this course, we will teach you how to")
print(results)

