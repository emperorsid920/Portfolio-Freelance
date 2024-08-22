from transformers import pipeline

# Initialize the text generation pipeline
generator = pipeline('text-generation', model='distilgpt2')

# Generate text
response = generator("What is the difference between Celsius and Fahrenheit?", max_length=100, num_return_sequences=1)

# Print the response
print(response[0]['generated_text'])
