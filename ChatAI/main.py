from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf

# Load pre-trained GPT model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2")

# Chatbot loop
conversation = ""
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting the chatbot.")
        break
    conversation += user_input + " "
    if ";" in user_input:
        # Tokenize the entire conversation
        input_ids = tokenizer.encode(conversation, return_tensors="tf")

        # Generate response
        output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Print response
        print("Chatbot:", response)

        # Reset conversation
        conversation = ""
