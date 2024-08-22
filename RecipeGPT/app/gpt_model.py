from transformers import pipeline

model = pipeline('text-generation', model='distilgpt2')

def generate_recipe(user_input):
    prompt = (f"Create a detailed recipe that is {user_input.dietary_restrictions}, uses {user_input.preferences}, "
              f"is suitable for {user_input.meal_type}, and includes preparation time, cooking time, and serving size.")
    result = model(prompt, max_length=250, num_return_sequences=1)
    return result[0]['generated_text']
