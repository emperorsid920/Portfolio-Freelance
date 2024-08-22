import requests

def get_nutritional_info(ingredient):
    # Example using USDA API
    api_key = "uLt7ldvRIRn4uKyza5Rwj0HVVSbafFF4HPW5nuyl"
    url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={ingredient}&api_key={api_key}"
    response = requests.get(url).json()
    return response['foods'][0]  # Adjust based on actual API response

def generate_meal_plan(user_input):
    meal_plan = [
        "Breakfast: Oatmeal with fruits",
        "Lunch: Grilled chicken salad",
        "Dinner: Stir-fried veggies with tofu"
    ]

    # Example of adding nutritional info
    nutritional_info = {}
    for item in meal_plan:
        ingredients = extract_ingredients(item)  # Implement this function to extract ingredients
        for ingredient in ingredients:
            nutritional_info[ingredient] = get_nutritional_info(ingredient)

    return {"meal_plan": meal_plan, "nutritional_info": nutritional_info}
