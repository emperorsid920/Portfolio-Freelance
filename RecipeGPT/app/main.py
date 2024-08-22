# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.



from fastapi import FastAPI

from pydantic import BaseModel

from app.gpt_model import generate_recipe

from app.meal_planner import generate_meal_plan

app = FastAPI()

class UserInput(BaseModel):
    preferences: str
    dietary_restrictions: str
    meal_type: str
    servings: int

@app.post("/generate-recipe")
def generate_recipe_endpoint(user_input: UserInput):
    recipe = generate_recipe(user_input)
    return {"recipe": recipe}

@app.post("/generate-meal-plan")
def generate_meal_plan_endpoint(user_input: UserInput):
    meal_plan = generate_meal_plan(user_input)
    return {"meal_plan": meal_plan}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
