# Custom-Diet-Recommendation-System
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product
import random
BMR_CONSTANTS = {
    'MALE': {'base': 66.47, 'weight': 13.75, 'height': 5.003, 'age': 6.755},
    'FEMALE': {'base': 655.1, 'weight': 9.563, 'height': 1.850, 'age': 4.676}
}
ACTIVITY_FACTORS = {
    'sedentary': 1.2,
    'light': 1.375,
    'moderate': 1.55,
    'heavy': 1.725,
    'extreme': 1.9
}
MACRO_RATIOS = {
    'weight_loss': {'protein': 0.40, 'carb': 0.35, 'fat': 0.25}, # High Protein, Lower Carbs
    'muscle_gain': {'protein': 0.35, 'carb': 0.45, 'fat': 0.20}, # High Carbs/Protein
    'maintenance': {'protein': 0.25, 'carb': 0.50, 'fat': 0.25}  # Balanced
}
CAL_PER_GRAM = {'protein': 4, 'carb': 4, 'fat': 9}
FOOD_DATABASE_DICT = {
    'name': [
        'Grilled Salmon w/ Asparagus', 'Chicken Breast Salad', 'Vegan Lentil Soup', 
        'Spinach Omelette', 'Brown Rice & Beans', 'Greek Yogurt w/ Berries',
        'Protein Smoothie', 'Tuna Sandwich (Whole Wheat)', 'Quinoa Stir Fry',
        'Beef Steak & Potatoes', 'Pancakes (Protein)', 'Veggie Scramble', 
        'Turkey Wrap', 'Tofu & Vegetable Curry', 'Overnight Oats',
        'Chicken Thigh & Greens', 'Avocado Toast', 'Cottage Cheese', 
        'Shrimp Scampi', 'Black Bean Burger', 'Miso Soup',
        'Egg Salad Sandwich', 'Roast Lamb w/ Mint', 'Chili con Carne',
        'Fruit Salad', 'Edamame Snack', 'Whey Protein Shake',
        'Whole Wheat Pita & Hummus', 'Mushroom Risotto', 'Cauliflower Pizza'
    ],
    'type': [
        'Dinner', 'Lunch', 'Lunch', 'Breakfast', 'Dinner', 'Breakfast', 
        'Breakfast', 'Lunch', 'Dinner', 'Dinner', 'Breakfast', 'Breakfast', 
        'Lunch', 'Dinner', 'Breakfast', 'Dinner', 'Breakfast', 'Snack', 
        'Dinner', 'Lunch', 'Snack', 'Lunch', 'Dinner', 'Dinner',
        'Snack', 'Snack', 'Snack', 'Snack', 'Dinner', 'Lunch'
    ],
    'calories': [
        550, 400, 350, 300, 500, 250, 320, 450, 480, 700, 420, 310, 
        380, 410, 300, 600, 350, 180, 650, 420, 100, 460, 750, 580,
        150, 120, 160, 280, 520, 490
    ],
    'protein_g': [
        45, 40, 20, 25, 18, 15, 30, 30, 22, 55, 35, 20, 
        32, 28, 10, 48, 12, 20, 50, 25, 5, 30, 60, 45, 
        3, 11, 35, 10, 15, 30
    ],
    'carb_g': [
        10, 20, 40, 15, 70, 35, 30, 45, 60, 30, 50, 40, 
        35, 50, 50, 25, 35, 8, 40, 55, 15, 35, 15, 45,
        35, 10, 5, 35, 70, 40
    ],
    'fat_g': [
        35, 15, 10, 15, 15, 5, 10, 20, 18, 35, 10, 10, 
        10, 12, 8, 30, 15, 8, 35, 15, 2, 25, 40, 25,
        1, 1, 1, 10, 20, 25
    ],
    'tags': [
        'fish,low-carb', 'high-protein', 'vegan,high-fiber', 'keto,low-carb', 'vegetarian', 'low-fat',
        'high-protein', 'lunch', 'dinner', 'high-protein', 'breakfast', 'vegetarian',
        'lunch', 'vegan', 'breakfast', 'high-protein', 'breakfast', 'snack',
        'seafood', 'vegetarian', 'low-cal', 'lunch', 'high-protein', 'dinner',
        'low-cal', 'vegan', 'high-protein', 'vegetarian', 'dinner', 'low-carb'
    ]
}

class DietRecommender:

    def __init__(self, food_data_dict):
        self.food_df = pd.DataFrame(food_data_dict)
        self.user_profile = {}
        self.target_nutrition = {}


    def calculate_bmr(self, gender, weight_kg, height_cm, age_years):
        const = BMR_CONSTANTS[gender.upper()]
        bmr = (const['base'] + 
               (const['weight'] * weight_kg) + 
               (const['height'] * height_cm) - 
               (const['age'] * age_years))
        return bmr

    def calculate_tdee(self, bmr, activity_level):
        factor = ACTIVITY_FACTORS.get(activity_level.lower(), 1.2)
        return bmr * factor

    def set_target_goals(self, gender, weight_kg, height_cm, age_years, activity_level, goal):
        bmr = self.calculate_bmr(gender, weight_kg, height_cm, age_years)
        tdee = self.calculate_tdee(bmr, activity_level)
        if goal == 'weight_loss':
            target_cal = tdee - 500  # Safe deficit
        elif goal == 'muscle_gain':
            target_cal = tdee + 300  # Caloric surplus
        else: # maintenance
            target_cal = tdee
        
        ratios = MACRO_RATIOS.get(goal, MACRO_RATIOS['maintenance'])

        self.target_nutrition = {
            'target_cal': target_cal,
            'target_protein_g': (target_cal * ratios['protein']) / CAL_PER_GRAM['protein'],
            'target_carb_g': (target_cal * ratios['carb']) / CAL_PER_GRAM['carb'],
            'target_fat_g': (target_cal * ratios['fat']) / CAL_PER_GRAM['fat']
        }
        
        self.user_profile = {
            'gender': gender, 'weight': weight_kg, 'height': height_cm, 'age': age_years, 
            'activity': activity_level, 'goal': goal, 'bmr': bmr, 'tdee': tdee
        }
        
        print("\n[INFO] Target Nutrition Calculated:")
        print(f"       Calories: {int(target_cal)} kcal")
        print(f"       Protein: {int(self.target_nutrition['target_protein_g'])}g")
        print(f"       Carbs: {int(self.target_nutrition['target_carb_g'])}g")
        print(f"       Fat: {int(self.target_nutrition['target_fat_g'])}g")

    def get_meal_options(self, meal_type, diet_tag=None):
        options = self.food_df[self.food_df['type'] == meal_type].copy()
        if diet_tag:
            options = options[options['tags'].str.contains(diet_tag, case=False, na=False)]
        return options.reset_index(drop=True)

    def find_best_plan(self, diet_tag=None, num_meals=3):
        if not self.target_nutrition:
            raise ValueError("Target goals must be set before recommending a plan.")

        target_vector = np.array([
            self.target_nutrition['target_protein_g'],
            self.target_nutrition['target_carb_g'],
            self.target_nutrition['target_fat_g']
        ]).reshape(1, -1)
        
        b_options = self.get_meal_options('Breakfast', diet_tag)
        l_options = self.get_meal_options('Lunch', diet_tag)
        d_options = self.get_meal_options('Dinner', diet_tag)
        
        if b_options.empty or l_options.empty or d_options.empty:
            print("[ERROR] Could not find sufficient meals for the given diet tag.")
            # Fallback to general recommendation if tags are too restrictive
            b_options = self.get_meal_options('Breakfast')
            l_options = self.get_meal_options('Lunch')
            d_options = self.get_meal_options('Dinner')


        best_similarity = -1
        best_plan = None
        
        all_combinations = list(product(b_options.iterrows(), l_options.iterrows(), d_options.iterrows()))
        sample_size = min(len(all_combinations), 5000)
        sampled_combinations = random.sample(all_combinations, sample_size)
        print(f"\n[INFO] Evaluating {sample_size} combinations...")


        for combo in sampled_combinations:
            plan = [item[1] for item in combo] # Extract DataFrame rows
            total_protein = sum(item['protein_g'] for item in plan)
            total_carb = sum(item['carb_g'] for item in plan)
            total_fat = sum(item['fat_g'] for item in plan)
            total_cal = sum(item['calories'] for item in plan)
            cal_diff_percent = abs(total_cal - self.target_nutrition['target_cal']) / self.target_nutrition['target_cal']
            if cal_diff_percent > 0.10: # Only consider plans within 10% of target calories
                continue
            plan_vector = np.array([total_protein, total_carb, total_fat]).reshape(1, -1)
            
            similarity = cosine_similarity(target_vector, plan_vector)[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_plan = {
                    'Breakfast': plan[0]['name'],
                    'Lunch': plan[1]['name'],
                    'Dinner': plan[2]['name'],
                    'Total Calories': total_cal,
                    'Total Protein': total_protein,
                    'Total Carbs': total_carb,
                    'Total Fat': total_fat,
                    'Similarity Score': best_similarity
                }
        
        return best_plan

def main():
    print("==============================================")
    print(" CUSTOM DIET RECOMMENDATION SYSTEM (AI-Based) ")
    print("==============================================")
    
    gender = 'male'
    weight_kg = 80 # kg
    height_cm = 180 
    age_years = 30 
    activity_level = 'moderate' 
    health_goal = 'muscle_gain' 
    dietary_preference = 'high-protein' 

    print("\n[USER PROFILE]:")
    print(f"  Gender: {gender.title()}, Age: {age_years}, Weight: {weight_kg}kg, Height: {height_cm}cm")
    print(f"  Activity: {activity_level.title()}, Goal: {health_goal.title()}")
    print(f"  Preference: {dietary_preference.title()}")
    
    
    recommender = DietRecommender(FOOD_DATABASE_DICT)
    
    try:
        recommender.set_target_goals(gender, weight_kg, height_cm, age_years, activity_level, health_goal)
        
        print("\n[INFO] Searching for optimal meal plan...")
        recommended_plan = recommender.find_best_plan(diet_tag=dietary_preference, num_meals=3)
        
        if recommended_plan:
            print("\n==============================================")
            print("         YOUR PERSONALIZED DIET PLAN          ")
            print("==============================================")
            print(f"  Best Match Score (Cosine Similarity): {recommended_plan['Similarity Score']:.4f}")
            print("\n  --- MEALS ---")
            print(f"  Breakfast: {recommended_plan['Breakfast']}")
            print(f"  Lunch:     {recommended_plan['Lunch']}")
            print(f"  Dinner:    {recommended_plan['Dinner']}")
            
            print("\n  --- NUTRITIONAL SUMMARY ---")
            print(f"  Target Calories: {int(recommender.target_nutrition['target_cal'])} kcal")
            print(f"  Plan Calories:   {recommended_plan['Total Calories']} kcal")
            print("-" * 33)
            print(f"  Macro (g):  | Plan | Target | Ratio Check")
            print(f"  Protein:    | {int(recommended_plan['Total Protein']):<3}| {int(recommender.target_nutrition['target_protein_g']):<5}| {'PASS' if abs(recommended_plan['Total Protein'] - recommender.target_nutrition['target_protein_g']) < 20 else 'CHECK'}")
            print(f"  Carbohydr.: | {int(recommended_plan['Total Carbs']):<3}| {int(recommender.target_nutrition['target_carb_g']):<5}| {'PASS' if abs(recommended_plan['Total Carbs'] - recommender.target_nutrition['target_carb_g']) < 20 else 'CHECK'}")
            print(f"  Fat:        | {int(recommended_plan['Total Fat']):<3}| {int(recommender.target_nutrition['target_fat_g']):<5}| {'PASS' if abs(recommended_plan['Total Fat'] - recommender.target_nutrition['target_fat_g']) < 20 else 'CHECK'}")
            print("==============================================")
        else:
            print("\n[ERROR] Could not find a meal plan that meets the caloric tolerance. Try adjusting the goal or preferences.")

    except ValueError as e:
        print(f"\n[CRITICAL ERROR] {e}")


if __name__ == "__main__":
    main()
