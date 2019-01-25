## Imports ##
import re

import numpy as np
import scipy.optimize as optim
import pandas as pd
from IPython.display import display, HTML, Markdown
import ipywidgets as ipw

## Recipe Inversion Algorithm
def estim_quantities(A, b, tol=None):
    # Number of ingredients
    p = A.shape[1]
    
    # Objective function
    def obj(x):
        return np.sum((A@x-b)**2)

    # Ordering constraint matrix
    # each row: c_i^T * x >= 0
    C = np.zeros([p, p])
    # Require that each value is larger than the previous
    for i in range(p):
        for j in range(p):
            if i == j:
                C[i,j] = 1
            elif j == i-1:
                C[i,j] = -1

    # non-negative constraint (0 <= c_i^T *x <= inf)
    lin_const = optim.LinearConstraint(C, 0, np.inf, keep_feasible=False)
    
    # Solve!
    result = optim.minimize(obj, np.zeros(p), method="COBYLA", constraints=lin_const, tol=tol)

    return result

## Recipe Estimation Calculations
def solve_recipe(recipe, nutrient_info):
    nutrition_facts = calculate_nutrition_facts(recipe, nutrient_info)
    # Sorted in decreasing order - have to reverse
    sorted_ingredients = sort_ingredient_list(recipe)[::-1]

    # Convert to correctly sorted arrays
    df = pd.DataFrame(nutrient_info)
    A = np.array(df[sorted_ingredients], dtype=float)

    # Save order and values of nutrients
    nutrients = list(map(str_density_to_total, df.index))
    nutrient_vals = [nutrition_facts[nutrient] for nutrient in nutrients]
    b = np.array(nutrient_vals, dtype=float)

    # Solve
    result = estim_quantities(A, b)
    x = result.x
    
    # Convert back to recipe
    estim_recipe = dict(zip(sorted_ingredients, x))
    
    return estim_recipe


## Ingredients Nutrition Data ##

def parse_fda_db_csv(filename):
    nutrient_name_map = {
        'Calories': 'Energy',
        'Total Fat': 'Total lipid (fat)',
        'Cholesterol': 'Cholesterol',
        'Sodium': 'Sodium, Na',
        'Total Carbohydrate': 'Carbohydrate, by difference',
        'Dietary Fiber': 'Fiber, total dietary',
        'Total Sugars': 'Sugars, total',
        'Protein': 'Protein'
    }
    
    df = pd.read_csv(filename, skiprows=4, encoding='latin1')
    nut_dict = {}
    for name_common, name_db in nutrient_name_map.items():
        filtered_df = df[df['Nutrient']==name_db]
        if len(filtered_df) > 0:
            # Convert from value per 100g to value per 1g
            nut_val = filtered_df['1Value per 100 g'].iloc[0] / 100
            nut_units = filtered_df['Unit'].iloc[0]
            name_with_units = '{} ({}/g)'.format(name_common, nut_units)
            nut_dict[name_with_units] = nut_val
    return nut_dict

# String manipulation helpers
def str_density_to_total(nut_str):
    return re.sub(r'\((.*)\/g\)', r'(\1)', nut_str)

def str_total_to_density(nut_str):
    return re.sub(r'\((.*)\)', r'(\1/g)', nut_str)

## Calculations for FDA label
def calculate_nutrition_facts(recipe, nutrient_info):
    nutrition_facts = {
        str_density_to_total(nutrient): 0.0
        for nutrient in nutrient_info[list(nutrient_info.keys())[0]].keys()
    }
    
    # quantities in grams
    for ingredient, quantity in recipe.items():
        for nutrient, value in nutrient_info[ingredient].items():
            total_nutrient = str_density_to_total(nutrient)
            nutrition_facts[total_nutrient] += quantity * value
            
    return nutrition_facts     

def sort_ingredient_list(recipe):
    # Return list of ingredients in decreasing order of quantity in grams
    return [x[0] for x in reversed(sorted(recipe.items(), key=lambda x: x[1]))]


## Draw FDA Nutrient Table ##

def display_nutrient_table(nutrients, vals):
    display(Markdown(
        "| Nutrient | Value |\n|---|---|\n"
        + '\n'.join([
            "| {} | {:.2f} |".format(nutrient, val)
            for nutrient, val in zip(nutrients, vals)
        ])
    ))
    
def draw_fda_label(ingredients, nutrients, vals):
    inner_tab = ipw.Output()
    with inner_tab:
        display_nutrient_table(nutrients, vals)
    display(ipw.VBox(
        [
            ipw.HTML("<h2>Nutrition Facts</h2>",),
            inner_tab,
            ipw.Box(layout=ipw.Layout(height='5px')),
            ipw.HTML("<b>Ingredients:</b> {}".format(
                ', '.join(ingredients)
            ))

        ],
        layout=ipw.Layout(
            border='1px solid black',
            width='200px',
            align_items='center',
        )
    ))

def gen_fda_label(ingredients, nutrients, vals):
    whole_out = ipw.Output()
    with whole_out:
        draw_fda_label(ingredients, nutrients, vals)
    return whole_out

def gen_fda_label_from_recipe(recipe, nutrient_info):
    ingredient_list = sort_ingredient_list(recipe)
    nutrition_facts = calculate_nutrition_facts(recipe, nutrient_info)
    return gen_fda_label(ingredient_list, nutrition_facts.keys(), nutrition_facts.values())

def draw_fda_label_from_recipe(recipe, nutrient_info):
    ingredient_list = sort_ingredient_list(recipe)
    nutrition_facts = calculate_nutrition_facts(recipe, nutrient_info)
    return draw_fda_label(ingredient_list, nutrition_facts.keys(), nutrition_facts.values())


## Draw Recipe Comparison ##
def draw_recipe_comparison(recipe, estim_recipe):
    ingredients = list(recipe.keys())
    x_true = np.array([recipe[ing] for ing in ingredients])
    x_estim = np.array([estim_recipe[ing] for ing in ingredients])
    individual_abs_errs = np.abs(x_true - x_estim)
    individual_rel_errs = individual_abs_errs / x_true
    
    global_rel_err = np.linalg.norm(x_true-x_estim) / np.linalg.norm(x_true)
    
    data = np.array([x_true, x_estim, individual_abs_errs, 100*individual_rel_errs]).T
    cols = ['True Recipe (g)', 'Estim. Recipe (g)', 'Abs. Error (g)', 'Rel. Error (%)']
    df = pd.DataFrame(data, index=ingredients, columns=cols)
    
    display(HTML("<h3>Recipe Estimation Results</h3>"))
    display(ipw.Box(layout=ipw.Layout(height='5px')))
    display(HTML(df.to_html(float_format=lambda s: '{:.2f}'.format(s))))
    print("Global Relative Error: {:.2e}".format(global_rel_err))

def gen_recipe_comparison(recipe, estim_recipe):
    out = ipw.Output()
    with out:
        draw_recipe_comparison(recipe, estim_recipe)
    return out

## Recipe Widget ##

debug_view = ipw.Output(layout={'border': '1px solid black'})
class RecipeWidget(ipw.VBox):
    def __init__(self, recipe=None):
        if recipe is None:
            self.recipe = example_recipe
        else:
            self.recipe = recipe
            
        self.do_calculations()
        self.init_outputs()
        self.init_layout()
        self.init_logic()
        
    
    def do_calculations(self):
        self.estim_recipe = solve_recipe(self.recipe, nutrient_info)
    
    def init_sliders(self):
        self.ingredients = list(self.recipe.keys())
        self.sliders = [
            ipw.IntSlider(
                description='{} (g)'.format(ingredient),
                value=self.recipe[ingredient],
                min=0, 
                max=150,
                continuous_update=False
            )
            for ingredient in self.ingredients
        ]
        out = ipw.Output()
        with out:
            display(HTML("<h3>Edit Recipe</h3>"))
            
        self.slider_box = ipw.VBox([
            out,
            *self.sliders
        ])
    
    def init_outputs(self):
        self.init_sliders()
        
        self.fda_label = gen_fda_label_from_recipe(self.recipe, nutrient_info)
        self.results_table = gen_recipe_comparison(self.recipe, self.estim_recipe)
    
    def init_layout(self):
        super().__init__([
            ipw.HBox([
                self.fda_label,
                ipw.Box(layout=ipw.Layout(width='15px')),
                ipw.VBox(
                    [
                        self.slider_box,
                        self.results_table
                    ],
                )
            ])
        ], layout=ipw.Layout(height='500px'))
        
    def init_logic(self):
        for slider in self.sliders:
            slider.observe(self.update, names='value')
        
    def update_recipe(self):
        self.recipe = {
            ingredient: self.sliders[self.ingredients.index(ingredient)].value
            for ingredient in self.ingredients
        }
    
    @debug_view.capture(clear_output=True)
    def update(self, *change):
        self.update_recipe()
        self.do_calculations()
        self.redraw_outputs()
    
    def redraw_outputs(self):
        self.fda_label.clear_output(wait=True)
        with self.fda_label:
            draw_fda_label_from_recipe(self.recipe, nutrient_info)
            
        self.results_table.clear_output(wait=True)
        with self.results_table:
            draw_recipe_comparison(self.recipe, self.estim_recipe)
            

#### MAIN ####

ingredients = [
    'Broccoli',
    'Carrots',
    'Onions',
    'Garlic',
    'Tomatoes'
]
nutrient_info = {
    ingredient: parse_fda_db_csv(
        'ingredient_data/{}.csv'.format(
            ingredient.lower()
        )
    )
    for ingredient in ingredients
}

nutrient_info_df = pd.DataFrame(nutrient_info)

# Initial values for widget
example_recipe = {
    'Tomatoes': 100,
    'Onions': 50,
    'Garlic': 30,
    'Broccoli': 50,
    'Carrots': 60
}
