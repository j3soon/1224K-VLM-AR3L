env_clip_prompts = {
    'combat_spider': "combat a spider",
    'milk_cow': "milk a cow",
    'hunt_cow': "hunt a cow",
    'shear_sheep': "shear a sheep",
    'harvest_water': "harvest a water",

    "drawer-open-v2": "The drawer is opened.",
    "sweep-into-v2": "The green cube is in the hole.", 
    "soccer-v2": "The soccer ball is in the goal.", 

    "CartPole-v1": "pole vertically upright on top of the cart.",
    "RingWorld": "Keep moving clockwise around the ring.", 

    
    "RopeFlattenEasy": "The blue rope is straightened.",
    "PassWater": "The container, which holds water, is as close to the red circle as possible without causing too many water droplets to spill.",
    "ClothFoldDiagonal": "The cloth is folded diagonally from top left corner to bottom right corner.",
}

task_prompt = {
    'combat_spider': "combat a spider",
    'milk_cow': "milk a cow",
    'hunt_cow': "hunt a cow",
    'shear_sheep': "shear a sheep",
    'harvest_water': "harvest a water",

    'drawer-open-v2': "to maximize drawer opening",
    'sweep-into-v2': "to minimize the distance between the green cube and the hole",
    'soccer-v2': "to minimize the distance between the soccer ball and the goal",

    'CartPole-v1': "to balance the brown pole on the black cart to be upright",
    "RingWorld": "Keep moving clockwise around the ring.", 

    "RopeFlattenEasy": "to straighten the blue rope",
    "PassWater": "to move the container, which holds water, to be as close to the red circle as possible without causing too many water droplets to spill",
    "ClothFoldDiagonal": "to fold the cloth diagonally from top left corner to bottom right corner",
}

# two label
## one stage
two_label_query_prompt = """
The goal is {}. Is Image 2 more likely to achieve the goal? 
Reply a single line of 1 if yes, otherwise 0.
"""

two_label_env_query_prompt = {}
for env_name, prompt in task_prompt.items():
    two_label_env_query_prompt[env_name] = two_label_query_prompt.format(prompt)

## two stage
two_label_thought_prompt = """
The goal is {}. Is Image 2 more likely to achieve the goal? 
"""

two_label_summary_prompt = """
Based on the text below to the question:
The goal is {}. Is Image 2 more likely to achieve the goal?
{}

Reply a single line of 1 if yes, otherwise 0.
"""

two_label_env_thought_prompt = {}
two_label_env_summary_prompt = {}
for env_name, prompt in task_prompt.items():
    two_label_env_thought_prompt[env_name] = two_label_thought_prompt.format(prompt)
    two_label_env_summary_prompt[env_name] = two_label_summary_prompt.format(prompt, "{}")

# three label
## one stage
query_prompt = """
The goal is {}.
Is the goal better achieved in Image 1 or Image 2? 
Reply a single line of 1 if the goal is better achieved in Image 1, or 2 if it is better achieved in Image 2.
Reply 0 if unsure or there is no difference."""

env_query_prompt = {}
for env_name, prompt in task_prompt.items():
    env_query_prompt[env_name] = query_prompt.format(prompt)

## two stage
thought_prompt = """
The goal is {}. Is the goal better achieved in Image 1 or Image 2? 
"""

CoT_prompt = """
1. What is shown in Image 1?
2. What is shown in Image 2?
3. The goal is {}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?
"""

summary_prompt = """
Based on the text below to the question:
The goal is {}. Is the goal better achieved in Image 1 or Image 2? 
{}

Reply a single line of 1 if the goal is better achieved in Image 1, or 2 if it is better achieved in Image 2.
Reply 0 if the text is unsure or there is no difference.
"""

env_thought_prompt = {}
env_summary_prompt = {}
env_query_CoT_prompt = {}
for env_name, prompt in task_prompt.items():
    env_thought_prompt[env_name] = thought_prompt.format(prompt)
    env_summary_prompt[env_name] = summary_prompt.format(prompt, "{}")
    env_query_CoT_prompt[env_name] = CoT_prompt.format(prompt)

query_triple_prompt = """
The goal is {}.
Which image (Image 1, Image 2, or Image 3) best achieves the goal, and which image least achieves the goal?
Output exactly one line in the format <BEST>,<WORST> using only the image identifiers.  
If unsure for either position, output 0 in that position.  
Examples:  
- 2, 1  
"""

env_query_triple_prompt = {}
for env_name, prompt in task_prompt.items():
    env_query_triple_prompt[env_name] = query_triple_prompt.format(prompt)

phi_free_query_template = """
The goal is {}. Is Image 2 more likely to achieve the goal? 
"""

phi_summary_query_template = """
Based on the text below to the question:
The goal is {}. Is Image 2 more likely to achieve the goal?
{}

Reply a single line of 1 if yes, otherwise 0.
"""

phi_free_query_env_prompts = {}
phi_summary_env_prompts = {}

for env_name, prompt in task_prompt.items():
    phi_free_query_env_prompts[env_name] = phi_free_query_template.format(prompt)
    phi_summary_env_prompts[env_name] = phi_summary_query_template.format(prompt, "{}")