ASEMIC_WORKFLOW = """
ASEMIC Workflow (Acquire, Scrub, Explore, Model, Interpret, Communicate):

1. Acquire: Understand the uploaded dataset and choose the column to predict (the target).
   Choosing the target locks away a test set that stays untouched until the final evaluation.
2. Scrub: Clean the training rows: fill or drop missing values, fix types, encode categories, scale numbers.
3. Explore: Analyze the training rows with charts and statistics to find patterns.
4. Model: Train and compare machine learning models.
5. Interpret: Understand what drives the model's predictions and how good it really is.
6. Communicate: Share findings, predictions and the exported model.
"""

# ASEMIC Stage definitions
STAGES = [
    "ACQUIRE",
    "SCRUB",
    "EXPLORE",
    "MODEL",
    "INTERPRET",
    "COMMUNICATE",
]
