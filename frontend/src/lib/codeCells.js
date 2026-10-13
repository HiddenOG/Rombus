/**
 * The three kinds of code cell, the tool that runs each (the same tools as custom steps), and the
 * starting code for an empty cell of each kind.
 */

export const CODE_KINDS = {
    look: { label: 'Look', tool: 'custom_view', hint: 'Explore: changes are thrown away' },
    change: { label: 'Change data', tool: 'custom_code', hint: 'learn/apply: also replayed on the test rows' },
    models: { label: 'Models', tool: 'custom_models', hint: 'Trained, then graded by the final exam' },
};

export const CODE_TEMPLATES = {
    look: '# `train` holds the training rows (a copy: changes here are thrown away).\n# Set `result` to what you want to see.\nresult = train.head()\n',
    change: `import numpy as np
import pandas as pd

TRAIN_ONLY = False  # True if this only drops or filters training rows (outliers, duplicates)


def learn(train):
    # Look at the TRAINING rows; return what apply() needs (medians, category lists...)
    return {}


def apply(df, learned):
    # Change the rows using only \`learned\`: this also runs on the test rows and on new rows
    return df
`,
    models: `from sklearn.ensemble import RandomForestRegressor

# Your models, as {name: model}. They're trained on the training rows and graded by the final exam.
models = {"forest": RandomForestRegressor(random_state=0)}
`,
};
