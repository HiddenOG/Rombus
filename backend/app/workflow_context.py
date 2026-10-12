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

# The stage a piece of work belongs to. The tracker follows the work (it shows how far you've got,
# not a gate): running one of these moves it forward to that stage, never back.
STAGE_OF_TOOL = {
    **{t: "EXPLORE" for t in ("plot_histogram", "plot_boxplot", "plot_correlation", "plot_scatterplot",
                              "plot_missing", "show_unique_values", "filter_rows", "group_summary", "pivot_table")},
    **{t: "MODEL" for t in ("train_models", "auto_ml", "custom_models")},
    **{t: "INTERPRET" for t in ("explain_model", "final_exam")},
    **{t: "COMMUNICATE" for t in ("save_model",)},
}


def follow_stage(current: str, tool: str) -> str:
    """The stage after running `tool`: the later of where you are and where that work belongs."""
    reached = STAGE_OF_TOOL.get(tool)
    if not reached or current not in STAGES or STAGES.index(reached) <= STAGES.index(current):
        return current
    return reached
