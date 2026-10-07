import json
import traceback
from app.config import settings
from app.session import DataSession
from app.workflow_context import ASEMIC_WORKFLOW, IRIS_CASE_STUDY

# Lazy-loaded Gemini model (avoids import error when SDK not installed, e.g. in tests)
_gemini_model = None


def _get_model():
    """Initialize the Gemini model on first use."""
    global _gemini_model
    if _gemini_model is None:
        import google.generativeai as genai
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        _gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')
    return _gemini_model


AVAILABLE_TOOLS_DOC = """
AVAILABLE TOOLS:
- acquire_iris_data: {} (Loads Iris dataset from UCI repo)
- select_target: {"column": "name"} (Set the target/label column)
- remove_columns: {"columns": ["name1", "name2"]} (Drop columns)
- check_duplicates: {} (Count duplicate rows)
- remove_duplicates: {} (Remove duplicate rows)
- clean_numeric_string: {"column": "name"} (Remove $, commas, etc. and convert to numeric)
- standardize_text: {"column": "name"} (Lowercase, strip whitespace)
- impute_missing: {"column": "name", "strategy": "mean/median/mode"} (Fill missing in ONE column)
- impute_all_missing: {"strategy": "mean/median/mode"} (Fill missing in ALL columns at once — auto-selects mean for numeric, mode for categorical)
- encode_categorical: {"column": "name", "method": "label/onehot"} (Encode ONE categorical column)
- encode_all_categorical: {"method": "label/onehot"} (Encode ALL categorical columns at once, skips target)
- scale_features: {"columns": ["name1", "name2"]} (StandardScaler on selected columns)
- scale_all_features: {} (Scale ALL numeric features at once, skips target)
- log_transform: {"column": "name"} (Apply log1p transformation)
- train_test_split: {"test_size": 0.2, "stratify": true/false} (Split data)
- train_models: {"algorithms": ["rf", "gb", "ridge", "lasso", "lr", "svc", "dt", "knn", "xgb"], "cv_folds": 5}
- auto_ml: {} (Automatically find the best model with hyperparameter tuning)
- predict: {"features": {"col1": val1, "col2": val2}} (Make prediction)
- explain_model: {"algorithm": "rf"} (Generate SHAP explanations)
- save_model: {"algorithm": "rf"} (Save model to pickle file)
- get_data_info: {} (df.info() + df.describe() summary)
- plot_histogram: {"column": "name"} (Histogram data with bins)
- plot_boxplot: {"column": "name"} (Box plot statistics)
- plot_correlation: {} (Correlation matrix for numeric features)
- plot_scatterplot: {"x": "col1", "y": "col2", "hue": "col3"} (Scatter plot data)
- preview_data: {"rows": 5} (Preview first N rows)
- undo: {} (Revert last data transformation)
- transition_stage: {} (Move to next ASEMIC stage)

IMPORTANT: When the user asks to clean, impute, encode, or scale "all columns" or "everything",
prefer the batch tools (impute_all_missing, encode_all_categorical, scale_all_features)
instead of calling single-column tools one at a time.
"""


def build_chat_prompt(session: DataSession, user_message: str) -> str:
    """Build the full prompt for Gemini including all context."""
    state = session.get_summary()
    column_summary = session.get_compact_column_summary()

    prompt = f"""
You are an expert Data Science Copilot following the ASEMIC workflow.

ASEMIC WORKFLOW REFERENCE:
{ASEMIC_WORKFLOW}

CASE STUDY REFERENCE:
{IRIS_CASE_STUDY}

CRITICAL: Use ONLY the exact column names from the COLUMN DATA below. Never invent column names.

CURRENT ASEMIC STAGE: {state['stage']}
DATASET: {state['filename']} ({state['shape'][0]} rows × {state['shape'][1]} columns)
TARGET: {state['target']} | PROBLEM: {state['problem_type']} | SPLIT: {state['has_split']}
TRAINED MODELS: {', '.join(state['trained_models']) if state['trained_models'] else 'None'}
CAN UNDO: {state['can_undo']} ({state['undo_count']} snapshots)
COMPLETED STEPS: {', '.join(state['completed_steps'][-10:]) if state['completed_steps'] else 'None'}

ALL COLUMNS:
{column_summary}

CONVERSATION HISTORY (last messages):
{json.dumps(session.memory[-10:], indent=2)}

USER MESSAGE: "{user_message}"

{AVAILABLE_TOOLS_DOC}

INSTRUCTIONS:
1. Guide the user through ASEMIC workflow stages in order.
2. If the user asks for something, suggest the appropriate tool(s).
3. Use column evidence to make data-driven suggestions (e.g., suggest imputation for columns with high missing ratio, encoding for categorical columns).
4. When suggesting actions, always use EXACT column names from the data.
5. If models are trained, you can suggest predictions or model explanations.
6. Suggest the undo tool if the user wants to revert a step.

RESPONSE FORMAT (respond with ONLY valid JSON, no markdown fences):
{{
    "response": "Your expert advice, explanation, or confirmation message...",
    "suggested_actions": [
        {{"tool": "tool_name", "arguments": {{}}, "explanation": "Why this action..."}}
    ]
}}
"""
    return prompt


async def get_copilot_response(session: DataSession, user_message: str) -> dict:
    """Get a response from Gemini for the chat."""
    prompt = build_chat_prompt(session, user_message)

    try:
        response = _get_model().generate_content(prompt)
        text = response.text

        # Extract JSON from response
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            json_str = text[start:end]
            parsed = json.loads(json_str)

            # Validate structure
            if "response" not in parsed:
                parsed["response"] = "I've analyzed the situation."
            if "suggested_actions" not in parsed:
                parsed["suggested_actions"] = []

            return parsed
        else:
            return {
                "response": text.strip(),
                "suggested_actions": [],
            }

    except json.JSONDecodeError as e:
        print(f"GEMINI JSON PARSE ERROR: {e}")
        print(f"Raw text: {text[:500]}")
        return {
            "response": "I analyzed your data but had trouble formatting my response. Could you rephrase your question?",
            "suggested_actions": [],
        }
    except Exception as e:
        print(f"GEMINI ERROR: {e}")
        print(traceback.format_exc())
        return {
            "response": f"I'm having trouble reasoning right now. (Error: {str(e)})",
            "suggested_actions": [],
        }
