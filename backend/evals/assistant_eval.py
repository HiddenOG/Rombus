"""Does the assistant use the verified library first? Measured with the real AI.

Run from backend/:  python -m evals.assistant_eval path/to/bengaluru_house_prices.csv

Each request has the answers that count as right: a built-in step (by name), a code cell of a kind,
or words only. Every proposed step and every code cell is then RUN on a copy of the notebook, so
"right" also means "it works". Results go to evals/results/assistant-<provider>-<time>.json.
"""

import asyncio
import copy
import json
import sys
import time
from pathlib import Path

import pandas as pd

from app import assistant
from app.config import settings
from app.session import DataSession
from app.tools import execute_tool

PAUSE = 12  # seconds between requests: the prompt is ~3k tokens, and free tiers count tokens per minute

# Steps already done for the cases marked CLEANED (where the notebook would be by then)
CLEANED = [("clean_numeric_string", {"column": "total_sqft"})]

# (request, acceptable answers[, steps done first]): "step:<tool>", "code:<kind>", "words"
CASES = [
    # Plain requests a built-in covers: should be steps, not code
    ("fill the blanks in bath with the median", {"step:impute_missing"}),
    ("fil al the missing vals", {"step:impute_all_missing"}),
    ("drop duplictes", {"step:remove_duplicates"}),
    ("get rid of the society column", {"step:remove_columns"}),
    # The built-in keeps "34 Sq. Meter" as 34; code that converts units is also a right answer
    ("total_sqft has ranges like 1200-1400, make it a number", {"step:clean_numeric_string", "code:change"}),
    ("one hot encode area_type", {"step:encode_categorical"}),
    ("show me the distribution of price", {"step:plot_histogram"}),
    ("average price per location", {"step:group_summary", "step:pivot_table"}),
    ("drop rows where bath > 10", {"step:drop_rows_where"}),
    ("group locations with 10 or fewer rows into other", {"step:group_rare_categories"}),
    ("remove price per sqft outliers more than 1 std from the mean of their location",
     {"step:remove_outliers"}, CLEANED),
    # Before total_sqft is a number, the formula can't work: say so, or clean it first
    ("remove price per sqft outliers beyond 1 std within each location",
     {"words", "step:clean_numeric_string", "code:change"}),
    ("add a column with sqft per bathroom", {"step:add_column"}, CLEANED),
    ("train a random forest and gradient boosting", {"step:train_models"}),
    ("scale everything", {"step:scale_all_features"}),
    ("correlation heatmap", {"step:plot_correlation"}),
    ("rename total_sqft to sqft", {"step:rename_column"}),
    ("show the rows where price > 300 and bath >= 4", {"step:filter_rows"}),
    ("where are the nulls?", {"step:plot_missing"}),
    ("what are the different values in area_type", {"step:show_unique_values"}),
    ("drop rows that have blanks in size", {"step:drop_missing_rows"}),
    # Not in the library: code
    ("drop training rows where total_sqft divided by the number of bedrooms in size is under 300",
     {"code:change"}),
    ("for each location, drop 3-bedroom flats whose price per sqft is below the mean of 2-bedroom flats there",
     {"code:change"}),
    ("box plot of price for each area_type side by side", {"code:look", "step:plot_boxplot"}),
    ("try a ridge regression with alpha 3", {"code:models"}),
    # Questions and traps: no leaky step
    ("what is the target and how many test rows are locked?", {"words"}),
    ("add a price_per_sqft column = price / total_sqft as a feature", {"words"}),
    ("add a price_per_sqft column = price / total_sqft as a feature", {"words"}, CLEANED),
    ("make a feature: the mean price of each location", {"words", "code:look"}, CLEANED),
]


def outcome(result):
    if result.get("failed"):
        return "failed"
    if result.get("step"):
        return f"step:{result['step']['tool']}"
    if result.get("code"):
        return f"code:{result['kind']}"
    return "words"


def runs(session, result):
    """Does what it proposed actually run? (on a copy: the notebook itself never changes)"""
    trial = copy.deepcopy(session)
    if result.get("step"):
        r = execute_tool(result["step"]["tool"], trial, result["step"]["arguments"])
    elif result.get("code"):
        tool = {"look": "custom_view", "change": "custom_code", "models": "custom_models"}[result["kind"]]
        r = execute_tool(tool, trial, {"code": result["code"]})
    else:
        return None
    return bool(r.get("success")), str(r.get("message", ""))[:200]


async def main(csv_path: str):
    settings.ALLOW_CUSTOM_CODE = True
    df = pd.read_csv(csv_path)
    base = DataSession(df, Path(csv_path).name)
    assert execute_tool("select_target", base, {"column": "price"})["success"]

    rows, right, worked, provider = [], 0, 0, None
    for request, expected, *setup in CASES:
        session = copy.deepcopy(base)
        for tool, args in (setup[0] if setup else []):
            assert execute_tool(tool, session, args)["success"]
        start = time.time()
        result = await assistant.assist(session, request)
        got = outcome(result)
        ran = runs(session, result)
        ok = got in expected
        right += ok
        worked += bool(ran and ran[0])
        provider = provider or (f"{result.get('provider')}-{result.get('model')}" if not result.get("failed") else None)
        mark = "OK " if ok else "XX "
        answered_by = result.get("model") or "-"
        state = " (after cleaning)" if setup else ""
        print(f"{mark}{got:28} ran={ran[0] if ran else '-'!s:5} {time.time() - start:5.1f}s {answered_by[:22]:22} {request}{state}")
        if not ok or (ran and not ran[0]):
            print(f"      reply: {result.get('reply', '')[:220]!r}")
            if ran and not ran[0]:
                print(f"      error: {ran[1]}")
        rows.append({"request": request, "expected": sorted(expected), "got": got, "ok": ok,
                     "ran": ran, "model": result.get("model"), "reply": result.get("reply"), "step": result.get("step"), "code": result.get("code")})
        await asyncio.sleep(PAUSE)  # free tiers: stay under the per-minute token limits

    proposed = sum(1 for r in rows if r["ran"] is not None)
    models = {}
    for r in rows:
        models[r["model"]] = models.get(r["model"], 0) + 1
    print("Answered by:", ", ".join(f"{m} ({n})" for m, n in models.items()))
    print(f"\nRight kind of answer: {right}/{len(CASES)}   Proposals that ran: {worked}/{proposed}")
    out = Path(__file__).parent / "results" / f"assistant-{(provider or 'none').replace('/', '_')}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"right": right, "cases": len(CASES), "ran": worked, "proposed": proposed, "rows": rows},
                              indent=2, default=str), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
