"""Text-cleaning steps, run through a real notebook: cleaning a customer call list."""

import numpy as np
import pandas as pd
import pytest

from app.export import pipeline_script
from app.intent import asked_for
from app.session import DataSession
from app.tools import execute_tool


def run(session, tool, **args):
    result = execute_tool(tool, session, args)
    assert result["success"], result["message"]
    return result


def call_list():
    df = pd.DataFrame({
        "CustomerID": range(1, 11),
        "First_Name": ["Frodo", "Abed", "Walter", "Dwight", "Jon", "Ron", "Jeff", "Sherlock", "Gandalf", "Peter"],
        "Last_Name": ["Baggins", "Nadir", "/White", "Schrute", "Snow", "Swanson", "Winger", "Holmes", "...", "Parker_"],
        "Phone_Number": ["123-545-5421", "123/643/9775", "7066950392", "123-543-2345", "876|678|3469",
                         "304-762-2467", None, "876|678|3469", "N/a", "(123) 456 7890"],
        "Address": ["123 Shire Lane, Shire", "93 West Main Street", "298 Drugs Driveway", "980 Paper Avenue, Pennsylvania, 18503",
                    "123 Dragons Road", "768 City Parkway", "1209 South Street", "98 Clue Drive", "123 Middle Earth", "25th Main Street, New York"],
        "Paying Customer": ["Yes", "No", "N", "Y", "Y", "Yes", "N/a", "No", "Yes", "Y"],
        "Do_Not_Contact": ["No", "Yes", "No", "Y", "Y", "No", "No", "N", "No", "No"],
        "Not_Useful_Column": [True] * 10,
    })
    return pd.concat([df, df.iloc[:2]], ignore_index=True)  # 2 duplicate rows


@pytest.fixture
def session():
    s = DataSession(call_list(), "Customer Call List.xlsx")
    run(s, "clean_only")
    return s


def test_strip_characters(session):
    run(session, "strip_characters", column="Last_Name", characters="._/")
    assert list(session.df["Last_Name"][[2, 8, 9]]) == ["White", "", "Parker"]


def test_format_phone_numbers(session):
    result = run(session, "format_text", column="Phone_Number", keep="digits", pattern="###-###-####")
    phones = session.df["Phone_Number"]
    assert list(phones[[0, 1, 4, 9]]) == ["123-545-5421", "123-643-9775", "876-678-3469", "123-456-7890"]
    assert phones.isna()[[6, 8]].all()  # None, and "N/a" (no digits left) are blanks now
    assert "nothing left and became blanks" in result["message"]


def test_split_address(session):
    run(session, "split_column", column="Address", into="Street_Address, State, Zip_Code")
    row = session.df.loc[3]
    assert (row["Street_Address"], row["State"], row["Zip_Code"]) == ("980 Paper Avenue", "Pennsylvania", "18503")
    assert pd.isna(session.df.loc[1, "State"]) and "Address" in session.df  # the original stays


def test_replace_values(session):
    run(session, "replace_values", column="Paying Customer", find="Yes", replace_with="Y")
    run(session, "replace_values", column="Paying Customer", find="No", replace_with="N")
    assert set(session.df["Paying Customer"].dropna()) == {"Y", "N", "N/a"}
    run(session, "replace_values", find="N/a", replace_with="")  # every text column, to a blank
    assert session.df["Paying Customer"].isna().sum() == 1
    # whole-value matching doesn't touch "No" inside other words
    assert "Snow" in set(session.df["Last_Name"])


def test_replace_part_of_text(session):
    run(session, "replace_values", column="Address", find="Street", replace_with="St", match="part of text")
    assert session.df.loc[1, "Address"] == "93 West Main St"


def test_the_whole_notebook(session, tmp_path, monkeypatch):
    """The call-list notebook, cell by cell, in the app; then its exported script reproduces it."""
    run(session, "remove_duplicates")                                                      # [2]
    run(session, "remove_columns", columns=["Not_Useful_Column"])                          # [2]
    run(session, "strip_characters", column="Last_Name", characters="._/")                 # [2]
    run(session, "format_text", column="Phone_Number", keep="digits", pattern="###-###-####")  # [3]
    run(session, "split_column", column="Address", into="Street_Address, State, Zip_Code")  # [4]
    for col in ("Paying Customer", "Do_Not_Contact"):                                     # [4]-[5]
        run(session, "replace_values", column=col, find="Yes", replace_with="Y")
        run(session, "replace_values", column=col, find="No", replace_with="N")
    run(session, "replace_values", find="N/a", replace_with="")                            # [5]
    run(session, "drop_rows_where", conditions='Do_Not_Contact == "Y"')                    # [6]
    run(session, "drop_missing_rows", columns=["Phone_Number"])                            # [7]

    df = session.df
    # 10 customers - 3 who said Yes/Y to "do not contact" - 2 with no usable phone (None, "N/a")
    assert len(df) == 5
    assert (df["Do_Not_Contact"] != "Y").all() and df["Phone_Number"].str.fullmatch(r"\d{3}-\d{3}-\d{4}").all()

    monkeypatch.chdir(tmp_path)
    session.original_df.to_csv("calls.csv", index=False)
    scope = {"__name__": "pipeline"}
    exec(compile(pipeline_script(session), "pipeline.py", "exec"), scope)
    pd.testing.assert_frame_equal(scope["main"]("calls.csv"), session.df, check_dtype=False)


def test_also_replays_on_test_rows_in_model_mode():
    df = pd.DataFrame({"phone": ["(123) 456 7890", "N/a"] * 20, "y": [0, 1] * 20})
    s = DataSession(df, "d.csv")
    run(s, "select_target", column="y")
    run(s, "format_text", column="phone", pattern="###-###-####")
    assert set(s.prepare(s.test_df)["phone"].dropna()) == {"123-456-7890"}


@pytest.mark.parametrize("tool, message", [
    ("replace_values", "change Yes to Y"), ("strip_characters", "remove the underscores"),
    ("format_text", "format the phone numbers"), ("split_column", "split the address"),
])
def test_only_when_asked(tool, message):
    assert asked_for(tool, message) and not asked_for(tool, "what should I do next?")
