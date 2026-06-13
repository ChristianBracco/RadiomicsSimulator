from pathlib import Path
import pandas as pd

def map_outcome(label):

    if label in [1, 3]:
        return 1

    return 0


def load_dataset(
    excel_path: str
):

    excel_path = Path(excel_path)

    if not excel_path.exists():
        raise FileNotFoundError(excel_path)

    df = pd.read_excel(
        excel_path
    )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    print()
    # print("===== COLUMNS =====")

    # for i, c in enumerate(df.columns):

        # print(
            # f"{i+1}: {c}"
        # )

    # print("===================")
    # print()

    if "Label" not in df.columns:

        raise Exception(
            "Label column not found"
        )

    df["BinaryOutcome"] = (
        df["Label"]
        .apply(map_outcome)
    )

    return df