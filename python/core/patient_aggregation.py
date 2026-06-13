import pandas as pd

EXCLUDED = [
    "Patient",
    "Label",
    "Organ",
    "BinaryOutcome"
]

def aggregate_by_patient(
    df: pd.DataFrame
):

    features = [
        c
        for c in df.columns
        if c not in EXCLUDED
    ]

    rows = []

    for patient, group in df.groupby(
        "Patient"
    ):

        row = {

            "Patient": patient,

            "BinaryOutcome":
            int(
                group[
                    "BinaryOutcome"
                ].iloc[0]
            )
        }

        for feature in features:

            row[
                f"{feature}_mean"
            ] = group[
                feature
            ].mean()

            row[
                f"{feature}_max"
            ] = group[
                feature
            ].max()

        rows.append(row)

    return pd.DataFrame(rows)