import numpy as np

def correlation_pruning(
    df,
    threshold=0.95
):

    corr = (
        df
        .corr()
        .abs()
    )

    upper = corr.where(
        np.triu(
            np.ones(
                corr.shape
            ),
            k=1
        ).astype(bool)
    )

    to_drop = [

        column

        for column
        in upper.columns

        if any(
            upper[column]
            > threshold
        )
    ]

    return (
        df.drop(
            columns=to_drop
        ),
        to_drop
    )