import pandas as pd

def compute_all_risk_scores(predictions_df, alpha=0.3):
    """
    Compute both equal-weight (expanding mean) and EWMA risk scores
    for multiple error types (AbsError, SqError, RawError).

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain columns ['FIPS', 'Year', 'True', 'Predicted', 'AbsError'].
    alpha : float, optional (default=0.3)
        Smoothing factor for EWMA (higher = more recent years weighted more).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['FIPS', 'Year',
         'AbsError_Risk', 'SqError_Risk', 'RawError_Risk',
         'AbsError_EWMA_Risk', 'SqError_EWMA_Risk', 'RawError_EWMA_Risk']
    """

    df = predictions_df.copy()

    # --- Ensure required columns exist ---
    if 'SqError' not in df.columns:
        df['SqError'] = (df['True'] - df['Predicted']) ** 2
    if 'RawError' not in df.columns:
        df['RawError'] = df['True'] - df['Predicted']

    # --- Ensure proper ordering ---
    df = df.sort_values(['FIPS', 'Year']).copy()

    errors = ['AbsError', 'SqError', 'RawError']

    for error in errors:
        # Equal-weight expanding mean risk
        df[f'{error}_Risk'] = (
            df.groupby('FIPS')[error]
              .expanding()
              .mean()
              .reset_index(level=0, drop=True)
        )

        # Exponentially weighted moving average (EWMA) risk
        df[f'{error}_EWMA_Risk'] = (
            df.groupby('FIPS')[error]
              .apply(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
              .reset_index(level=0, drop=True)
        )

    # --- Select just the risk columns and identifiers ---
    risk_cols = (
        ['FIPS', 'Year'] +
        [f'{err}_Risk' for err in errors] +
        [f'{err}_EWMA_Risk' for err in errors]
    )

    return df[risk_cols]


