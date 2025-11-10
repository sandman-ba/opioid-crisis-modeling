### 11/06/25, EB: Here I am cleaning the CDC opioid prescription dispensing rate data file so that it's in wide-format with columns 'FIPS', '20XX Disp Rate' for the years 2019-2023.
### Then I combine this with Andrew's dispensing rate dataset, which has data from 2014-2020. But, the rates in 2019 and 2020 were revised, so I drop those years from Andrew's dataset before merging.
### Then I save the cleaned dataset to 'EB_Urbanicity/Data/Prescription_dispensing_rates.csv' for later use in model_utils.py.

import polars as pl

CDC_DISPENSING_DATA = "EB_Urbanicity/Data/CDC_County_Opioid_Dispensing_Rates.csv"
DEAS_DISPENSING_DATA = "EB_Urbanicity/Data/deas_opioid_dispensing_rates.csv"

def CDC_dispensing_data_cleaning(
    cdc_dispensing_path: str = CDC_DISPENSING_DATA
) -> pl.DataFrame:
    
    # === Load the long-format dataset and keep only necessary columns ===
    df_long = (
        pl.read_csv(cdc_dispensing_path)
        .select([
            pl.col("STATE_COUNTY_FIP_U").alias("FIPS"),
            pl.col("YEAR"),
            pl.col("opioid_dispensing_rate")
        ])
        # Replace 'Data unavailable' with 0s, as AD did.
        .with_columns(
            pl.when(pl.col("opioid_dispensing_rate") == "Data unavailable")
            .then(0)
            .otherwise(pl.col("opioid_dispensing_rate"))
            .cast(pl.Float64)
            .alias("opioid_dispensing_rate")
        )
        # Pad FIPS codes with leading zeros
        .with_columns(
            pl.col("FIPS")
            .cast(pl.Utf8)
            .str.zfill(5)
        )
    )

    # === Pivot (unstack) to wide format ===
    df_wide = (
        df_long
        .pivot(
            values="opioid_dispensing_rate",
            index="FIPS",
            on="YEAR"
        )
    )
 
    year_cols = [col for col in df_wide.columns if col != "FIPS" and col[:4] in ["2019","2020","2021","2022","2023"]]
    df_wide = df_wide.select(["FIPS"] + year_cols)

    # Sort columns for consistency
    cols = ["FIPS"] + sorted([col for col in df_wide.columns if col != "FIPS"])
    df_wide = df_wide.select(cols)

    # === Rename year columns to match "20XX DR" format ===
    rename_map = {col: f"{col} DR" for col in df_wide.columns if col != "FIPS"}
    df_wide = df_wide.rename(rename_map)
    
    # print(df_wide.head(10))
    
    return df_wide


def combine_dispensing_datasets(
    cdc_df: pl.DataFrame,
    deas_dispensing_path: str = DEAS_DISPENSING_DATA
) -> pl.DataFrame:
    
    # === Load Andrew's dispensing dataset ===
    deas_df = (
        pl.read_csv(deas_dispensing_path)
        .with_columns(
            pl.col("FIPS")
            .cast(pl.Utf8)
            .str.zfill(5)
        )
        # Drop 2019 and 2020 columns since CDC revised those rates
        .drop(["2019 DR", "2020 DR"])
    )

    # print(deas_df.head(10))

    # --- Merge on FIPS with full outer join ---
    combined_df = (
        deas_df.join(cdc_df, on="FIPS", how="full", suffix="_cdc")
        .with_columns(
            pl.coalesce(["FIPS", "FIPS_cdc"]).alias("FIPS")
        )
        .drop("FIPS_cdc")
    )

    #######
    # --- Fill missing year columns with 'NA' ---
    # fill_exprs = [
    #     pl.col(c).fill_null("NA").alias(c)
    #     for c in combined_df.columns
    #     if c != "FIPS"
    # ]
    # combined_df = combined_df.with_columns(fill_exprs)

    # --- Sort columns so years appear in order ---
    year_cols = sorted([c for c in combined_df.columns if c != "FIPS"])
    combined_df = combined_df.select(["FIPS"] + year_cols)

    # print("Combined dataset preview:")
    # print(combined_df.head(10))


    return combined_df


def main():
    cdc_df = CDC_dispensing_data_cleaning(cdc_dispensing_path=CDC_DISPENSING_DATA)
    combined_df = combine_dispensing_datasets(cdc_df, deas_dispensing_path=DEAS_DISPENSING_DATA)
    combined_df.write_csv("EB_Urbanicity/Data/Prescription_dispensing_rates.csv")
    # print(combined_df.head(10))

if __name__ == "__main__":
    main()