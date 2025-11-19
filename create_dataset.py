from pathlib import Path
from functools import partial
from argparse import ArgumentParser
from polars import (
    scan_csv,
    LazyFrame,
    Schema,
    UInt32,
    UInt16,
    Float32,
    col,
    concat,
    Expr,
)

PROCESSED_DATA_PATH: Path = Path("data/Processed")
MORTALITY_SOURCES: dict[str, Path] = {
    "hepvu": Path("Mortality/HepVu_mortality_rates_final.csv"),
    "cdc-wonder": Path("Mortality/Mortality_final_rates.csv"),
}
PRESCRIPTION_SOURCE: Path = Path("Prescriptions/Prescription_dispensing_rates.csv")


parser: ArgumentParser = ArgumentParser(
    prog="create_dataset", description="Script to create opioid crisis dataset"
)

parser.add_argument(
    "--mortality_source",
    type=str,
    default="cdc-wonder",
    choices=list(MORTALITY_SOURCES),
    help="Specify a source for mortality data. Default: %(default)s.",
)

parser.add_argument(
    "--save_path",
    type=Path,
    default=Path("data/all_data.parquet"),
    help="Specify a save path for the dataset (will be saved in parquet format). Default: %(default)s.",
)


def _column_name_to_int(name: str) -> int:
    return int(name.split(maxsplit=1)[0])


def _get_years_from_schema(schema: Schema) -> list[int]:
    years: list[int] = [
        _column_name_to_int(name) for name in schema.names() if name != "FIPS"
    ]
    return years


def _compute_id_from_fips(fips: int, year: int) -> int:
    return fips * (10**4) + year


def get_id_column(years: list[int]) -> Expr:
    id_column: Expr = (
        concat(
            [
                col("FIPS").map_elements(partial(_compute_id_from_fips, year=year))
                for year in years
            ]
        )
        .cast(UInt32)
        .alias("id")
    )
    return id_column


def get_fips_column(years: list[int]) -> Expr:
    return concat([col("FIPS") for year in years]).cast(UInt16).alias("fips")


def get_year_column() -> Expr:
    return (col("id") % (10**4)).cast(UInt16).alias("year")


def load_mortality_data(source: str) -> LazyFrame:
    """
    Function that loads mortality data from a given source and returns a lazy frame.

    Parameters
    ----------
    source : str, Options = [hepvu, cdc-wonder], Source for mortality data.

    Returns
    -------
    mortality_data : polars.LazyFrame, Schema = {id: int, fips: int, year: int, opioid_related_mortality: float}
    """

    if source not in list(MORTALITY_SOURCES):
        raise ValueError(
            f"Mortality data source {source} not available. Please choose from:\n {(*list(MORTALITY_SOURCES),)}"
        )

    bad_mortality_data: LazyFrame = scan_csv(
        PROCESSED_DATA_PATH / MORTALITY_SOURCES[source]
    )
    years: list[int] = _get_years_from_schema(bad_mortality_data.collect_schema())

    mortality_data: LazyFrame = (
        bad_mortality_data.select(
            [
                get_id_column(years),
                get_fips_column(years),
                concat([col(f"{year} MR") for year in years])
                .cast(Float32)
                .alias("opioid_related_mortality"),
            ]
        )
        .with_columns(get_year_column())
        .select(["id", "fips", "year", "opioid_related_mortality"])
    )

    return mortality_data


def load_prescription_data() -> LazyFrame:
    """
    Function that loads prescription data and returns a lazy frame.

    Returns
    -------
    prescription_data : polars.LazyFrame, Schema = {id: int, fips: int, year: int, prescription_rate: float}
    """

    bad_prescription_data: LazyFrame = scan_csv(
        PROCESSED_DATA_PATH / PRESCRIPTION_SOURCE
    )
    years: list[int] = _get_years_from_schema(bad_prescription_data.collect_schema())
    prescription_data: LazyFrame = (
        bad_prescription_data.select(
            [
                get_id_column(years),
                get_fips_column(years),
                concat([col(f"{year} DR") for year in years])
                .cast(Float32)
                .alias("prescription_rate"),
            ]
        )
        .with_columns(get_year_column())
        .select(["id", "fips", "year", "prescription_rate"])
    )

    return prescription_data


def main() -> None:
    args = parser.parse_args()
    mortality_data = load_mortality_data(args.mortality_source)
    prescription_data = load_prescription_data()
    all_data = mortality_data.join(
        prescription_data.select(["id", "prescription_rate"]), on="id"
    )
    all_data.sink_parquet(args.save_path)
    print(mortality_data.head(5).collect())
    print(prescription_data.head(5).collect())
    print(all_data.head(5).collect())


if __name__ == "__main__":
    main()
