from pathlib import Path
from functools import partial
from argparse import ArgumentParser
from polars import scan_csv, LazyFrame, Schema, UInt32, UInt16, Float32, col, concat

PROCESSED_DATA_PATH: Path = Path("data/Processed")
MORTALITY_SOURCES: dict[str, Path] = {
    "hepvu": Path("Mortality/HepVu_mortality_rates_final.csv"),
    "cdc-wonder": Path("Mortality/Mortality_final_rates.csv"),
}


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


def _column_name_to_int(name: str) -> int:
    return int(name.split(maxsplit=1)[0])


def _get_years_from_schema(schema: Schema) -> list[int]:
    years: list[int] = [
        _column_name_to_int(name) for name in schema.names() if name != "FIPS"
    ]
    return years


def _compute_id_from_fips(fips: int, year: int) -> int:
    return fips * (10**4) + year


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
                concat(
                    [
                        col("FIPS").map_elements(
                            partial(_compute_id_from_fips, year=year)
                        )
                        for year in years
                    ]
                )
                .cast(UInt32)
                .alias("id"),
                concat([col("FIPS") for year in years]).cast(UInt16).alias("fips"),
                concat([col(f"{year} MR") for year in years])
                .cast(Float32)
                .alias("opioid_related_mortality"),
            ]
        )
        .with_columns(year=(col("id") % (10**4)).cast(UInt16))
        .select(["id", "fips", "year", "opioid_related_mortality"])
    )

    return mortality_data


def main() -> None:
    args = parser.parse_args()
    mortality_data = load_mortality_data(args.mortality_source)
    print(mortality_data.head(5).collect())


if __name__ == "__main__":
    main()
