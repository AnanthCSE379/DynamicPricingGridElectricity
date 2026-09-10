#!/usr/bin/env python3
"""
extract_tariffs.py
Extracts half-hourly dynamic Time-of-Use (dToU) tariffs from Tariffs.xlsx (Sheet1),
converts Excel serial date timestamps to ISO datetime, and saves to Parquet and CSV.
"""

import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import polars as pl

def excel_serial_to_datetime(serial_val: float) -> datetime:
    """
    Excel serial date: Day 1 is 1900-01-01. Due to the 1900 leap year bug in Excel,
    the epoch baseline is 1899-12-30.
    """
    base_date = datetime(1899, 12, 30)
    return base_date + timedelta(days=serial_val)

def extract_tariffs(xlsx_path: str, output_parquet: str, output_csv: str):
    print(f"[1/3] Reading raw XML from {xlsx_path}...")
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Cannot find {xlsx_path}")

    with zipfile.ZipFile(xlsx_path, "r") as z:
        # Load shared strings
        sst_root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        strings = [
            elem.text or ""
            for elem in sst_root.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
        ]

        # Parse Sheet1
        sheet_root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        rows = sheet_root.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row")
        print(f"[2/3] Processing {len(rows)} rows from Sheet1...")

        records_dt = []
        records_tariff = []

        # Row 0 is header ('TariffDateTime', 'Tariff')
        for r in rows[1:]:
            c_list = r.findall("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c")
            if len(c_list) < 2:
                continue

            # Cell 0: serial date
            v0 = c_list[0].find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
            # Cell 1: tariff string or string index
            c1 = c_list[1]
            t1 = c1.attrib.get("t")
            v1 = c1.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")

            if v0 is not None and v1 is not None and v0.text is not None and v1.text is not None:
                serial_float = float(v0.text)
                dt = excel_serial_to_datetime(serial_float)
                # Round to nearest minute to eliminate floating point imprecision
                dt = dt.replace(microsecond=0)
                if dt.second >= 30:
                    dt += timedelta(seconds=60 - dt.second)
                else:
                    dt -= timedelta(seconds=dt.second)

                tariff_str = strings[int(v1.text)] if t1 == "s" else v1.text
                records_dt.append(dt)
                records_tariff.append(tariff_str.strip())

    df = pl.DataFrame({
        "DateTime": records_dt,
        "Tariff": records_tariff
    }).with_columns([
        pl.col("DateTime").cast(pl.Datetime("ms")),
        pl.col("Tariff").cast(pl.Categorical)
    ]).sort("DateTime")

    # Add numeric price column in pence/kWh as defined in Low Carbon London trial
    # Normal = 11.76 p/kWh, High = 67.20 p/kWh, Low = 3.99 p/kWh
    df = df.with_columns(
        pl.when(pl.col("Tariff") == "High").then(67.20)
        .when(pl.col("Tariff") == "Low").then(3.99)
        .otherwise(11.76)
        .cast(pl.Float32)
        .alias("price_p_per_kwh")
    )

    print(f"[3/3] Saving {df.height} tariff slots ({df['DateTime'].min()} to {df['DateTime'].max()})...")
    df.write_parquet(output_parquet)
    df.write_csv(output_csv)
    print(f"Tariffs successfully written to:\n - {output_parquet}\n - {output_csv}")
    print("\nTariff distribution:")
    print(df["Tariff"].value_counts())

if __name__ == "__main__":
    xlsx_file = "Tariffs.xlsx"
    parquet_out = "tariffs.parquet"
    csv_out = "tariffs.csv"
    extract_tariffs(xlsx_file, parquet_out, csv_out)
