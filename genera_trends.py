#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera los archivos JSON de tendencias para CPC:
  - trends_cpc_tmax_annual.json
  - trends_cpc_tmax_monthly.json
  - trends_cpc_tmin_annual.json
  - trends_cpc_tmin_monthly.json

Cada uno contiene:
  * name: nombre del estado (con acentos normalizados)
  * slope: pendiente de la serie (°C/año)
  * series: lista de {period: "YYYY" o "YYYY-MM", value: media}
"""

import unicodedata
import json
import numpy as np
import pandas as pd
import xarray as xr
import regionmask
from scipy.stats import linregress
import fiona
import geopandas as gpd

# ——————————————
# Parámetros
# ——————————————
DATA_DIR   = "/home/sig07/brisia_mapas"
SHAPEFILE  = f"{DATA_DIR}/dest_2015gw.shp"
YEARS      = list(range(1979, 2026))
NAME_COL   = "NOM_ENT"     # Nombre del campo con el estado
VARS       = ["tmax", "tmin"]
# ——————————————

# 1) Cargo shapefile con UTF-8 y normalizo acentos
with fiona.open(SHAPEFILE, encoding="utf-8") as src:
    feats = list(src); crs = src.crs
estados = gpd.GeoDataFrame.from_features(feats, crs=crs)
estados[NAME_COL] = estados[NAME_COL].apply(lambda s: unicodedata.normalize("NFC", s))

# 2) Abrimos un año para extraer las coords de lat
ds0 = xr.open_dataset(f"{DATA_DIR}/tmax.{YEARS[0]}.nc")
if "latitude" in ds0.coords:
    ds0 = ds0.rename({"latitude":"lat","longitude":"lon"})
latitudes = ds0["lat"]
# máscara 2D para recortes por estado
tas0   = ds0["tmax"]
mask2d = regionmask.Regions(
    name="estados",
    numbers=estados.index.values,
    names=estados["CVE_ENT"].values,
    outlines=estados.geometry.values
).mask(tas0.isel(time=0))
# pesos por latitud (1D) – se broadcast automáticamente sobre lon/time
weights_lat = xr.DataArray(
    np.cos(np.deg2rad(latitudes)),
    coords={"lat": latitudes},
    dims=["lat"]
)
ds0.close()

# 3) Loop sobre variables
for var in VARS:

    # —— Anual —— #
    print(f"\nGenerando trends_cpc_{var}_annual.json")
    n_states = len(estados)
    n_years  = len(YEARS)
    annual_means = np.zeros((n_states, n_years), dtype=float)

    for j, year in enumerate(YEARS):
        print(f"  Procesando {var.upper()} {year} …", end="\r")
        ds  = xr.open_dataset(f"{DATA_DIR}/{var}.{year}.nc")
        if "latitude" in ds.coords:
            ds = ds.rename({"latitude":"lat","longitude":"lon"})
        tas = ds[var].load()  # dims (time,lat,lon)

        # aplicamos pesos por latitud
        weights = weights_lat  # dims (lat), broadcast a (time,lat,lon)

        for i, idx in enumerate(estados.index):
            sel = (mask2d == idx)            # bool (lat,lon)
            num = (tas * weights).where(sel).sum(dim=("lat","lon"))
            den =      weights    .where(sel).sum(dim=("lat","lon"))
            # ahora num/den es (time,) → promediamos en time
            annual_means[i, j] = (num/den).mean(dim="time").item()
        ds.close()

    # hallamos pendiente y construimos GeoJSON
    features = []
    for i, row in estados.iterrows():
        series = annual_means[i, :]
        slope  = linregress(YEARS, series).slope
        features.append({
            "type": "Feature",
            "geometry": row.geometry.__geo_interface__,
            "properties": {
                "name":   row[NAME_COL],
                "slope":  float(slope),
                "series": [
                    {"period": str(YEARS[k]), "value": float(series[k])}
                    for k in range(n_years)
                ]
            }
        })
    out_annual = f"{DATA_DIR}/trends_cpc_{var}_annual.json"
    with open(out_annual, "w", encoding="utf-8") as f:
        json.dump({"type":"FeatureCollection","features":features},
                  f, ensure_ascii=False, indent=2)
    print(f"Guardado: {out_annual}")

    # —— Mensual —— #
    print(f"\nGenerando trends_cpc_{var}_monthly.json")
    # cargamos todos los años juntos
    ds_all = xr.open_mfdataset(
        [f"{DATA_DIR}/{var}.{y}.nc" for y in YEARS],
        combine="by_coords"
    )
    if "latitude" in ds_all.coords:
        ds_all = ds_all.rename({"latitude":"lat","longitude":"lon"})
    tas_all = ds_all[var]  # dims (time,lat,lon)

    # creamos etiquetas "YYYY-MM"
    monthly_period      = tas_all["time"].dt.strftime("%Y-%m")
    monthly_period.name = "period"

    # agrupamos por mes (evita resample)
    monthly = tas_all.groupby(monthly_period).mean(dim="time")

    # renombramos "period"→"time" y convertimos a datetime
    monthly = monthly.rename({"period":"time"})
    monthly["time"] = pd.to_datetime(monthly["time"].values + "-01")

    times         = monthly.time.values
    n_months      = len(times)
    monthly_means = np.zeros((n_states, n_months), dtype=float)

    # iteramos meses y estados
    for j in range(n_months):
        m = monthly.isel(time=j).load()  # dims (lat,lon)
        for i, idx in enumerate(estados.index):
            sel = (mask2d == idx)
            num = (m * weights_lat).where(sel).sum(dim=("lat","lon"))
            den =      weights_lat   .where(sel).sum(dim=("lat","lon"))
            # force compute y extrae el escalar
            arr = (num/den).compute()
            # arr.values es array size 1
            monthly_means[i, j] = arr.values.item()
    ds_all.close()

    # construimos GeoJSON mensual
    features = []
    for i, row in estados.iterrows():
        series = monthly_means[i, :]
        xs     = np.arange(n_months)
        slope_m= linregress(xs, series).slope
        slope_y= slope_m * 12
        features.append({
            "type": "Feature",
            "geometry": row.geometry.__geo_interface__,
            "properties": {
                "name":   row[NAME_COL],
                "slope":  float(slope_y),
                "series": [
                    {"period": str(times[k])[:7], "value": float(series[k])}
                    for k in range(n_months)
                ]
            }
        })
    out_monthly = f"{DATA_DIR}/trends_cpc_{var}_monthly.json"
    with open(out_monthly, "w", encoding="utf-8") as f:
        json.dump({"type":"FeatureCollection","features":features},
                  f, ensure_ascii=False, indent=2)
    print(f"Guardado: {out_monthly}")

print("\n¡Todos los JSON de CPC han sido generados!")

