#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera archivos trends_cpc_tmax.json y trends_cpc_tmin.json con las tendencias anuales
de temperatura por estado, preservando correctamente los acentos en los nombres.
"""

import unicodedata
import json
import numpy as np
import xarray as xr
import regionmask
from scipy.stats import linregress
import fiona
import geopandas as gpd

# ——————————————
#  Rutas y parámetros
# ——————————————
DATA_DIR    = "/home/sig07/brisia_mapas"
SHAPEFILE   = f"{DATA_DIR}/dest_2015gw.shp"
YEARS       = list(range(1979, 2026))
NAME_COL    = "NOM_ENT"  # Ajusta al nombre real de tu columna de nombre de estado
VARS        = ["tmin"]
# ——————————————

# 1) Carga el shapefile forzando UTF‑8 y normaliza los nombres
with fiona.open(SHAPEFILE, encoding="utf-8") as src:
    features = list(src)
    crs      = src.crs

estados = gpd.GeoDataFrame.from_features(features, crs=crs)

# Normaliza la columna de nombre (acomodando acentos)
estados[NAME_COL] = estados[NAME_COL].apply(
    lambda s: unicodedata.normalize("NFC", s)
)

# 2) Prepara la máscara 2D (lat–lon) con el primer archivo TMAX de CPC
ds0 = xr.open_dataset(f"{DATA_DIR}/tmin.{YEARS[0]}.nc")
if "latitude" in ds0.coords and "longitude" in ds0.coords:
    ds0 = ds0.rename({"latitude": "lat", "longitude": "lon"})
tas0 = ds0["tmin"]
mask2d = regionmask.Regions(
    name="estados_mexico",
    numbers=estados.index.values,
    names=   estados["CVE_ENT"].values,
    outlines=estados.geometry.values
).mask(tas0.isel(time=0))
ds0.close()

# 3) Para cada variable (tmax, tmin), calcula y guarda el JSON
for var in VARS:
    print(f"\n=== Generando trends_cpc_{var}.json ===")
    n_states     = len(estados)
    n_years      = len(YEARS)
    annual_means = np.zeros((n_states, n_years), dtype=float)

    # Bucle por año
    for j, year in enumerate(YEARS):
        print(f"  Procesando {var.upper()} {year}…", end="\r")
        ds = xr.open_dataset(f"{DATA_DIR}/{var}.{year}.nc")
        if "latitude" in ds.coords and "longitude" in ds.coords:
            ds = ds.rename({"latitude":"lat", "longitude":"lon"})
        tas = ds[var].load()

        # Pesos por latitud
        weights = xr.DataArray(
            np.cos(np.deg2rad(ds["lat"].values)),
            coords={"lat": ds["lat"]},
            dims=["lat"]
        ).broadcast_like(tas).load()

        # Calcula media anual ponderada para cada estado
        for i, idx in enumerate(estados.index):
            sel = (mask2d == idx)
            num = (tas * weights).where(sel).sum(dim=("lat","lon"))
            den =      weights    .where(sel).sum(dim=("lat","lon"))
            annual_means[i, j] = (num/den).mean(dim="time").item()

        ds.close()

    # 4) Construye el GeoJSON
    features_out = []
    for i, row in estados.iterrows():
        series = annual_means[i, :]
        slope  = linregress(YEARS, series).slope

        features_out.append({
            "type": "Feature",
            "geometry": row.geometry.__geo_interface__,
            "properties": {
                "name":   row[NAME_COL],
                "slope":  float(slope),
                "series": [
                    {"year": int(YEARS[k]), "value": float(series[k])}
                    for k in range(n_years)
                ]
            }
        })

    geojson = {"type": "FeatureCollection", "features": features_out}

    # 5) Guarda el JSON con codificación UTF-8
    out_path = f"{DATA_DIR}/trends_cpc_{var}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    print(f"\nGuardado: {out_path}")

