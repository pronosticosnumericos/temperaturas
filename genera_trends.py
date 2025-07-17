#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera trends.json con acentos bien representados en los nombres de los estados.
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
OUT_JSON    = f"{DATA_DIR}/trends.json"
YEARS       = list(range(1979, 2026))
NAME_COL    = "NOM_ENT"  # Ajusta al nombre real de la columna con el nombre del estado
# ——————————————

# 1) Carga el shapefile forzando UTF‑8 y normaliza los nombres
with fiona.open(SHAPEFILE, encoding="utf-8") as src:
    features = list(src)
    crs      = src.crs

estados = gpd.GeoDataFrame.from_features(features, crs=crs)

# Normaliza la columna de nombre para garantizar forma compuesta (NFC)
estados[NAME_COL] = estados[NAME_COL].apply(
    lambda s: unicodedata.normalize("NFC", s)
)

# 2) Prepara la máscara 2D (lat–lon) usando el primer año
ds0 = xr.open_dataset(f"{DATA_DIR}/tmax.{YEARS[0]}.nc")
if "latitude" in ds0.coords and "longitude" in ds0.coords:
    ds0 = ds0.rename({"latitude": "lat", "longitude": "lon"})
tas0 = ds0["tmax"]
mask2d = regionmask.Regions(
    name="estados_mexico",
    numbers=estados.index.values,
    names=   estados["CVE_ENT"].values,
    outlines=estados.geometry.values
).mask(tas0.isel(time=0))
ds0.close()

# 3) Calcula las medias anuales ponderadas
n_states     = len(estados)
n_years      = len(YEARS)
annual_means = np.zeros((n_states, n_years), dtype=float)

for j, year in enumerate(YEARS):
    print(f"Procesando {year}…")
    ds = xr.open_dataset(f"{DATA_DIR}/tmax.{year}.nc")
    if "latitude" in ds.coords and "longitude" in ds.coords:
        ds = ds.rename({"latitude":"lat","longitude":"lon"})
    tas = ds["tmax"].load()

    # Pesos por latitud
    weights = xr.DataArray(
        np.cos(np.deg2rad(ds["lat"].values)),
        coords={"lat": ds["lat"]},
        dims=["lat"]
    ).broadcast_like(tas).load()

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

# 5) Guarda preservando UTF‑8 y acentos
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(geojson, f, ensure_ascii=False, indent=2)

print("GeoJSON guardado en", OUT_JSON)

