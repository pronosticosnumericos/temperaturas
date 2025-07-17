#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versión ligera y estable para generar trends.json con nombres de estados normalizados
y sin errores de codificación de acentos.
"""

import unicodedata
import xarray as xr
import geopandas as gpd
import regionmask
import numpy as np
import json
from scipy.stats import linregress

# Rutas
DATA_DIR  = "/home/sig07/brisia_mapas"
SHAPEFILE = f"{DATA_DIR}/dest_2015gw.shp"
OUT_JSON  = f"{DATA_DIR}/trends.json"
YEARS     = list(range(1979, 2026))

# 1) Carga shapefile con encoding Latin-1 y normaliza los nombres
estados = gpd.read_file(SHAPEFILE, encoding="latin1").to_crs(epsg=4326)
NAME_COL = "NOM_ENT"  # ajusta al nombre real de tu columna de estados

# Normaliza cadenas Unicode (acento compuesto) para asegurar acentos correctos
estados[NAME_COL] = estados[NAME_COL].apply(
    lambda s: unicodedata.normalize("NFC", s)
)

# 2) Prepara máscara 2D (lat–lon) usando el primer año
ds0 = xr.open_dataset(f"{DATA_DIR}/tmax.{YEARS[0]}.nc")
if "latitude" in ds0.coords and "longitude" in ds0.coords:
    ds0 = ds0.rename({"latitude": "lat", "longitude": "lon"})
tas0 = ds0["tmax"]
mask2d = regionmask.Regions(
    name="estados_mexico",
    numbers=estados.index.values,
    names=   estados["CVE_ENT"].values,
    outlines=estados.geometry.values
).mask(tas0.isel(time=0))  # dims: lat, lon
ds0.close()

n_states = len(estados)
n_years  = len(YEARS)
annual_means = np.zeros((n_states, n_years), dtype=float)

# 3) Bucle por año para calcular media anual ponderada
for j, year in enumerate(YEARS):
    print(f"Procesando {year}…")
    ds = xr.open_dataset(f"{DATA_DIR}/tmax.{year}.nc")
    if "latitude" in ds.coords and "longitude" in ds.coords:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    tas = ds["tmax"].load()

    # recalcula pesos por latitud para este año
    weights = xr.DataArray(
        np.cos(np.deg2rad(ds["lat"].values)),
        coords={"lat": ds["lat"]},
        dims=["lat"]
    ).broadcast_like(tas).load()

    # media espacial ponderada y luego promedio anual
    for i, idx in enumerate(estados.index):
        sel = (mask2d == idx)
        num = (tas * weights).where(sel).sum(dim=("lat", "lon"))
        den =      weights    .where(sel).sum(dim=("lat", "lon"))
        annual_means[i, j] = (num / den).mean(dim="time").item()

    ds.close()

# 4) Construye el GeoJSON con nombres normalizados
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
                {"year": int(YEARS[k]), "value": float(series[k])}
                for k in range(n_years)
            ]
        }
    })

geojson = {"type": "FeatureCollection", "features": features}

# 5) Guarda el GeoJSON asegurando UTF-8 y preservando acentos
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(geojson, f, ensure_ascii=False, indent=2)

print("GeoJSON guardado en", OUT_JSON)

