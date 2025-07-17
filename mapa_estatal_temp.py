#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mapa de temperatura promedio mensual por estado
Requisitos: xarray, geopandas, regionmask, numpy, matplotlib, pandas
"""

import xarray as xr
import geopandas as gpd
import regionmask
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import locale

# Parámetros
locale.setlocale(locale.LC_TIME, "es_ES.UTF-8")   # o "es_MX.UTF-8"
NETCDF_PATH    = "/home/sig07/tmax.2025.nc"
SHAPEFILE_PATH = "/home/sig07/dest_2015gw.shp"
VAR_NAME       = "tmax"
MONTH          = 6  # junio

# 1) Cargar NetCDF y renombrar coords si es necesario
ds = xr.open_dataset(NETCDF_PATH)
if "latitude" in ds.coords and "longitude" in ds.coords:
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
tas = ds[VAR_NAME]  # dims: time, lat, lon

# 2) Cargar shapefile y proyectar a WGS84
estados = gpd.read_file(SHAPEFILE_PATH).to_crs(epsg=4326)
state_ids = estados.index.to_list()

# 3) Máscara 2D usando time=0
regions = regionmask.Regions(
    name="estados_mexico",
    numbers=state_ids,
    names=estados["CVE_ENT"].values,
    outlines=estados.geometry.values
)
mask2d = regions.mask(tas.isel(time=0))  # dims: lat, lon
mask2d.name = "estados_mexico"

# 4) Pesos por latitud
weights = xr.DataArray(
    np.cos(np.deg2rad(ds["lat"].values)),
    coords={"lat": ds["lat"]},
    dims=["lat"]
).broadcast_like(tas)  # dims: time, lat, lon

# 5) Bucle sobre cada estado para calcular promedio diario
daily_list = []
for sid in state_ids:
    sel = (mask2d == sid)    # bool mask lat×lon
    num = (tas * weights).where(sel).sum(dim=("lat","lon"))
    den =      weights   .where(sel).sum(dim=("lat","lon"))
    daily_list.append((num / den).rename(f"st{sid}"))

# 6) Concatenar en DataArray (estado_index, time)
daily = xr.concat(daily_list, dim="estado_index")
daily = daily.assign_coords(estado_index=("estado_index", state_ids))

# 7) Agrupar por mes y seleccionar junio
monthly = daily.groupby("time.month").mean(dim="time")
temps_month = monthly.sel(**{"month": MONTH})

# 8) Convertir a DataFrame y unir con GeoDataFrame
ser = temps_month.to_series().rename("temp_avg")
df = ser.reset_index()
df["estado_index"] = df["estado_index"].astype(int)

estados["estado_index"] = estados.index
estados_plot = estados.merge(df, on="estado_index")

# 9) Graficar con colorbar manual
fig, ax = plt.subplots(figsize=(10, 8))

# Dibuja el mapa, guarda la colección para la leyenda
col = estados_plot.plot(
    column="temp_avg",
    ax=ax,
    legend=False,        # desactivamos la leyenda automática
    cmap="hot_r",
    edgecolor="black",
    linewidth=0.5
)

# Crear ScalarMappable para la barra de colores
vmin = estados_plot["temp_avg"].min()
vmax = estados_plot["temp_avg"].max()
sm = plt.cm.ScalarMappable(
    cmap="hot_r",
    norm=plt.Normalize(vmin=vmin, vmax=vmax)
)
sm._A = []  # para que Matplotlib no dé warning

# Añadir colorbar horizontal bajo el mapa
cbar = fig.colorbar(
    sm,
    ax=ax,
    orientation="vertical",
    fraction=0.03,   # grosor de la barra
    pad=0.04         # espacio entre mapa y barra
)
cbar.set_label("Temperatura (°C)", fontsize=12)
cbar.ax.tick_params(rotation=45, labelsize=10)

# 10) Título en español gracias al locale
time_str = pd.to_datetime(f"2025-{MONTH:02d}-01").strftime("%B %Y")
ax.set_title(f"Temperatura máxima promedio en {time_str} por estado", fontsize=14)

ax.axis("off")
plt.tight_layout()
plt.show()
