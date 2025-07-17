for year in $(seq 2025 2025); do
  #url="https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmax.${year}.nc"
  url2="https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmin.${year}.nc"  
  #echo "Descargando tmax.${year}.nc..."
  echo "Descargando tmin.${year}.nc..."  
  #wget -c "$url"
  wget -c "$url2"
done
