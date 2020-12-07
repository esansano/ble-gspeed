# ble-gspeed dataset
Gait speed estimation using BLE devices

## Dataset

- `mac`: The MAC address of the detected beacon.
- `rssi`: The RSSI value obtained for the beacon. 
- `device`: A four-character descriptor for the smartwatch that performed the scan.
- `timestamp`: The time stamp at which the scan was received.
- `user`: The id of the user that was performing the experiment.
- `direction`: A number (0 or 1) indicating the direction of the walk. 
- `walk_id`: A number that identifies each walk. 
- `speed`: The actual speed of the user, in $m/s$.
	

`doi`: 10.5281/zenodo.4261381

`link`: https://zenodo.org/record/4261381#.X8aMjllKj0o

`description`: GSPEED - BLE-based gait speed dataset

## Related paper

`doi`: 10.3390/data5040115

`link`: https://www.mdpi.com/2306-5729/5/4/115/pdf

If you use this dataset, please cite this paper:

```
    Sansano-Sansano, E.; Aranda, F.J.; Montoliu, R.; Álvarez, F.J. 
    BLE-GSpeed: A New BLE-Based Dataset to Estimate User Gait Speed. 
    Data 2020, 5, 115.
```
