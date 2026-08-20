# Data access

## What is released

- `data/points/point_registry.csv` and `.parquet`: exact 9,596 fixed sites used in the study, including projected and geographic coordinates.
- `data/temporal_split_v1.csv`: released target-month split labels and target-valid sample counts.
- `scripts/data/`: acquisition, export merging, and split-generation code.

The complete derived point-by-month reflectance panel, example reflectance
values, model weights, and paper-result outputs are not distributed. The
release instead provides the information needed to reconstruct the panel from
public Sentinel-2 collections.

The files in `examples/` are fully synthetic. They contain no Sentinel-2
reflectance, target observations, administrative boundary, or geographic
coordinate and are included only to exercise the public software contract.

## Exact-site reconstruction

Upload `data/points/point_registry.csv` to Google Earth Engine as a table and select `longitude` and `latitude` as the geometry columns. Use the resulting asset with `--point-asset`. This route preserves the exact sampling registry and is preferred for replication.

Earth Engine table upload documentation: <https://developers.google.com/earth-engine/guides/table_upload>

## Boundary-based reconstruction

A boundary is required only when generating a new 500 m sampling frame. It is not included because administrative-boundary redistribution rights and versions vary by provider. Obtain an authorized boundary from an official source, for example:

- Tianditu Data Source Center: <https://cloudcenter.tianditu.gov.cn/dataSource>
- Tianditu administrative-region services: <https://lbs.tianditu.gov.cn/server/search2.html>
- the competent municipal or provincial natural-resources authority.

Pass an uploaded Earth Engine boundary asset with `--region-asset`, or a local authorized GeoJSON with `--boundary-geojson`. A boundary-derived frame may differ at edge locations from the released exact sites.

## Source-data terms

Sentinel-2 images remain governed by their original Copernicus terms and Google
Earth Engine access conditions. Copernicus permits reuse and redistribution of
Sentinel data subject to its attribution requirements; an Earth Engine export
does not replace those source-provider terms. This repository does not
redistribute source imagery or the derived monthly reflectance panel.

Uploading a boundary to a user-owned Earth Engine asset does not by itself
establish redistribution rights for the underlying boundary file. The study
boundary is therefore not released because the local copy does not include
verifiable provider and license metadata. The exact fixed-site registry needed
for replication is released instead.

- Copernicus Sentinel data legal notice: <https://cds.climate.copernicus.eu/licences/ec-sentinel>
- Google Earth Engine terms: <https://explorer.earthengine.google.com/terms>
- Earth Engine asset management: <https://developers.google.com/earth-engine/guides/manage_assets>
