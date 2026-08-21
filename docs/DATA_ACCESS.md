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

The original 500 m sampling frame used the Huanghua City administrative
boundary (`adcode` 130983) obtained from the official Alibaba Cloud
DataV.GeoAtlas area selector:

- area selector: <https://datav.aliyun.com/portal/school/atlas/area_selector>
- provider-hosted GeoJSON: <https://geo.datav.aliyun.com/areas_v3/bound/130983.json>
- map version: `areas_v3`, updated May 2021
- underlying map-data provider: AMap Open Platform
- accessed: 20 August 2026
- reference SHA-256: `F7EFF392ADE7A2B2048C553B3C3ADC6268FFF03C6372EE981C55CE67E6D4D1DD`

The provider-hosted GeoJSON was compared with the boundary used for the
frozen Earth Engine asset. Their bounds are identical
(`[117.085117, 38.154614, 117.958012, 38.643864]`) and the symmetric-difference
area is zero. The repository does not redistribute this boundary because the
DataV selector states that the map data are for learning and exchange and does
not provide a general redistribution licence.

### Coordinate-system caveat

DataV states that `areas_v3` originates from AMap Open Platform. AMap's
official documentation identifies its platform coordinates as GCJ-02, while
the DataV selector documentation does not explicitly state whether its
open-format export is transformed to another datum. The frozen study asset
used the coordinate sequence supplied by the DataV GeoJSON. Therefore, exact
frame reconstruction requires preserving those coordinate values as supplied;
do not apply an additional datum conversion, simplification, buffer, or
replacement boundary. This statement documents the frozen run and does not
assert a CRS transformation that DataV has not documented.

Upload the downloaded GeoJSON to a user-owned Earth Engine asset and pass its
identifier with `--region-asset`, or use it locally with
`--boundary-geojson`. A regenerated boundary-based frame can still differ at
edge locations if import or geometry-processing settings change. For exact
replication, upload `data/points/point_registry.csv` and use `--point-asset`.

## Source-data terms

Sentinel-2 images remain governed by their original Copernicus terms and Google
Earth Engine access conditions. Copernicus permits reuse and redistribution of
Sentinel data subject to its attribution requirements; an Earth Engine export
does not replace those source-provider terms. This repository does not
redistribute source imagery or the derived monthly reflectance panel.

Uploading a boundary to a user-owned Earth Engine asset does not by itself
establish redistribution rights for the underlying boundary file. The study
boundary is therefore linked to its verified provider source rather than
redistributed. The exact fixed-site registry needed for replication is released
directly.

- Copernicus Sentinel data legal notice: <https://cds.climate.copernicus.eu/licences/ec-sentinel>
- Google Earth Engine terms: <https://explorer.earthengine.google.com/terms>
- Earth Engine asset management: <https://developers.google.com/earth-engine/guides/manage_assets>
- DataV.GeoAtlas selector documentation: <https://help.aliyun.com/zh/datav/datav-7-0/user-guide/introduction-to-features-of-a-range-selector>
- AMap coordinate-system statement: <https://lbs.amap.com/faq/advisory/others/39838>
