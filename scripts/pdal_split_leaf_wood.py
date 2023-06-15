#!/usr/bin/env python
# -- coding: utf-8 --

import pdal
import json
from pathlib import Path

infiles = Path("H:/helios/output/").glob("tls_tree*/*/tls_tree*.laz")

for f in infiles:
    print(f)
    if "mls" in f.stem:
        print(f"Skipping {f}")
        continue
    f_leaf = f.as_posix().replace(".laz", "_leaves_nosor.las")
    f_wood = f.as_posix().replace(".laz", "_wood_nosor.las")
    f_blossom = f.as_posix().replace(".laz", "_blossom_nosor.las")

    print(f"Writing:\n\t{f_leaf}\n\t{f_wood}\n\t{f_blossom}")
    json_config = [
                {
                    "type": "readers.las",
                    "filename": f.as_posix(),
                    "nosrs": True
                },
                # {
                #     "type": "filters.outlier",
                #     "method": "statistical",
                #     "mean_k": 6,
                #     "multiplier": 1.0
                # },
                # {
                #    "type": "filters.sample",
                #    "radius": 0.0025
                # },
                {
                    "type": "writers.las",
                    "where": "Classification == 1",
                    "forward": "all",
                    "filename": f_leaf,
                    "extra_dims": "all"
                },
                {
                    "type": "writers.las",
                    "where": "Classification == 0",
                    "forward": "all",
                    "filename": f_wood,
                    "extra_dims": "all"
                },
                {
                    "type": "writers.las",
                    "where": "Classification == 2",
                    "forward": "all",
                    "filename": f_blossom,
                    "extra_dims": "all"
                }
            ]
    pipeline = pdal.Pipeline(json.dumps(json_config))
    pipeline.execute()
