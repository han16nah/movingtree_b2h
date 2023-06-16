# ---------------------------------------------------------------------------#
# AUTHOR: Hannah Weiser                                                      #
#                                                                            #
# Calculation of geometric features                                          #
# ---------------------------------------------------------------------------#


# ----------------- #
# ---  IMPORTS  --- #
# ----------------- #
import pdal
import warnings
import json


def pdal_covariance_features(input_path: str,
                             output_path: str,
                             feature_set: str,
                             knn: int = 10,
                             radius: float = None,
                             min_k: int = 3,
                             threads: int = 1,
                             mode: str = "raw"
                             ) -> None:
    """
    Compute covariance features using PDAL
    https://pdal.io/en/stable/stages/filters.covariancefeatures.html
    :param input_path: path of point cloud to compute features from (LAS/LAZ)
    :param output_path: path of point cloud to create (LAS/LAZ)
    :param feature_set: comma separated string with the features to compute
    :param knn: number of nearest neighbours defining the point neighbourhood
    :param radius: radius defining the point neighbourhood (if specified, will use radius instead of knn)
    :param min_k: minimum number of neighbours in radius (radius search only)
    :param threads: number of threads usd for computing the feature descriptor
    :param mode: compute features a) using the square root of the computed eigenvalues ("SQRT"),
                 b) normalizing the eigenvalues such that they sum to one ("Normalized"), or
                 c) using the eigenvalues directly ("Raw")
    :return: None
    """
    supported_features = [
        "Anisotropy",
        "DemantkeVerticality",
        "Density"
        "Eigenentropy",
        "Linearity",
        "Omnivariance",
        "Planarity",
        "Scattering",
        "EigenValueSum",
        "SurfaceVariation",
        "Verticality",
        "Dimensionality",  # Linearity, Planarity, Scattering, Verticality
        "all"  # all supported features
    ]
    supported_modes = [
        "Raw",
        "Normalized",
        "SQRT"
    ]
    provided_features = feature_set.split(",")
    assert set(provided_features).issubset(supported_features)
    assert mode in supported_modes
    if "all" in provided_features and len(provided_features) > 1:
        warnings.warn("'all' means all features are computed. No need to specify further features.")
        feature_set = "all"

    json_config = [
        {
            "type": "readers.las",
            "filename": f"{input_path}",
            "nosrs": "true"
        },
        {
            "type": "filters.covariancefeatures",
            "knn": knn,
            "threads": threads,
            "feature_set": feature_set,
            "mode": mode
        },
        {
            "type": "writers.las",
            "filename": output_path,
            "compression": "true",
            "extra_dims": "all",
            "forward": "all"
        }
    ]
    if radius:
        json_config[1]["radius"] = radius
        json_config[1]["min_k"] = min_k

    pipeline = pdal.Pipeline(json.dumps(json_config))
    pipeline.execute()


def pdal_normals(input_path: str,
                 output_path: str,
                 knn: int = 8,
                 always_up: bool = True,
                 refine: bool = False
                 ) -> None:
    """
    Compute Normals using PDAL (output file will contain NormalX, NormalY, NormalZ and Curvature)
    https://pdal.io/en/stable/stages/filters.normal.html
    :param input_path: path of point cloud to compute features from (LAS/LAZ)
    :param output_path: path of point cloud to create (LAS/LAZ)
    :param knn: the number of k nearest neighbours
    :param always_up: flag indicating whether or not normals should be inverted only when the Z component is negative
    :param refine: flag indicating whether or not to reorient normals using minimum spanning tree propagation
    :return: None
    """
    json_config = [
        {
            "type": "readers.las",
            "filename": f"{input_path}",
            "nosrs": "true"
        },
        {
            "type": "filters.normal",
            "knn": knn
        },
        {
            "type": "writers.las",
            "filename": output_path,
            "compression": "true",
            "extra_dims": "all",
            "forward": "all"
        }
    ]
    if not always_up:
        json_config[1]["always_up"] = "false"
    if refine:
        json_config[1]["refine"] = "true"

    pipeline = pdal.Pipeline(json.dumps(json_config))
    pipeline.execute()


def pdal_eigenvalues(input_path: str,
                     output_path: str,
                     knn: int = 8,
                     normalize: bool = False
                     ) -> None:
    """
    Compute Eigenvalues using PDAL (output file will contain NormalX, NormalY, NormalZ and Curvature)
    https://pdal.io/en/stable/stages/filters.eigenvalues.html
    :param input_path: path of point cloud to compute features from (LAS/LAZ)
    :param output_path: path of point cloud to create (LAS/LAZ)
    :param knn: the number of k nearest neighbours
    :param normalize: flag indicating whether or not to normalize eigenvalues such that the sum is 1
    :return: None
    """
    json_config = [
        {
            "type": "readers.las",
            "filename": f"{input_path}",
            "nosrs": "true"
        },
        {
            "type": "filters.eigenvalues",
            "knn": knn
        },
        {
            "type": "writers.las",
            "filename": output_path,
            "compression": "true",
            "extra_dims": "all",
            "forward": "all"
        }
    ]
    if normalize:
        json_config[1]["normalize"] = "true"

    pipeline = pdal.Pipeline(json.dumps(json_config))
    pipeline.execute()
