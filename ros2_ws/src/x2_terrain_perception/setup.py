from setuptools import find_packages, setup

package_name = "x2_terrain_perception"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="X2 Terrain Locomotion",
    maintainer_email="liangkaifeng1987@gmail.com",
    description="Phase 1 terrain perception pipeline (perception only).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pointcloud_projector = x2_terrain_perception.pointcloud_projector:main",
            "ground_plane_estimator = x2_terrain_perception.ground_plane_estimator:main",
            "heightmap_node = x2_terrain_perception.heightmap_node:main",
            "slope_detector = x2_terrain_perception.slope_detector:main",
            "stair_detector = x2_terrain_perception.stair_detector:main",
            "gap_detector = x2_terrain_perception.gap_detector:main",
            "traversability_estimator = x2_terrain_perception.traversability_estimator:main",
            "terrain_classifier = x2_terrain_perception.terrain_classifier:main",
            "visualization_node = x2_terrain_perception.visualization_node:main",
        ],
    },
)
