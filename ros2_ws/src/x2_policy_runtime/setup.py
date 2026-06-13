from setuptools import find_packages, setup

package_name = "x2_policy_runtime"

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
    description="Phase 5/6 ONNX policy runtime (gated by safety approval flag).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "joint_policy_node = x2_policy_runtime.joint_policy_node:main",
        ],
    },
)
