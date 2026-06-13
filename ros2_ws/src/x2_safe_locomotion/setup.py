from setuptools import find_packages, setup

package_name = "x2_safe_locomotion"

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
    description="Phase 2 safe SDK locomotion (velocity-level only).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "input_source_registrar = x2_safe_locomotion.input_source_registrar:main",
            "velocity_adapter = x2_safe_locomotion.velocity_adapter:main",
            "command_smoother = x2_safe_locomotion.command_smoother:main",
            "motion_state_monitor = x2_safe_locomotion.motion_state_monitor:main",
            "safety_supervisor = x2_safe_locomotion.safety_supervisor:main",
            "emergency_stop_node = x2_safe_locomotion.emergency_stop_node:main",
        ],
    },
)
