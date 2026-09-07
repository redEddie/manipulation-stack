import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="manipulation-stack",
    version="0.1.0",
    author="Chanwook Jeon",
    author_email="eddie3618@knu.ac.kr",
    description="Teleoperation, scene-based data collection, and VLA policy deployment for a manipulation station",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/redEddie/manipulation-stack",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
    license="MIT",
    # The real dependency list is requirements.txt (GUI/library) and
    # requirements_node.txt (robot node). Only what importing `mstack` itself
    # needs is declared here; the hardware stack is installed per interpreter.
    install_requires=[
        "numpy",
        "h5py",
        "PyYAML",
    ],
)
