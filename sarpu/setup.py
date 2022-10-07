import setuptools


setuptools.setup(
    name="sarpu",
    version="0.0.1",
    author="Jessa Bekker, Bennet Bernstein, Shweta Chopra",
    author_email="jessa.bekker@gmail.com",
    description="Adapted Package for learning from Positive and Unlabeled data under the SAR assumption",
    long_description_content_type="text/markdown",
    url="https://github.com/mlresearch/sarpu",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'sarpu = sarpu.__main__:main'
        ]
    },
    python_requires=">=3.7",
    install_requires=[
        "bitarray",
        "ipykernel",
        "matplotlib",
        "nbconvert",
        "numpy",
        "pandas",
        "requests",
        "scikit-learn",
        "scipy",
        "seaborn",
        "dill",
        "cvxopt",
        "tice @ git+https://github.com/bluelabsio/BL-SAR-PU.git@installation-edits#subdirectory=lib/tice",
        "km @ git+https://github.com/bluelabsio/BL-SAR-PU.git@installation-edits#subdirectory=lib/km"
    ],
)
