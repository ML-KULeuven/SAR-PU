import setuptools


setuptools.setup(
    name="sarpu",
    version="0.0.1",
    author="Jessa Bekker",
    author_email="jessa.bekker@gmail.com",
    description="Package for learning from Positive and Unlabeled data under the SAR assumption",
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
)
