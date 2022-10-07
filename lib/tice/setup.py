import setuptools

setuptools.setup(
    name="tice",
    version="0.0.1",
    author="Jessa Bekker",
    author_email="jessa.bekker@gmail.com",
    description="Package for estimating the class prior from Positive and Unlabeled data under the SAR assumption",
    long_description_content_type="text/markdown",
    url="https://github.com/mlresearch/tice",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'tice = tice.__main__:main'
        ]
    },
)
