import setuptools

setuptools.setup(
    name="km",
    description="Kernel Mixture Proportion Estimation",
    version="0.0.1",
    author="Ramaswamy",
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            'km = km.__main__:main'
        ]
    },
)
