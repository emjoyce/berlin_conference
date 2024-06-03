import setuptools

setuptools.setup(
    name="berlin_code",
    version="0.0.1",
    author="Emily Joyce",
    author_email="emily.m.joyce1@gmail.com",
    description="functions used in generating berlin connectomics conference figures and analysis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"berlin_code": "cluster_analysis", "berlin_code":"plot"},  # noqa: F601
    packages=setuptools.find_packages(where = 'src'),
    python_requires=">=3.6",
)