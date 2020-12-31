import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

exec(open("hats_pytorch/_version.py").read())

setuptools.setup(
    name="hats_pytorch",
    version=__version__,
    author="Marco Cannici",
    author_email="marco.cannici@polimi.it",
    description="A pytorch implementation of HATS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marcocannici/hats_pytorch",
    packages=["hats_pytorch"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 License",
    ],
    python_requires='>=3.6',
)
