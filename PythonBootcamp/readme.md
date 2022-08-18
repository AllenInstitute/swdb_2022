## For those who are comfortable with Conda environments (or want to become so):
Here are instructions for creating a Python 3 Conda environment that will ensure that all of the Python Bootcamp notebooks can be run:

First, create a new Python 3 environment as defined in the `requirements.yml` file.

    conda env create -f requirements.yml

The above line should have created a new environment called `swdb_2019`. Now activate the environment:

    conda activate swdb_2019

Register the environment with Jupyter (so you can select it from a dropdown in the browser):

    python -m ipykernel install --user --name swdb_2019