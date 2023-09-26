from setuptools import find_packages, setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")




with open('requirements.txt') as inn:
    requirements = inn.read().splitlines()

setup(
    name = 'LPDT',

    version = '0.0.1',

    description = 'Locally differentially Private Decision Tree',

    long_description = long_description,

    long_description_content_type = "text/markdown",

    url = 'https://github.com/Karlmyh/LPDT',

    author = "Yuheng Ma, Han Zhang",

    author_email = "yma@ruc.edu.cn",
    
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    
    
    packages = find_packages(),

    python_requires = '>=3',
    
    install_requires = requirements,
)
