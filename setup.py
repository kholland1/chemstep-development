import os.path
import setuptools

HERE = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(HERE, "README.md")) as f:
    README = f.read()

setuptools.setup(
    name="chemstep",
    version="0.1.0",
    author="Olivier Mailhot",
    description="chemstep package",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    package_data={},
    test_suite='tests',
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
