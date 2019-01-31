import setuptools
import subprocess
import os
from setuptools.command.install import install

class PySkeltonInstall(install):

    def run(self):
        super().run()
        self.compile_lib()

    def compile_lib(self):
        wd = self.install_lib + "PySkelton"
        # print(f"----- The building is here: {wd} -----")
        subprocess.run(["make", "field_eval"],cwd=wd,shell=True)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PySkelton",
    version="0.0.1",
    author="Alvaro Fuentes",
    author_email="aj.fuentes.suarez@yandex.com",
    description="Skeleton-based modeling library with scaffolding and convolution surfaces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.inria.fr/afuentes/pyskelton",
    packages=setuptools.find_packages(),
    package_data={'': ['*.c','makefile','field_eval_static.so'],},
    keywords="scaffold skeleton modeling convolution surfaces",
    install_requires=["pyhull","numpy","pyroots"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: POSIX :: Linux",
    ],
    cmdclass={'install': PySkeltonInstall},
)
