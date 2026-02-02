from setuptools import setup, find_packages
import platform  
import os
  
# Read in requirements.txt
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]

is_win = (platform.system() == 'Windows')
if is_win:
    pd_files = ['*.pyd', '*.dll', '*.pyi']
else :
    pd_files = ['*.so', '*.pyi']


setup(  
    name = "pyqpanda_alg",  
    version = "2.0.0",  
    license = "Apache Licence",  
    author = "OriginQ",
    install_requires=requirements,
    description= "A Quantum Algorithm Development and Runtime Environment Kit, based on pyqpanda3.",    
    packages = find_packages(),  
    
    package_data={
        '':pd_files
    },
    include_package_data = True,  
    classifiers=[
	"Development Status :: 4 - Beta",
	"Operating System :: MacOS :: MacOS X",
	"Operating System :: Microsoft :: Windows :: Windows 10",
	"Operating System :: POSIX :: Linux",
	"Topic :: Software Development :: Libraries :: Python Modules",
	"Programming Language :: Python",
	"Programming Language :: Python :: 3.11",
	"Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13"
	],
)