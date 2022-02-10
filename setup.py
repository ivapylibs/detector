from setuptools import setup
setup(name='detector',
      version='1.0',
      description='Classes implementing detection based processing pipelines.', 
      author='IVALab',
      packages=['detector'],
      install_requires=['numpy', 'matplotlib', "roipoly"]
)
