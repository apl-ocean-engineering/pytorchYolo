from distutils.core import setup

setup(name='pytorchYolo',
      version='0.1',
      description='YOLO implementation for PYtorch geared towards AMP images',
      url='https://github.com/apl-ocean-engineering/pytorchYolo',
      author='Mitchell Scott',
      author_email='miscott@uw.edu',
      license='MIT',
      packages=['pytorchYolo'],
      long_description=open('README.md').read(),
)
