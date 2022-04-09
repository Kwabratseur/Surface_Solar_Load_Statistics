from distutils.core import setup

setup(
    name='surface_solar_load_statistics',
    version='0.3.0',
    description='Surface Solar Load Statistics - Calculate solar load on any inclined surface, use statistical models bound to location to split up radiation components.',
    author="Jeroen van 't Ende",
    author_email='jeroen.vantende@outlook.com',
    url='https://github.com/Kwabratseur/STOC',
    packages=['surface_solar_load_statistics'],
    #scripts=['bin/'],
    install_requires=["pysolar",
                      "matplotlib",
                      "scipy",
                      "numpy"],
)
