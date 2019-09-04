from setuptools import setup, find_packages

# You can have one or more plugins.  Just list them all here.
# For each one, add a setup function in plugins/__init__.py
#
entry_points = """
[ginga.rv.plugins]
hscplanner=plugins:setup_hscplanner
"""

setup(
    name = 'naojutils_ginga_plugins',
    version = "0.2",
    description = "Specialty plugins for the Ginga reference viewer",
    author = "OCS Group, Software Division, Subaru Telescope, NAOJ",
    license = "BSD",
    url = "http://naojsoft.github.com/naojutils",
    install_requires = ["ginga>=2.7.2"],
    packages = find_packages(),
    include_package_data = True,
    package_data = {},
    entry_points = entry_points,
)
