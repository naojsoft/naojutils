from setuptools import setup, find_packages

# You can have one or more plugins.  Just list them all here.
# For each one, add a setup function in plugins/__init__.py
#
entry_points = """
[ginga.rv.plugins]
HSCPlanner=plugins:setup_HSCPlanner
"""

setup(
    name = 'NAOJGingaPlugins',
    version = "0.2",
    description = "Plugins for the Ginga reference viewer",
    author = "OCS Group, Software Division, Subaru Telescope, NAOJ",
    license = "BSD",
    # change this to your URL
    url = "http://naojsoft.github.com/naojutils",
    install_requires = ["ginga>=2.6.2"],
    packages = find_packages(),
    include_package_data = True,
    package_data = {},
    entry_points = entry_points,
)
