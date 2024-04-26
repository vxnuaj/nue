# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys


sys.path.insert(0, os.path.abspath('..'))


project = 'nue'
copyright = '2024, vxnuaj'
author = 'vxnuaj'
release = '0.0.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.todo', 'sphinx.ext.githubpages']

templates_path = ['_templates']
exclude_patterns = ['Thumbs.db', '.DS_Store']
html_static_path = ['_static']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = 'alabaster'
html_theme_options = {
    'show_powered_by': False,
    'show_relbar': False,
    'show_searchbar': False,
    'logo': 'logo.png',
    'github_user': 'your-github-username',
    'github_repo': 'your-github-repo-name',
}