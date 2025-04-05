import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "Text-Summarizer"
AUTHOR_USER_NAME = "jairathnishant"
SRC_REPO = "textSummarizer"
AUTHOR_EMAIL = "jairath.nishant@gmail.com"

setuptools.setup(
    name= SRC_REPO,
    version=__version__,
    author = AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="NLP app",
    long_description=long_description,
    long_description_content = "text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls = {
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where='src')
)


# from setuptools import setup, find_packages

# setup(
#     name="textSummarizer",
#     version="0.0.1",
#     author="jairathnishant",
#     author_email="jairath.nishant@gmail.com",
#     description="NLP app",
#     packages=find_packages(where="src"),
#     package_dir={"": "src"},
# )
