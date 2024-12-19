from setuptools import find_packages, setup
from typing import List

# Define the editable mode constant
HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    Read the requirements.txt file and return a list of requirements,
    excluding the '-e .' entry.
    """
    requirements = []
    with open(file_path, "r") as file_obj:
        # Read all lines and strip newline characters
        requirements = [req.strip() for req in file_obj.readlines()]

        # Remove the '-e .' entry if it exists
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

# Setup function
setup(
    name="mlprojects",
    version="0.0.1",
    author="Bhagyashree Wagh",
    author_email="bwagh@uw.edu",
    packages=find_packages(),  # Automatically find packages
    install_requires=get_requirements("requirements.txt"),  # Load dependencies
)

