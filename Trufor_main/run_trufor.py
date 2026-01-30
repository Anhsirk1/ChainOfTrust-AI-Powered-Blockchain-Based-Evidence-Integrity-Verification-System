import subprocess
import os

ROOT = os.path.abspath(os.path.dirname(__file__))
TEST_DOCKER = os.path.join(ROOT, "test_docker")

# USE PROJECT VENV PYTHON
TRUFOR_PYTHON = os.path.abspath(
    os.path.join(ROOT, "..", "venv", "Scripts", "python.exe")
)


def run_trufor(image_path, output_dir):

    if not os.path.exists(TRUFOR_PYTHON):
        raise FileNotFoundError("VENV python not found")

    if not os.path.exists(TEST_DOCKER):
        raise FileNotFoundError("test_docker folder missing")

    os.makedirs(output_dir, exist_ok=True)

    subprocess.run([
        TRUFOR_PYTHON,
        os.path.join(TEST_DOCKER, "src", "trufor_test.py"),
        "-gpu", "-1",
        "-in", os.path.abspath(image_path),
        "-out", os.path.abspath(output_dir)
    ], cwd=TEST_DOCKER, check=True)