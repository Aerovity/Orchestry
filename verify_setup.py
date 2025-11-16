"""Setup script for Orchestry.

Helps verify installation and setup.
"""

import subprocess
import sys
from pathlib import Path


def check_python_version() -> bool:
    """Check Python version."""
    print("Checking Python version...")
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def install_dependencies() -> bool:
    """Install required packages."""
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False


def check_env_file() -> bool:
    """Check for .env file."""
    print("\nChecking environment setup...")
    env_path = Path(".env")

    if not env_path.exists():
        print("⚠ .env file not found")
        print("  Creating from .env.example...")

        example_path = Path(".env.example")
        if example_path.exists():
            with open(example_path) as f:
                content = f.read()
            with open(env_path, "w") as f:
                f.write(content)
            print("✓ .env file created")
            print("  → Please edit .env and add your ANTHROPIC_API_KEY")
            return False
        print("✗ .env.example not found")
        return False

    # Check if API key is set
    with open(env_path) as f:
        content = f.read()

    if "your-api-key-here" in content or not any(
        "ANTHROPIC_API_KEY=" in line and "=" in line and line.split("=")[1].strip()
        for line in content.split("\n")
    ):
        print("⚠ ANTHROPIC_API_KEY not set in .env")
        print("  → Please edit .env and add your API key")
        return False

    print("✓ .env file configured")
    return True


def check_config() -> bool:
    """Check config file."""
    print("\nChecking configuration...")
    config_path = Path("config.yaml")

    if not config_path.exists():
        print("✗ config.yaml not found")
        return False

    print("✓ config.yaml found")
    return True


def run_tests() -> bool:
    """Run basic tests."""
    print("\nRunning basic tests...")
    try:
        subprocess.check_call([sys.executable, "tests/test_basic.py"])
        return True
    except subprocess.CalledProcessError:
        print("⚠ Some tests failed (this is OK if you haven't set up API key yet)")
        return False


def create_directories() -> bool:
    """Create necessary directories."""
    print("\nCreating directories...")
    dirs = ["runs", "tests"]

    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)

    print("✓ Directories created")
    return True


def main() -> None:
    """Run setup."""
    print("=" * 60)
    print("Orchestry Setup")
    print("=" * 60 + "\n")

    steps = [
        ("Python Version", check_python_version),
        ("Dependencies", install_dependencies),
        ("Directories", create_directories),
        ("Configuration", check_config),
        ("Environment", check_env_file),
        ("Tests", run_tests),
    ]

    results = []

    for step_name, step_func in steps:
        try:
            success = step_func()
            results.append((step_name, success))
        except Exception as e:
            print(f"✗ {step_name} failed with error: {e}")
            results.append((step_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Setup Summary")
    print("=" * 60)

    all_passed = True
    for step_name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {step_name}")
        if not success:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ Setup complete! You can now run Orchestry:")
        print("\n  python main.py --test")
        print("\nFor more options, see README.md")
    else:
        print("\n⚠ Setup incomplete. Please address the issues above.")
        print("\nCommon issues:")
        print("  1. Set ANTHROPIC_API_KEY in .env file")
        print("  2. Install dependencies: pip install -r requirements.txt")
        print("\nSee README.md for detailed instructions.")

    print()


if __name__ == "__main__":
    main()
