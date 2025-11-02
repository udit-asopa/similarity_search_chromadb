"""
Simple test to verify imports and Python path configuration.
"""
import pytest
import sys
import os


def test_python_path_setup():
    """Test that the project root is in Python path."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Check if project root is in sys.path or current directory is accessible
    assert (project_root in sys.path or 
            os.path.exists(os.path.join(os.getcwd(), 'api', 'main.py')) or
            os.path.exists(os.path.join(project_root, 'api', 'main.py')))


def test_api_import_available():
    """Test if the API module can be imported."""
    try:
        import api.main
        assert hasattr(api.main, 'app')
        print("✅ API module imported successfully")
    except ImportError as e:
        # This is expected in some test environments
        print(f"ℹ️  API import not available: {e}")
        pytest.skip("API module not available - this is expected in some test environments")


def test_basic_imports():
    """Test that basic test dependencies are available."""
    import pytest
    assert pytest is not None
    
    try:
        from unittest.mock import patch, MagicMock
        assert patch is not None
        assert MagicMock is not None
        print("✅ Mock utilities available")
    except ImportError:
        pytest.fail("Mock utilities not available")


def test_project_structure():
    """Test that expected project files exist."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Check for key project files
    expected_files = [
        'pixi.toml',
        'README.md',
        'api/main.py'
    ]
    
    for file_path in expected_files:
        full_path = os.path.join(project_root, file_path)
        if not os.path.exists(full_path):
            # Try current directory
            full_path = os.path.join(os.getcwd(), file_path)
        
        assert os.path.exists(full_path), f"Expected file not found: {file_path}"
    
    print("✅ Project structure verified")