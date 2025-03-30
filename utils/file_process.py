import os
import zipfile
import tempfile
import shutil
import uuid
import requests
from pathlib import Path
from urllib.parse import urlparse
from contextlib import contextmanager
from PIL import Image  # Still needed for image validation if desired

# Single directory for all temporary files
TMP_DIR = Path("/tmp") if os.environ.get("VERCEL") else Path("tmp_uploads")

def is_url(path_or_url):
    """Check if the provided string is a URL."""
    if not isinstance(path_or_url, str):
        return False
    try:
        result = urlparse(path_or_url)
        return all([result.scheme, result.netloc])
    except:
        return False

def download_file(url):
    """Download a file from a URL to a temporary location and return its path."""
    # Create a temporary file with the correct extension
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path) or f"downloaded_file_{str(uuid.uuid4())[:8]}"
    
    # Ensure tmp directory exists
    os.makedirs(TMP_DIR, exist_ok=True)
    
    download_path = TMP_DIR / f"download_{str(uuid.uuid4())[:8]}_{filename}"
    
    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(download_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return download_path

@contextmanager
def managed_file_upload(file_path_or_url):
    """
    Context manager that processes an uploaded file or URL and cleans up after it's used.
    
    Args:
        file_path_or_url: Path to the uploaded file or URL to download
        
    Yields:
        tuple: (directory path, list of filenames)
    """
    temp_download = None
    output_dir = None
    filenames = []
    
    try:
        # Handle URL if provided
        if is_url(file_path_or_url):
            try:
                temp_download = download_file(file_path_or_url)
                file_path = temp_download
            except Exception as e:
                yield str(e), []
                return
        else:
            file_path = Path(file_path_or_url)
        
        # Create unique session ID to avoid filename conflicts
        session_id = str(uuid.uuid4())[:8]
        
        # Ensure tmp directory exists
        os.makedirs(TMP_DIR, exist_ok=True)
        
        if zipfile.is_zipfile(file_path):
            # Handle ZIP file
            output_dir = TMP_DIR / f"zip_{session_id}"
            os.makedirs(output_dir, exist_ok=True)
            
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)
                filenames = zip_ref.namelist()
        else:
            # Handle single file
            output_dir = TMP_DIR
            dest_path = output_dir / f"{session_id}_{os.path.basename(str(file_path))}"
            shutil.copy2(file_path, dest_path)
            filenames = [os.path.basename(str(dest_path))]
        
        # Yield the results for the caller to use
        yield str(output_dir), filenames
        
    finally:
        # Clean up downloaded file if it exists
        if temp_download and os.path.exists(temp_download):
            os.remove(temp_download)
            
        # Clean up extracted/copied files
        if output_dir and output_dir.exists() and output_dir != TMP_DIR:
            shutil.rmtree(output_dir)
        elif output_dir == TMP_DIR:
            # Just remove the specific files we created
            for filename in filenames:
                file_to_remove = output_dir / filename
                if file_to_remove.exists():
                    os.remove(file_to_remove)

def process_uploaded_file(file_path_or_url):
    """
    Processes uploaded files or URLs and returns paths without automatic cleanup.
    
    Args:
        file_path_or_url: Path to the file or URL to download
        
    Returns:
        tuple: (directory path, list of filenames)
    """
    temp_download = None
    
    try:
        # Handle URL if provided
        if is_url(file_path_or_url):
            try:
                temp_download = download_file(file_path_or_url)
                file_path = temp_download
            except Exception as e:
                return str(e), []
        else:
            file_path = Path(file_path_or_url)
        
        # Ensure tmp directory exists
        os.makedirs(TMP_DIR, exist_ok=True)
        
        # Generate a unique session ID
        session_id = str(uuid.uuid4())[:8]
        
        if zipfile.is_zipfile(file_path):
            # Handle ZIP file
            output_dir = TMP_DIR / f"zip_{session_id}"
            os.makedirs(output_dir, exist_ok=True)
            
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)
                filenames = zip_ref.namelist()
            
            return str(output_dir), filenames
        else:
            # Handle single file
            dest_path = TMP_DIR / f"{session_id}_{os.path.basename(str(file_path))}"
            shutil.copy2(file_path, dest_path)
            
            return str(TMP_DIR), [os.path.basename(str(dest_path))]
    finally:
        # Clean up downloaded file if needed but not the processed result
        if temp_download and os.path.exists(temp_download):
            os.remove(temp_download)