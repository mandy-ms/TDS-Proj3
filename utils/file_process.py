import os
import zipfile
import tempfile
import shutil
import uuid
from pathlib import Path
from contextlib import contextmanager
from PIL import Image  # Still needed for image validation if desired

# Single directory for all temporary files
TMP_DIR = Path("/tmp") if os.environ.get("VERCEL") else Path("tmp_uploads")

@contextmanager
def managed_file_upload(file_path):
    """
    Context manager that processes an uploaded file and cleans up after it's used.
    
    Args:
        file_path: Path to the uploaded file
        
    Yields:
        tuple: (directory path, list of filenames)
    """
    file_path = Path(file_path)
    output_dir = None
    filenames = []
    
    try:
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
            dest_path = output_dir / f"{session_id}_{file_path.name}"
            shutil.copy2(file_path, dest_path)
            filenames = [dest_path.name]
        
        # Yield the results for the caller to use
        yield str(output_dir), filenames
        
    finally:
        # Clean up after use
        if output_dir and output_dir.exists() and output_dir != TMP_DIR:
            shutil.rmtree(output_dir)
        elif output_dir == TMP_DIR:
            # Just remove the specific files we created, not the whole directory
            for filename in filenames:
                file_to_remove = output_dir / filename
                if file_to_remove.exists():
                    os.remove(file_to_remove)

def process_uploaded_file(file_path):
    """
    Legacy function for backward compatibility.
    Processes uploaded files and returns paths without automatic cleanup.
    
    Args:
        file_path: Path to the uploaded file
        
    Returns:
        tuple: (directory path, list of filenames)
    """
    file_path = Path(file_path)
    
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
        # Handle single file - put everything in the same directory
        dest_path = TMP_DIR / f"{session_id}_{file_path.name}"
        shutil.copy2(file_path, dest_path)
        
        return str(TMP_DIR), [dest_path.name]