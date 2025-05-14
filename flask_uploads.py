import os
from flask import current_app
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

# Supported image extensions
IMAGES = ('jpg', 'jpeg', 'png')

class UploadSet:
    """A class to manage file uploads for a specific set of files."""
    def __init__(self, name, extensions):
        self.name = name
        self.extensions = extensions
        self.dest = None

    def config_destination(self, destination):
        """Set the destination folder for uploads."""
        self.dest = destination

    def save(self, storage, folder=None, name=None):
        """Save an uploaded file to the configured destination."""
        if not isinstance(storage, FileStorage):
            raise TypeError("Storage must be a werkzeug.datastructures.FileStorage object")

        # Validate file extension
        filename = secure_filename(storage.filename)
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        if ext not in self.extensions:
            raise ValueError(f"File extension '{ext}' not allowed. Allowed: {', '.join(self.extensions)}")

        # Determine destination path
        dest_folder = self.dest
        if folder:
            dest_folder = os.path.join(dest_folder, folder)
        os.makedirs(dest_folder, exist_ok=True)

        # Generate filename if not provided
        if name:
            filename = name
            if '.' not in filename:
                filename = f"{filename}.{ext}"
        else:
            # Use original filename or generate a unique one
            base, _ = os.path.splitext(filename)
            counter = 1
            new_filename = filename
            while os.path.exists(os.path.join(dest_folder, new_filename)):
                new_filename = f"{base}_{counter}.{ext}"
                counter += 1
            filename = new_filename

        # Save the file
        dest_path = os.path.join(dest_folder, filename)
        storage.save(dest_path)
        return filename

def configure_uploads(app, upload_sets):
    """Configure upload sets with the Flask app."""
    for upload_set in upload_sets if isinstance(upload_sets, (list, tuple)) else [upload_sets]:
        if not hasattr(app, 'upload_set_configs'):
            app.upload_set_configs = {}
        destination = app.config.get(f'UPLOADED_{upload_set.name.upper()}_DEST')
        if not destination:
            raise ValueError(f"No destination configured for upload set '{upload_set.name}'. "
                             f"Set 'UPLOADED_{upload_set.name.upper()}_DEST' in app config.")
        upload_set.config_destination(destination)

# Example usage:
# photos = UploadSet('photos', IMAGES)
# configure_uploads(app, photos)