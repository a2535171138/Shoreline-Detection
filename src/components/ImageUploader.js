// ImageUploader.js
import React, { useRef, useEffect } from 'react';

function ImageUploader({ onFileUpload }) {
    const fileInputRef = useRef(null);
    const dropzoneRef = useRef(null);

    const handleUploadClick = () => {
        fileInputRef.current.click();
    };

    const handleFileChange = (event) => {
        const files = Array.from(event.target.files);
        onFileUpload(files);
        event.target.value = null;
    };

    const handleDragOver = (event) => {
        event.preventDefault();
        event.stopPropagation();
        event.dataTransfer.dropEffect = 'copy';
    };

    const handleDrop = (event) => {
        event.preventDefault();
        event.stopPropagation();
        const files = Array.from(event.dataTransfer.files);
        onFileUpload(files);
    };

    useEffect(() => {
        const dropzone = dropzoneRef.current;
        dropzone.addEventListener('dragover', handleDragOver);
        dropzone.addEventListener('drop', handleDrop);

        return () => {
            dropzone.removeEventListener('dragover', handleDragOver);
            dropzone.removeEventListener('drop', handleDrop);
        };
    }, [onFileUpload]);

    return (
        <div ref={dropzoneRef} className="dropzone">
            <button className="btn btn-primary mb-2" onClick={handleUploadClick}>Upload Image</button>
            <input
                type="file"
                multiple
                className="form-control mb-3 d-none"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/*"
            />
        </div>
    );
}

export default ImageUploader;
