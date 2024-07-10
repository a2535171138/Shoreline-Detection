import React, { useState } from 'react';
import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import MiniDrawer from './components/MiniDrawer';
import MainWorkArea from './components/MainWorkArea';
import axios from 'axios';

function App() {
    const [uploadedImageFiles, setUploadedImageFiles] = useState([]);
    const [predictionResults, setPredictionResults] = useState([]);
    const [showResults, setShowResults] = useState(false);
    const [processedImages, setProcessedImages] = useState(new Set());
    const [displayModes, setDisplayModes] = useState({});

    const handleFileUpload = (files) => {
        const imageFiles = Array.from(files).map(file => ({
            file,
            url: URL.createObjectURL(file),
            id: Date.now() + Math.random()
        }));
        setUploadedImageFiles(prevFiles => [...prevFiles, ...imageFiles]);
        setPredictionResults(prevResults => [...prevResults, ...new Array(files.length).fill(null)]);
    };

    const handleToggleAllDisplayModes = () => {
        setDisplayModes(prevModes => {
            const newModes = {};
            const currentMode = Object.values(prevModes)[0] || 'binary';
            const nextMode = currentMode === 'binary' ? 'color' : 'binary';
            uploadedImageFiles.forEach((_, index) => {
                newModes[index] = nextMode;
            });
            return newModes;
        });
    };

    const handleDeleteImage = (index) => {
        setUploadedImageFiles(prevFiles => {
            const file = prevFiles[index];
            URL.revokeObjectURL(file.url);
            const newFiles = prevFiles.filter((_, i) => i !== index);
            setPredictionResults(prevResults => prevResults.filter((_, i) => i !== index));
            setProcessedImages(prevProcessed => {
                const newProcessed = new Set(prevProcessed);
                newProcessed.delete(file.id);
                return newProcessed;
            });
            return newFiles;
        });
    };

    const handleClearImages = () => {
        setUploadedImageFiles(prevFiles => {
            prevFiles.forEach(file => URL.revokeObjectURL(file.url));
            return [];
        });
        setPredictionResults([]);
        setProcessedImages(new Set());
        setShowResults(false);
    };

    const handleGetResult = async () => {
        if (uploadedImageFiles.length > 0) {
            setShowResults(true);
            const newPredictionResults = [...predictionResults];

            for (let i = 0; i < uploadedImageFiles.length; i++) {
                const imageFile = uploadedImageFiles[i];
                if (!processedImages.has(imageFile.id)) {
                    const formData = new FormData();
                    formData.append('file', imageFile.file);

                    try {
                        const response = await axios.post('http://localhost:5000/predict', formData, {
                            headers: {
                                'Content-Type': 'multipart/form-data'
                            }
                        });

                        newPredictionResults[i] = {
                            binaryResult: `data:image/png;base64,${response.data.binary_result}`,
                            colorResult: `data:image/png;base64,${response.data.color_result}`,
                            pixelsResult: response.data.pixels_result,
                            processingTime: response.data.processingTime,
                        };

                        setProcessedImages(prev => new Set(prev).add(imageFile.id));
                    } catch (error) {
                        console.error('Error fetching prediction result:', error);
                        newPredictionResults[i] = 'error';
                    }
                    setPredictionResults([...newPredictionResults]);
                }
            }
        }
    };

    return (
        <MiniDrawer
            onFileUpload={handleFileUpload}
            onClearImages={handleClearImages}
            onGetResult={handleGetResult}
            onToggleAllDisplayModes={handleToggleAllDisplayModes}
        >
            <MainWorkArea
                uploadedImageFiles={uploadedImageFiles}
                onDeleteImage={handleDeleteImage}
                predictionResults={predictionResults}
                showResults={showResults}
                displayModes={displayModes}
                setDisplayModes={setDisplayModes}
            />
        </MiniDrawer>
    );
}

export default App;