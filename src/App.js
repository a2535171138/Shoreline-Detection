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

    const handleFileUpload = (files) => {
        const imageFiles = Array.from(files).map(file => ({
            file,
            url: URL.createObjectURL(file),
            id: Date.now() + Math.random()
        }));
        setUploadedImageFiles(prevFiles => [...prevFiles, ...imageFiles]);
        setPredictionResults(prevResults => [...prevResults, ...new Array(files.length).fill(null)]);
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
                        const resultImage = `data:image/png;base64,${response.data.result}`;
                        const processingTime = response.data.processingTime;

                        newPredictionResults[i] = {
                            result: resultImage,
                            processingTime: processingTime,
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
        >
            <MainWorkArea
                uploadedImageFiles={uploadedImageFiles}
                onDeleteImage={handleDeleteImage}
                predictionResults={predictionResults}
                showResults={showResults}
            />
        </MiniDrawer>
    );
}

export default App;
