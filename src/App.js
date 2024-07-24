import React, {useRef, useState} from 'react';
import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import MiniDrawer from './components/MiniDrawer';
import MainWorkArea from './components/MainWorkArea';
import axios from 'axios';
import Alert from '@mui/material/Alert';
import Button from "@mui/material/Button";

function App() {
    const [uploadedImageFiles, setUploadedImageFiles] = useState([]);
    const [predictionResults, setPredictionResults] = useState([]);
    const [showResults, setShowResults] = useState(false);
    const [processedImages, setProcessedImages] = useState(new Set());
    const [displayModes, setDisplayModes] = useState({});
    const [isLoading, setIsLoading] = useState(false);
    const [duplicateImageWarning, setDuplicateImageWarning] = useState(false);
    const [qualityControlEnabled, setQualityControlEnabled] = useState(false);
    const [classificationEnabled, setClassificationEnabled] = useState(false);
    const [qualityCheckEnabled, setQualityCheckEnabled] = useState(false);
    const [hasResults, setHasResults] = useState(false);
    const [currentView, setCurrentView] = useState('upload'); // 'upload' 或 'results'

    const fileInputRef = useRef(null);

    const toggleQualityControl = async () => {
        try {
            const response = await axios.post('http://localhost:5000/toggle_quality_control');
            setQualityControlEnabled(response.data.enabled);
        } catch (error) {
            console.error('Error toggling quality control:', error);
        }
    };

    const toggleClassification = async () => {
        try {
            const response = await axios.post('http://localhost:5000/toggle_classification');
            setClassificationEnabled(response.data.enabled);
        } catch (error) {
            console.error('Error toggling classification:', error);
        }
    };

    const toggleQualityCheck = async () => {
        try {
            const response = await axios.post('http://localhost:5000/toggle_quality_check');
            setQualityCheckEnabled(response.data.enabled);
            return response.data.enabled;
        } catch (error) {
            console.error('Error toggling quality check:', error);
            return qualityCheckEnabled; // 保持原状态不变
        }
    };

    const handleFileUpload = (files) => {
        const imageFiles = Array.from(files)
            .filter(file => file.type.startsWith('image/'));

        const duplicateImages = imageFiles.filter(file =>
            uploadedImageFiles.some(uploadedFile =>
                uploadedFile.file.name === file.name && uploadedFile.file.size === file.size
            )
        );

        if (duplicateImages.length > 0) {
            setDuplicateImageWarning(true);
            setTimeout(() => setDuplicateImageWarning(false), 5000);
            return;
        }

        const newImageFiles = imageFiles
            .map(file => ({
                file,
                url: URL.createObjectURL(file),
                id: Date.now() + Math.random()
            }));

        if (newImageFiles.length > 0) {
            setUploadedImageFiles(prevFiles => [...prevFiles, ...newImageFiles]);
            setPredictionResults(prevResults => [...prevResults, ...new Array(newImageFiles.length).fill(null)]);
        }
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

    const handleDeleteImage = async (index) => {
        const file = uploadedImageFiles[index];
        try {
            // 向后端发送删除请求
            await axios.delete(`http://localhost:5000/delete_result/${file.file.name}`);
        } catch (error) {
            console.error('Error deleting prediction result:', error);
        }
        setUploadedImageFiles(prevFiles => {
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

    const handleClearImages = async () => {
        try {
            await axios.post('http://localhost:5000/clear_results');
            setUploadedImageFiles(prevFiles => {
                prevFiles.forEach(file => URL.revokeObjectURL(file.url));
                return [];
            });
            setPredictionResults([]);
            setProcessedImages(new Set());
            setShowResults(false);
        } catch (error) {
            console.error('Error clearing results:', error);
        }
        setHasResults(false);
    };

    const handleGetResult = async () => {
        if (uploadedImageFiles.length > 0) {
            setShowResults(true);
            setCurrentView('results');
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
                            ...newPredictionResults[i],
                            binaryResult: `data:image/png;base64,${response.data.binary_result}`,
                            colorResult: `data:image/png;base64,${response.data.color_result}`,
                            pixelsResult: response.data.pixels_result,
                            processingTime: response.data.processingTime,
                        };

                        setProcessedImages(prev => new Set(prev).add(imageFile.id));
                    } catch (error) {
                        console.error('Error fetching prediction result:', error);
                        newPredictionResults[i] = {
                            error: error.response?.data?.error || 'Unknown error occurred'
                        };
                    }
                    setPredictionResults([...newPredictionResults]);
                    if (newPredictionResults.some(result => result !== null && result !== 'error')) {
                        setHasResults(true);
                    }
                }
            }
        }
    };
    const switchToUploadView = () => {
        setCurrentView('upload');
        setShowResults(false);
    };

    const handleDownloadAll = async (type) => {
        setIsLoading(true);
        try {
            const filenames = uploadedImageFiles.map(file => file.file.name);
            const response = await axios.get(`http://localhost:5000/download_all/${type}`, {
                params: { filenames },
                responseType: 'blob'
            });

            // 检查是否有结果可下载
            if (response.status === 400) {
                throw new Error('No results to download');
            }

            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            if (type === 'pixels') {
                link.setAttribute('download', 'all_pixels_results.csv');
            } else {
                link.setAttribute('download', `all_${type}_results.zip`);
            }
            document.body.appendChild(link);
            link.click();
            link.remove();
        } catch (error) {
            console.error(`Error downloading ${type} results:`, error);
            // 显示错误消息给用户
            alert(error.message || `Error downloading ${type} results`);
        }
        setIsLoading(false);
    };

    return (
        <MiniDrawer
            onFileUpload={() => fileInputRef.current.click()}
            onClearImages={handleClearImages}
            onGetResult={handleGetResult}
            onToggleAllDisplayModes={handleToggleAllDisplayModes}
            onDownloadAll={handleDownloadAll}
            onToggleQualityCheck={async () => {
                const newState = await toggleQualityCheck();
                setQualityCheckEnabled(newState);
            }}
            qualityCheckEnabled={qualityCheckEnabled}
            hasResults={hasResults}
            onSwitchView={() => setCurrentView(currentView === 'upload' ? 'results' : 'upload')}
            currentView={currentView}
        >
            {duplicateImageWarning && (
                <Alert severity="warning" onClose={() => setDuplicateImageWarning(false)}>
                    Duplicate image(s) detected. Please select a different image.
                </Alert>
            )}

            <input
                type="file"
                multiple
                onChange={(e) => handleFileUpload(e.target.files)}
                style={{ display: 'none' }}
                ref={fileInputRef}
                accept="image/*"
            />

            <MainWorkArea
                uploadedImageFiles={uploadedImageFiles}
                onDeleteImage={handleDeleteImage}
                predictionResults={predictionResults}
                showResults={showResults}
                displayModes={displayModes}
                setDisplayModes={setDisplayModes}
                isLoading={isLoading}
                currentView={currentView}
                onFileUpload={() => fileInputRef.current.click()}
            />
        </MiniDrawer>
    );
}


export default App;