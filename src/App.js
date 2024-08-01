import React, {useEffect, useRef, useState,useCallback} from 'react';
import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import MiniDrawer from './components/MiniDrawer';
import MainWorkArea from './components/MainWorkArea';
import axios from 'axios';
import Alert from '@mui/material/Alert';
import { LogProvider, useLog } from './LogContext';
import SnackbarLog from './components/SnackbarLog';
import LogViewer from "./components/LogViewer";
import UserGuide from './components/UserGuide';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';

function AppContent() {
    const { addLog } = useLog();
    const [uploadedImageFiles, setUploadedImageFiles] = useState([]);
    const [predictionResults, setPredictionResults] = useState([]);
    const [showResults, setShowResults] = useState(false);
    const [processedImages, setProcessedImages] = useState(new Set());
    const [displayModes, setDisplayModes] = useState({});
    const [isLoading, setIsLoading] = useState(false);
    const [duplicateImageWarning, setDuplicateImageWarning] = useState(false);
    const [qualityCheckEnabled, setQualityCheckEnabled] = useState(false);
    const [hasResults, setHasResults] = useState(false);
    const [currentView, setCurrentView] = useState('upload');
    const [logViewerOpen, setLogViewerOpen] = useState(false);
    const [modelStatus, setModelStatus] = useState({});
    const [isInitialized, setIsInitialized] = useState(false);
    const [isDownloading, setIsDownloading] = useState(false);

    useEffect(() => {
        const initialize = async () => {
            if (!isInitialized) {
                try {
                    const response = await axios.get('http://localhost:5000/initialize');
                    setModelStatus(response.data);
                    setIsInitialized(true);

                    // 检查是否有正在下载的模型
                    const downloading = Object.values(response.data).includes("Downloading");
                    setIsDownloading(downloading);

                    // 处理模型状态并添加日志
                    Object.entries(response.data).forEach(([model, status]) => {
                        if (status === "Downloaded") {
                            addLog(`Model ${model} has been successfully downloaded.`, 'success');
                        } else if (status === "Download failed") {
                            addLog(`Failed to download model ${model}.`, 'error');
                        } else if (status === "Already exists") {
                            addLog(`Model ${model} is already available.`, 'info');
                        } else if (status === "Downloading") {
                            addLog(`Downloading model ${model}...`, 'info');
                        }
                    });

                    // 如果有模型正在下载，开始轮询
                    if (downloading) {
                        pollModelStatus();
                    }
                } catch (error) {
                    console.error('Failed to initialize:', error);
                    addLog('Failed to initialize application', 'error');
                }
            }
        };

        initialize();
    }, [addLog, isInitialized]);

    const pollModelStatus = useCallback(() => {
        const poll = setInterval(async () => {
            try {
                const response = await axios.get('http://localhost:5000/model_status');
                setModelStatus(response.data);

                const downloading = Object.values(response.data).includes("Downloading");
                setIsDownloading(downloading);

                if (!downloading) {
                    clearInterval(poll);
                    addLog('All models have finished downloading.', 'success');
                }

                Object.entries(response.data).forEach(([model, status]) => {
                    if (status === "Downloaded" && modelStatus[model] !== "Downloaded") {
                        addLog(`Model ${model} has been successfully downloaded.`, 'success');
                    } else if (status === "Download failed" && modelStatus[model] !== "Download failed") {
                        addLog(`Failed to download model ${model}.`, 'error');
                    }
                });
            } catch (error) {
                console.error('Failed to get model status:', error);
                clearInterval(poll);
            }
        }, 5000); // 每5秒轮询一次

        return () => clearInterval(poll);
    }, [addLog, modelStatus]);

    const handleViewLogs = () => {
        setLogViewerOpen(true);
    };

    const handleCloseLogViewer = () => {
        setLogViewerOpen(false);
    };

    const fileInputRef = useRef(null);

    const toggleQualityCheck = async () => {
        try {
            const response = await axios.post('http://localhost:5000/toggle_quality_check');
            setQualityCheckEnabled(response.data.enabled);
            addLog(`Quality check ${response.data.enabled ? 'enabled' : 'disabled'}`, 'info');
            return response.data.enabled;
        } catch (error) {
            addLog('Failed to toggle quality check', 'error');
            console.error('Error toggling quality check:', error);
            return qualityCheckEnabled;
        }
    };

    const handleFileUpload = (files) => {
        setCurrentView('upload');
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
            addLog(`Uploaded ${imageFiles.length} image(s)`, 'success');
        } else {
            addLog('No valid image files were selected', 'warning');
        }
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
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
        addLog(`Deleted image: ${file.file.name}`, 'info');
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
            setHasResults(false);
            addLog('All images and results cleared', 'info');
        } catch (error) {
            console.error('Error clearing results:', error);
            addLog('Failed to clear all images and results', 'error');
        }
    };

    const handleGetResult = async (scene) => {
        setCurrentView('results');
        if (uploadedImageFiles.length === 0) {
            addLog("No images to process", "warning");
            return;
        }

        addLog(`Starting image processing for ${scene}`, "info");
        setShowResults(true);
        let successCount = 0;
        let failCount = 0;

        for (let i = 0; i < uploadedImageFiles.length; i++) {
            const imageFile = uploadedImageFiles[i];
            if (!processedImages.has(imageFile.id)) {
                const formData = new FormData();
                formData.append('file', imageFile.file);
                formData.append('scene', scene);

                try {
                    addLog(`Processing image: ${imageFile.file.name}`, "info");
                    const response = await axios.post(`http://localhost:5000/predict/${scene}`, formData, {
                        headers: {
                            'Content-Type': 'multipart/form-data'
                        }
                    });

                    setPredictionResults(prevResults => {
                        const newResults = [...prevResults];
                        newResults[i] = {
                            binaryResult: `data:image/png;base64,${response.data.binary_result}`,
                            colorResult: `data:image/png;base64,${response.data.color_result}`,
                            pixelsResult: response.data.pixels_result,
                            processingTime: response.data.processingTime,
                            confidence: parseFloat(response.data.confidence)
                        };
                        return newResults;
                    });

                    setProcessedImages(prev => new Set(prev).add(imageFile.id));
                    addLog(`Successfully processed image: ${imageFile.file.name}`, "success");
                    successCount++;
                } catch (error) {
                    console.error('Error fetching prediction result:', error);
                    setPredictionResults(prevResults => {
                        const newResults = [...prevResults];
                        newResults[i] = {
                            error: error.response?.data?.error || 'Unknown error occurred'
                        };
                        return newResults;
                    });
                    addLog(`Failed to process image: ${imageFile.file.name}. ${error.response?.data?.error || 'Unknown error occurred'}`, "error");
                    failCount++;
                }
            } else {
                addLog(`Image already processed: ${imageFile.file.name}`, "info");
            }
        }

        setHasResults(true);
        addLog(`Processing complete. Successfully processed: ${successCount}, Failed: ${failCount}`, "info");
    };

    const handleDownloadAll = async (type) => {
        setIsLoading(true);
        try {
            const filenames = uploadedImageFiles.map(file => file.file.name);
            const response = await axios.get(`http://localhost:5000/download_all/${type}`, {
                params: { filenames },
                responseType: 'blob'
            });

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
            addLog(`Downloaded all ${type} results`, 'success');
        } catch (error) {
            console.error(`Error downloading ${type} results:`, error);
            alert(error.message || `Error downloading ${type} results`);
            addLog(`Failed to download ${type} results`, 'error');
        }
        setIsLoading(false);
    };

    return (
        <Router>
            <Routes>
                <Route path="/" element={
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
                        onViewLogs={handleViewLogs}
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
                            isDownloading={isDownloading}
                            modelStatus={modelStatus}

                        />
                        <LogViewer open={logViewerOpen} onClose={handleCloseLogViewer} />
                    </MiniDrawer>
                } />
                <Route path="/user-guide" element={<UserGuide />} />
            </Routes>
        </Router>
    );
}

function App() {
    return (
        <LogProvider>
            <AppContent />
            <SnackbarLog />
        </LogProvider>
    );
}

export default App;