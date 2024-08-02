import React, { useState, useEffect, useCallback } from 'react';
import { ClipLoader } from 'react-spinners';
import Modal from 'react-bootstrap/Modal';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Typography from '@mui/material/Typography';
import CardActionArea from '@mui/material/CardActionArea';
import CardActions from '@mui/material/CardActions';
import Button from '@mui/material/Button';
import DeleteRoundedIcon from '@mui/icons-material/DeleteRounded';
import DownloadRoundedIcon from '@mui/icons-material/DownloadRounded';
import IconButton from '@mui/material/IconButton';
import ZoomInRoundedIcon from '@mui/icons-material/ZoomInRounded';
import SwapHorizIcon from '@mui/icons-material/SwapHoriz';
import './MainWorkArea.css';
import Fab from '@mui/material/Fab';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';
import Zoom from '@mui/material/Zoom';
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import CloseIcon from '@mui/icons-material/Close';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
// import {useLog} from "../LogContext";

function MainWorkArea({ uploadedImageFiles, predictionResults, showResults, onDeleteImage, displayModes, setDisplayModes,currentView,onFileUpload }) {
    // const { addLog } = useLog();
    const [showModal, setShowModal] = useState(false);
    const [modalImage, setModalImage] = useState('');
    const [showScroll, setShowScroll] = useState(false);
    const [anchorEl, setAnchorEl] = useState(null);
    const [downloadIndex, setDownloadIndex] = useState(null);
    const transformComponentRef = React.useRef(null);

    const handleImageClick = (imageSrc) => {
        setModalImage(imageSrc);
        setShowModal(true);
    };

    const truncateFileName = (fileName, maxLength = 50) => {
        if (fileName.length > maxLength) {
            return fileName.substring(0, maxLength) + '...';
        }
        return fileName;
    };

    const formatDate = (date) => {
        const parsedDate = new Date(date);
        return isNaN(parsedDate) ? 'Invalid Date' : parsedDate.toLocaleString();
    };

    const checkScrollTop = useCallback(() => {
        if (!showScroll && window.pageYOffset > 300) {
            setShowScroll(true);
        } else if (showScroll && window.pageYOffset <= 300) {
            setShowScroll(false);
        }
    }, [showScroll]);

    const scrollToTop = () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    useEffect(() => {
        window.addEventListener('scroll', checkScrollTop);
        return () => {
            window.removeEventListener('scroll', checkScrollTop);
        };
    }, [checkScrollTop]);

    const handleDisplayModeChange = (index) => {
        setDisplayModes(prevModes => ({
            ...prevModes,
            [index]: prevModes[index] === 'binary' ? 'color' : 'binary'
        }));
    };

    const getDisplayImage = (result, index) => {
        if (!result || result.error) return null;
        return displayModes[index] === 'color' ? result.colorResult : result.binaryResult;
    };

    const handleDownloadClick = (event, index) => {
        setAnchorEl(event.currentTarget);
        setDownloadIndex(index);
    };

    const handleDownloadClose = () => {
        setAnchorEl(null);
        setDownloadIndex(null);
    };

    const downloadImage = (dataUrl, fileName) => {
        const link = document.createElement('a');
        link.href = dataUrl;
        link.download = fileName;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const downloadCSV = (filename, pixelsResult) => {
        // 创建CSV内容，添加所有列的标题
        const csvContent = "path,rectified site,camera,type,obstructi,downward,low,shadow,label\n" +
            `${filename},,,,,,,,${pixelsResult}`;

        // 创建 Blob 对象
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });

        // 创建下载链接
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', `${filename}_result.csv`);

        // 触发下载
        document.body.appendChild(link);
        link.click();

        // 清理
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
    };

    const handleDownload = (type) => {
        if (downloadIndex === null || !predictionResults[downloadIndex]) return;
        const result = predictionResults[downloadIndex];
        const fileName = uploadedImageFiles[downloadIndex].file.name;

        switch(type) {
            case 'binary':
                if (result.binaryResult) downloadImage(result.binaryResult, `${fileName}_binary.png`);
                break;
            case 'color':
                if (result.colorResult) downloadImage(result.colorResult, `${fileName}_color.png`);
                break;
            case 'pixels':
                if (result.pixelsResult) downloadCSV(fileName, result.pixelsResult);
                break;
            case 'all':
                if (result.binaryResult) downloadImage(result.binaryResult, `${fileName}_binary.png`);
                if (result.colorResult) downloadImage(result.colorResult, `${fileName}_color.png`);
                if (result.pixelsResult) downloadCSV(fileName, result.pixelsResult);
                break;
            default:
                console.error(`Unknown download type: ${type}`);
        }
        handleDownloadClose();
    };


    return (
        <div className="container-fluid p-3 d-flex flex-column">
            <div className="image-display border p-3 mb-4 flex-grow-1 overflow-auto">
                {currentView === 'upload' ? (
                    uploadedImageFiles.length > 0 ? (
                        <div className="row">
                            {uploadedImageFiles.map((image, index) => (
                                <div key={index} className="col-3 mb-3">
                                    <Card>
                                        <CardActionArea onClick={() => handleImageClick(image.url)}>
                                            <CardMedia
                                                component="img"
                                                height="140"
                                                image={image.url}
                                                alt={image.file.name}
                                            />
                                            <CardContent>
                                                <Typography gutterBottom variant="h5" component="div">
                                                    {truncateFileName(image.file.name)}
                                                </Typography>
                                                <Typography variant="body2" color="text.secondary">
                                                    Uploaded on: {new Date(image.file.lastModified).toLocaleString()} <br />
                                                </Typography>
                                            </CardContent>
                                        </CardActionArea>
                                        <CardActions>
                                            <div style={{ flex: 1 }} />
                                            <IconButton color="primary" onClick={() => handleImageClick(image.url)}>
                                                <ZoomInRoundedIcon />
                                            </IconButton>
                                            <IconButton color="error" onClick={() => onDeleteImage(index)} aria-label="delete image">
                                                <DeleteRoundedIcon />
                                            </IconButton>
                                        </CardActions>
                                    </Card>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div>
                            <p>Please upload images to begin shoreline detection.</p>
                        </div>
                    )
                ) : (
                    <div className="result-container">
                        {uploadedImageFiles.some((_, index) => predictionResults[index] !== null && predictionResults[index] !== 'error') && (
                            <h3 className="text-center mb-4">Shoreline Detection Results</h3>
                        )}
                        {uploadedImageFiles.map((image, index) => (
                            <React.Fragment key={index}>
                                <div className="d-flex justify-content-between align-items-center mb-2">
                                    <Typography variant="h5">Image {index + 1}</Typography>
                                    <Button size="small" variant="outlined" color="error" startIcon={<DeleteRoundedIcon />} onClick={() => onDeleteImage(index)}>
                                        Delete Result
                                    </Button>
                                </div>
                                <div className="result-row mb-4">
                                    <div className="result-image">
                                        <Card>
                                            <CardActionArea onClick={() => handleImageClick(image.url)}>
                                                <CardMedia
                                                    component="img"
                                                    height="140"
                                                    image={image.url}
                                                    alt={`Original ${index + 1}`}
                                                />
                                                <CardContent>
                                                    <Typography gutterBottom variant="h6" component="div">
                                                        Original Image
                                                    </Typography>
                                                    <Typography variant="body2" color="text.secondary">
                                                        File Name: {truncateFileName(image.file.name)} <br />
                                                        Uploaded on: {new Date(image.file.lastModified).toLocaleString()}
                                                    </Typography>
                                                </CardContent>
                                                <CardActions>
                                                    <div style={{ flex: 1 }} />
                                                    <IconButton color="primary" onClick={() => handleImageClick(image.url)}>
                                                        <ZoomInRoundedIcon />
                                                    </IconButton>
                                                </CardActions>
                                            </CardActionArea>
                                        </Card>
                                    </div>
                                    <div className="result-image">
                                        {predictionResults[index] === null ? (
                                            <div className="processing-placeholder">
                                                <ClipLoader color="#007bff" loading={true} size={50} />
                                                <p>Processing...</p>
                                            </div>
                                        ) : predictionResults[index].error ? (
                                            <div className="error-placeholder">
                                                <p>{predictionResults[index].error}</p>
                                            </div>
                                        ) : (
                                            <Card>
                                                <CardActionArea onClick={() => handleImageClick(getDisplayImage(predictionResults[index], index))}>
                                                    <CardMedia
                                                        component="img"
                                                        height="140"
                                                        image={getDisplayImage(predictionResults[index], index)}
                                                        alt={`Processed ${index + 1}`}
                                                    />
                                                    <CardContent>
                                                        <Typography gutterBottom variant="h6" component="div">
                                                            Processed Image ({displayModes[index] || 'binary'})
                                                        </Typography>
                                                        <Typography variant="body2" color="text.secondary">
                                                            Detected Shoreline for: {truncateFileName(image.file.name)} <br />
                                                            Processed on: {formatDate(predictionResults[index].processingTime)}<br />
                                                            Confidence: { predictionResults[index].confidence?.toFixed(4) ?? 'N/A'}
                                                        </Typography>
                                                    </CardContent>
                                                </CardActionArea>
                                                <CardActions>
                                                    <IconButton color="primary" onClick={() => handleDisplayModeChange(index)} aria-label="switch to the other image">
                                                        <SwapHorizIcon />
                                                    </IconButton>
                                                    <div style={{ flex: 1 }} />
                                                    <IconButton color="primary" onClick={() => handleImageClick(getDisplayImage(predictionResults[index], index))}>
                                                        <ZoomInRoundedIcon />
                                                    </IconButton>
                                                    <Button
                                                        size="small"
                                                        variant="outlined"
                                                        startIcon={<DownloadRoundedIcon />}
                                                        onClick={(event) => handleDownloadClick(event, index)}
                                                        aria-label="download options"
                                                    >
                                                        Download
                                                    </Button>
                                                </CardActions>
                                            </Card>
                                        )}
                                    </div>
                                </div>
                                {index < uploadedImageFiles.length - 1 && <hr className="divider" />}
                            </React.Fragment>
                        ))}
                    </div>
                )}
            </div>
            <Modal
                show={showModal}
                onHide={() => setShowModal(false)}
                size="xl"
                centered
                fullscreen
            >
                <Modal.Header closeButton>
                    <Modal.Title>Image View</Modal.Title>
                </Modal.Header>
                <Modal.Body style={{ height: 'calc(100vh - 56px)', padding: 0, position: 'relative', overflow: 'hidden' }}>
                    <TransformWrapper
                        initialScale={1}
                        centerOnInit={true}
                        minScale={0.1}
                        maxScale={8}
                        ref={transformComponentRef}
                    >
                        {({ zoomIn, zoomOut, resetTransform }) => (
                            <>
                                <TransformComponent
                                    wrapperStyle={{
                                        width: '100%',
                                        height: '100%',
                                    }}
                                    contentStyle={{
                                        width: '100%',
                                        height: '100%',
                                        display: 'flex',
                                        justifyContent: 'center',
                                        alignItems: 'center',
                                    }}
                                >
                                    <div style={{
                                        display: 'flex',
                                        justifyContent: 'center',
                                        alignItems: 'center',
                                        width: '100%',
                                        height: '100%',
                                    }}>
                                        <img
                                            src={modalImage}
                                            alt="Enlarged"
                                            style={{
                                                maxWidth: '100%',
                                                maxHeight: '100%',
                                                objectFit: 'contain',
                                            }}
                                        />
                                    </div>
                                </TransformComponent>

                                {/* Control Bar */}
                                <div style={{
                                    position: 'absolute',
                                    bottom: 20,
                                    left: '50%',
                                    transform: 'translateX(-50%)',
                                    backgroundColor: 'rgba(0,0,0,0.5)',
                                    borderRadius: 20,
                                    padding: '5px 10px',
                                    display: 'flex',
                                    gap: '10px',
                                    zIndex: 1000,
                                }}>
                                    <IconButton onClick={() => zoomOut()} color="primary">
                                        <ZoomOutIcon />
                                    </IconButton>
                                    <IconButton onClick={() => resetTransform()} color="primary">
                                        <RestartAltIcon />
                                    </IconButton>
                                    <IconButton onClick={() => zoomIn()} color="primary">
                                        <ZoomInIcon />
                                    </IconButton>
                                    <IconButton onClick={() => setShowModal(false)} color="primary">
                                        <CloseIcon />
                                    </IconButton>
                                </div>
                            </>
                        )}
                    </TransformWrapper>
                </Modal.Body>
            </Modal>
            <Menu
                anchorEl={anchorEl}
                open={Boolean(anchorEl)}
                onClose={handleDownloadClose}
            >
                <MenuItem onClick={() => handleDownload('binary')}>Download Binary Image</MenuItem>
                <MenuItem onClick={() => handleDownload('color')}>Download Color Image</MenuItem>
                <MenuItem onClick={() => handleDownload('pixels')}>Download Pixel Data (CSV)</MenuItem>
                <MenuItem onClick={() => handleDownload('all')}>Download All File</MenuItem>
            </Menu>
            <Zoom in={showScroll}>
                <Fab
                    color="primary"
                    size="large"
                    onClick={scrollToTop}
                    sx={{ position: 'fixed', bottom: 16, right: 16 }}
                >
                    <KeyboardArrowUpIcon />
                </Fab>
            </Zoom>
        </div>
    );
}

export default MainWorkArea;