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
import './MainWorkArea.css';
import Fab from '@mui/material/Fab';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';
import Zoom from '@mui/material/Zoom';

function MainWorkArea({ uploadedImageFiles, predictionResults, showResults, onDeleteImage  }) {
    const [showModal, setShowModal] = useState(false);
    const [modalImage, setModalImage] = useState('');
    const [showScroll, setShowScroll] = useState(false);

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

    return (
        <div className="container-fluid p-3 d-flex flex-column">
            <div className="image-display border p-3 mb-4 flex-grow-1 overflow-auto">
                {!showResults ? (
                    uploadedImageFiles.length > 0 ? (
                        <div className="row">
                            {uploadedImageFiles.map((image, index) => (
                                <div key={index} className="col-3 mb-3">
                                    <Card>
                                        <CardActionArea>
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
                                            <CardActions>
                                                <div style={{ flex: 1 }} />
                                                <IconButton color="primary" onClick={() => handleImageClick(image.url)}>
                                                    <ZoomInRoundedIcon />
                                                </IconButton>
                                                <IconButton color="error" onClick={() => onDeleteImage(index)}>
                                                    <DeleteRoundedIcon />
                                                </IconButton>
                                            </CardActions>
                                        </CardActionArea>
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
                    <div>
                        {uploadedImageFiles.some((_, index) => predictionResults[index] !== null && predictionResults[index] !== 'error') && (
                            <h3 className="text-center mb-4">Shoreline Detection Results</h3>
                        )}
                        {uploadedImageFiles.map((image, index) => (
                            <React.Fragment key={index}>
                                <div className="d-flex justify-content-between align-items-center mb-2">
                                    <Typography variant="h5">Image {index + 1}</Typography>
                                    <Button size="small" variant="outlined" color="error" startIcon={<DeleteRoundedIcon />} onClick={() => onDeleteImage(index)}>
                                        Delete
                                    </Button>
                                </div>
                                <div className="result-row mb-4">
                                    <div className="result-image">
                                        <Card>
                                            <CardActionArea>
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
                                        ) : predictionResults[index] === 'error' ? (
                                            <div className="error-placeholder">
                                                <p>Error processing image</p>
                                            </div>
                                        ) : (
                                            <Card>
                                                <CardActionArea>
                                                    <CardMedia
                                                        component="img"
                                                        height="140"
                                                        image={predictionResults[index].result}
                                                        alt={`Processed ${index + 1}`}
                                                    />
                                                    <CardContent>
                                                        <Typography gutterBottom variant="h6" component="div">
                                                            Processed Image
                                                        </Typography>
                                                        <Typography variant="body2" color="text.secondary">
                                                            Detected Shoreline for: {truncateFileName(image.file.name)} <br />
                                                            Processed on: {formatDate(predictionResults[index].processingTime)}
                                                        </Typography>
                                                    </CardContent>
                                                    <CardActions>
                                                        <div style={{ flex: 1 }} />
                                                        <IconButton color="primary" onClick={() => handleImageClick(predictionResults[index].result)}>
                                                            <ZoomInRoundedIcon />
                                                        </IconButton>
                                                        <Button size="small" variant="outlined" startIcon={<DownloadRoundedIcon />}>Download</Button>
                                                    </CardActions>
                                                </CardActionArea>
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
            <Modal show={showModal} onHide={() => setShowModal(false)} size="lg" centered>
                <Modal.Body>
                    <img src={modalImage} alt="Enlarged" style={{width: '100%', height: 'auto'}} />
                </Modal.Body>
            </Modal>
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
