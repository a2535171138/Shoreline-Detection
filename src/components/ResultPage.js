import React, { useState } from 'react';
import { ClipLoader } from 'react-spinners';
import Modal from 'react-bootstrap/Modal';

function ResultsPage({ uploadedImageFiles, predictionResults, onBack }) {
    const [showModal, setShowModal] = useState(false);
    const [modalImage, setModalImage] = useState('');

    const handleImageClick = (imageSrc) => {
        setModalImage(imageSrc);
        setShowModal(true);
    };

    return (
        <div className="results-page">
            <div className="results-container">
                <button onClick={onBack} className="btn btn-secondary mb-3 back-button">Back</button>
                {uploadedImageFiles.map((image, index) => (
                    <div key={index} className="result-row">
                        <div className="result-image">
                            <img
                                src={image.url}
                                alt={`Original ${index + 1}`}
                                onClick={() => handleImageClick(image.url)}
                                style={{cursor: 'pointer'}}
                            />
                            <p className="mt-2 text-center">Original</p>
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
                                <>
                                    <img
                                        src={predictionResults[index]}
                                        alt={`Processed ${index + 1}`}
                                        onClick={() => handleImageClick(predictionResults[index])}
                                        style={{cursor: 'pointer'}}
                                    />
                                    <p className="mt-2 text-center">Processed</p>
                                </>
                            )}
                        </div>
                    </div>
                ))}
            </div>

            <Modal show={showModal} onHide={() => setShowModal(false)} size="lg" centered>
                <Modal.Body>
                    <img src={modalImage} alt="Enlarged" style={{width: '100%', height: 'auto'}} />
                </Modal.Body>
            </Modal>
        </div>
    );
}

export default ResultsPage;