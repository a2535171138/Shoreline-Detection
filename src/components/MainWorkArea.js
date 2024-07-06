import React, { useState } from 'react';
import { ClipLoader } from 'react-spinners';
import Modal from 'react-bootstrap/Modal';

function MainWorkArea({ uploadedImageFiles, predictionResults, showResults }) {
    const [showModal, setShowModal] = useState(false);
    const [modalImage, setModalImage] = useState('');

    const handleImageClick = (imageSrc) => {
        setModalImage(imageSrc);
        setShowModal(true);
    };

    return (
        <div className="col-9 right-panel p-3 d-flex flex-column">
            <h1 className="text-center mb-4">Main working part</h1>
            <div className="image-display border p-3 mb-4 flex-grow-1 overflow-auto">
                {!showResults ? (
                    uploadedImageFiles.length > 0 ? (
                        <div className="row">
                            {uploadedImageFiles.map((image, index) => (
                                <div key={index} className="col-3 mb-3">
                                    <div className="card">
                                        <img src={image.url} alt={image.file.name} className="card-img-top" />
                                        <div className="card-body">
                                            <h5 className="card-title">{image.file.name}</h5>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="placeholder d-flex justify-content-center align-items-center flex-grow-1">
                            <p>Please upload images</p>
                        </div>
                    )
                ) : (
                    uploadedImageFiles.map((image, index) => (
                        <div key={index} className="result-row mb-4">
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
                    ))
                )}
            </div>

            <Modal show={showModal} onHide={() => setShowModal(false)} size="lg" centered>
                <Modal.Body>
                    <img src={modalImage} alt="Enlarged" style={{width: '100%', height: 'auto'}} />
                </Modal.Body>
            </Modal>
        </div>
    );
}

export default MainWorkArea;