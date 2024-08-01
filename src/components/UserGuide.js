// src/components/UserGuide.js

import React, { useState } from 'react';
import { Container, Typography, Box, Card, CardContent, CardMedia, Grid } from '@mui/material';
import UploadIcon from '@mui/icons-material/Upload';
import CheckCircleRoundedIcon from '@mui/icons-material/CheckCircleRounded';
import DeleteSweepRoundedIcon from '@mui/icons-material/DeleteSweepRounded';
import SwapHorizIcon from '@mui/icons-material/SwapHoriz';
import DownloadIcon from '@mui/icons-material/Download';
import './UserGuide.css';
import Button from '@mui/material/Button';
import CardActions from '@mui/material/CardActions';
import { Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import VisibilityIcon from '@mui/icons-material/Visibility';
import FilterIcon from '@mui/icons-material/Filter';
import HistoryIcon from '@mui/icons-material/History';


function UserGuide() {
    const [open, setOpen] = useState(false);
    const [dialogContent, setDialogContent] = useState('');
    const navigate = useNavigate();

    const handleClickOpen = (content) => {
        setDialogContent(content);
        setOpen(true);
    };

    const handleClose = () => {
        setOpen(false);
        setDialogContent('');
    };

    const handleBackToHome = () => {
        navigate('/'); 
    };

    const viewSwitchContent = (
        <>
            <Typography variant="body1" paragraph>
            The <strong>"View Switch" feature</strong> allows you to toggle between different views in the application, providing flexibility in how you analyze and interact with the images and their processed results. Here are some key points about this functionality:
            </Typography>
            <Typography variant="body1" paragraph>
            <strong>Purpose:</strong> The "View Switch" feature is designed to enhance user experience by allowing seamless switching between the view of uploaded images and the processed results. 
            </Typography>
            <Typography variant="body1" paragraph>
            <strong>How to Use:</strong> To use the "View Switch" feature, simply click on the toggle button. When the toggle is in the "Uploaded Images" position, you will see all the images you have uploaded. When switched to the "Results View" position, you will see the processed shoreline detection results.
            </Typography>
            <Typography variant="body1" paragraph>
            <strong>Visual Indicator:</strong> The toggle switch provides a clear visual indicator of the current view. When switched to the left, it shows the uploaded images, and when switched to the right, it shows the results view.
            </Typography>
            <Typography variant="body1" paragraph>
            <strong>Tooltip:</strong> Hovering over the "View Switch" toggle will display a tooltip that provides a brief description of the feature, ensuring that users are aware of its functionality.
            </Typography>
            <Typography variant="body1" paragraph>
            <strong>Use Cases:</strong> This feature is useful for quickly verifying the upload status of images, reviewing the results without re-uploading images.
            </Typography>
            <Typography variant="body1" paragraph>
            <strong>Efficiency:</strong> The "View Switch" feature helps to improve workflow efficiency by reducing the need to navigate through different sections of the application manually. It allows for quick toggling, saving time and effort.
            </Typography>
        </>        
    );

    const qualityCheckContent = (
        <>
            <Typography variant="body1" paragraph>
                The <strong>"Quality Check" feature</strong> helps you ensure that only high-quality images are processed, enhancing the accuracy and reliability of the shoreline detection results. Here are some key points about this functionality:
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Purpose:</strong> The "Quality Check" feature is designed to filter out images that do not meet certain quality standards, such as low-resolution images or images that do not contain shorelines. This helps in maintaining the overall quality of the processed results.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>How to Use:</strong> To use the "Quality Check" feature, simply toggle the switch. When enabled, the application will automatically screen uploaded images based on predefined quality criteria and exclude those that are deemed unsuitable for processing.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Quality Criteria:</strong> The application checks for various factors such as image resolution, clarity, and the presence of shorelines. Images that do not meet these criteria will be flagged and excluded from the processing queue.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Visual Indicator:</strong> The toggle switch provides a clear visual indicator of the current state. When switched to the right, the "Quality Check" feature is enabled, and images will be screened. When switched to the left, the feature is disabled, and all images will be processed without screening.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Tooltip:</strong> Hovering over the "Quality Check" toggle will display a tooltip that provides a brief description of the feature, ensuring that users are aware of its functionality.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Use Cases:</strong> This feature is particularly useful when dealing with large batches of images, as it ensures that the processing resources are focused on images that are most likely to yield accurate results.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Efficiency:</strong> The "Quality Check" feature helps to improve workflow efficiency by reducing the need to manually inspect and filter images before processing. It automates the quality assurance process, ensuring consistent and reliable results.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Processing Time:</strong> Enabling the "Quality Check" feature will increase the processing time as the application needs to screen and process the images. Please be prepared for longer wait times when this feature is active.
            </Typography>
        </>
    );      

    const uploadImageContent = (
        <>
            <Typography variant="body1" paragraph>
                Click on the <strong>"Upload Image" button</strong> to select and upload images from your computer. Only image files are accepted.  Here are some key points about uploading images:
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Accepted File Types:</strong> The application supports JPG, PNG, and JPEG formats.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Single or Multiple Uploads:</strong> You can upload one image or multiple images at the same time.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>How to Upload:</strong>
                <Typography component="span" variant="body1">
                    <br/>- When you click on the "Upload Image" button, a file dialog will open, allowing you to select the images from your computer.
                    <br/>- After selecting the images, click "Open" to upload them.
                </Typography>
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>View Uploaded Images:</strong> Once uploaded, the images will be displayed in the application. You will see thumbnails of the uploaded images.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Continue Uploading:</strong> You can continue to upload more images even after the initial upload by clicking the "Upload Image" button again.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Zoom Uploaded Images:</strong> You can view a larger version of any uploaded image by clicking the images or the zoom icon (🔍) below the image thumbnail.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Delete Uploaded Images:</strong> You can delete any uploaded image by clicking the delete icon (🗑) below the image thumbnail.
            </Typography>
        </>
    );

    const getResultContent = (
        <>
            <Typography variant="body1" paragraph>
                After uploading images, click on the <strong>"Get Result" button</strong> to start the shoreline detection process. Here are some key points about getting the results:
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Processing Time:</strong> The application may take some time to process the images, depending on the number and size of the images uploaded.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>View Results:</strong> Once the processing is complete, the results will be displayed to the right of the uploaded images. You will see the processed shoreline images in binary format along with the original ones. Each processed image will be displayed alongside its original image.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Result Formats:</strong> The application provides results in both binary and color formats. You can toggle between these display modes.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Zoom Images:</strong> You can view a larger version of any image (both original and processed) by clicking the images or the zoom icon (🔍) below the image thumbnail.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Download Results:</strong> To download the processed results, click the download icon (📥) below the processed image thumbnail. A menu will appear with the following options:
                <ul>
                    <li><strong>Download Binary Image:</strong> Downloads the processed image in binary format.</li>
                    <li><strong>Download Color Image:</strong> Downloads the processed image in color format.</li>
                    <li><strong>Download Pixel Data (CSV):</strong> Downloads the pixel data of the processed image in CSV format.</li>
                    <li><strong>Download All Files:</strong> Downloads all available formats for the processed image in a single zip file.</li>
                </ul>
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Delete Images:</strong> You can delete any uploaded or processed image by clicking the delete icon (🗑) below the image thumbnail. This will remove the image and its associated results from the application.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Error Handling:</strong> If there is an error during the processing, a message will be displayed indicating the issue. You can try re-uploading the image or checking the image format and size.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Clear Results:</strong> If you want to start over, you can click the "Clear Image" button to remove all the uploaded images and results.
            </Typography>
        </>
    );
    
    const clearImageContent = (
        <>
            <Typography variant="body1" paragraph>
                Click on the <strong>"Clear Image" button</strong> to remove all the uploaded images and reset the application state. Here are some key points about this functionality:
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Reset Application State:</strong> This action will clear all uploaded images, processed results, and any other data from the application, bringing it back to its initial state.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Remove Processed Results:</strong> Any results generated from previous image uploads will be permanently deleted.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Use Cases:</strong> Use this feature if you want to start over with a new set of images or if you encounter any issues with the current uploads and results.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Warning:</strong> Once you click the "Clear Image" button, all data will be permanently removed. Ensure you have saved any necessary results before using this feature.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Confirmation and Logging:</strong> When you click the "Clear Image" button, a confirmation prompt will appear to prevent accidental deletion of data. Additionally, an action log will record the details of the deleted content, including the time of deletion and the files that were removed.
            </Typography>
        </>
    );

    const toggleAllContent = (
        <>
            <Typography variant="body1" paragraph>
                The <strong>"Toggle All"</strong> button allows you to switch between different display modes for all the results at once. Here are some key points about this functionality:
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>How to Use:</strong> Click the "Toggle All" button to switch all images between binary and color display modes. The application will automatically update the display for all processed images.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Use Cases:</strong> This feature is particularly useful when analyzing multiple images and you need to view all results in a consistent format quickly.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Efficiency:</strong> The "Toggle All" button saves time and effort by allowing you to change the display mode for all images with a single click, rather than toggling each image individually.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Note:</strong> The display mode can be toggled back and forth as many times as needed to suit your analysis requirements.
            </Typography>
        </>
    );

    const downloadAllContent = (
        <>
            <Typography variant="body1" paragraph>
                Click on the <strong>"Download All"</strong> button to download all the results in different formats. This feature provides a convenient way to obtain all processed results in one go. Here are some key points about downloading results:
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Download Options:</strong> When you click the "Download All" button, you will see several options for downloading the results:
                <ul>
                    <li><strong>Download All Binary Images (ZIP):</strong> Downloads all processed images in binary format as a single ZIP file.</li>
                    <li><strong>Download All Color Images (ZIP):</strong> Downloads all processed images in color format as a single ZIP file.</li>
                    <li><strong>Download All Pixel Data (CSV):</strong> Downloads the pixel data for all processed images in CSV format.</li>
                    <li><strong>Download All Types (ZIP):</strong> Downloads all available formats (binary images, color images, pixel data) combined in a single ZIP file.</li>
                </ul>
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>How to Use:</strong> Choose the desired download option from the dropdown menu. The application will prepare the selected results and download them to your computer.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>File Organization:</strong> The downloaded ZIP file will contain folders or files named according to the original images and the type of processed results. This helps in keeping the data organized and easy to access.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Use Cases:</strong> This feature is useful for backing up results, sharing processed data with others, or performing further analysis on the downloaded data.
            </Typography>
        </>
    );

    const viewLogsContent = (
        <>
            <Typography variant="body1" paragraph>
            The <strong>"View Logs"</strong> button allows you to access a detailed history of all actions performed within the application. This includes uploading, processing, deleting images, and other significant events. Here are some key points about this functionality:
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Access History:</strong> View a comprehensive log of all the actions taken, including timestamps and details of each operation.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Track Changes:</strong> Monitor changes made to the application, which can be useful for auditing and troubleshooting purposes.
            </Typography>
            <Typography variant="body1" paragraph>
                <strong>Error Tracking:</strong> Easily identify and review any errors or issues that occurred during image processing or other tasks.
            </Typography>
        </>
    );

    return (
        <Container>
        <Box sx={{ my: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h2" component="h1" gutterBottom>
                User Guide
            </Typography>
            <Button variant="contained" color="primary" onClick={handleBackToHome}>
                BACK TO HOME PAGE
            </Button>
        </Box>
        <Typography variant="h5" component="h2" gutterBottom sx={{ mt: 1, mb: 4 }}>
            Welcome to the AI-Driven Shoreline Detection application. This guide will help you understand how to use the application effectively.
        </Typography>
        <Grid container spacing={4}>
        <Grid item xs={12} sm={6} md={3}>
            <Card>
                <CardMedia>
                <Box sx={{ textAlign: 'center', p: 2 }}>
                    <VisibilityIcon sx={{ fontSize: 50, color: '#3f51b5' }} />
                </Box>
                </CardMedia>
                <CardContent>
                <Typography gutterBottom variant="h5" component="div">
                    View Switch
                </Typography>
                <Typography variant="body2" color="text.secondary">
                    The "View Switch" feature lets you toggle between views, offering flexibility in analyzing and interacting with images and their results.
                </Typography>
                </CardContent>
                <CardActions>
                    <Button size="small" onClick={() => handleClickOpen(viewSwitchContent)}>Learn More</Button>
                </CardActions>
            </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
            <Card>
                <CardMedia>
                <Box sx={{ textAlign: 'center', p: 2 }}>
                    <FilterIcon sx={{ fontSize: 50, color: '#3f51b5' }} />
                </Box>
                </CardMedia>
                <CardContent>
                <Typography gutterBottom variant="h5" component="div">
                    Quality Check
                </Typography>
                <Typography variant="body2" color="text.secondary">
                    The "Quality Check" feature helps you ensure that only high-quality images are processed, enhancing the accuracy and reliability of the shoreline detection results.
                </Typography>
                </CardContent>
                <CardActions>
                    <Button size="small" onClick={() => handleClickOpen(qualityCheckContent)}>Learn More</Button>
                </CardActions>
            </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
            <Card>
                <CardMedia>
                <Box sx={{ textAlign: 'center', p: 2 }}>
                    <UploadIcon sx={{ fontSize: 50, color: '#3f51b5' }} />
                </Box>
                </CardMedia>
                <CardContent>
                <Typography gutterBottom variant="h5" component="div">
                    Upload Image
                </Typography>
                <Typography variant="body2" color="text.secondary">
                Click on the "Upload Image" button to select and upload images from your computer. Only image files are accepted. You can upload one or multiple images at once.
                </Typography>
                </CardContent>
                <CardActions>
                    <Button size="small" onClick={() => handleClickOpen(uploadImageContent)}>Learn More</Button>
                </CardActions>
            </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
            <Card>
                <CardMedia>
                <Box sx={{ textAlign: 'center', p: 2 }}>
                    <CheckCircleRoundedIcon sx={{ fontSize: 50, color: '#3f51b5' }} />
                </Box>
                </CardMedia>
                <CardContent>
                <Typography gutterBottom variant="h5" component="div">
                    Get Result
                </Typography>
                <Typography variant="body2" color="text.secondary">
                    Click on the "Get Result" button to start the shoreline detection process. The application will process the uploaded images and display the results.
                </Typography>
                </CardContent>
                <CardActions>
                    <Button size="small" onClick={() => handleClickOpen(getResultContent)}>Learn More</Button>
                </CardActions>
            </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
            <Card>
                <CardMedia>
                <Box sx={{ textAlign: 'center', p: 2 }}>
                    <DeleteSweepRoundedIcon sx={{ fontSize: 50, color: '#3f51b5' }} />
                </Box>
                </CardMedia>
                <CardContent>
                <Typography gutterBottom variant="h5" component="div">
                    Clear Image
                </Typography>
                <Typography variant="body2" color="text.secondary">
                    Click on the "Clear Image" button to completely clear all the uploaded images, remove any processed results, and fully reset the application back to its initial default state.
                </Typography>
                </CardContent>
                <CardActions>
                    <Button size="small" onClick={() => handleClickOpen(clearImageContent)}>Learn More</Button>
                </CardActions>
            </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
            <Card>
                <CardMedia>
                <Box sx={{ textAlign: 'center', p: 2 }}>
                    <SwapHorizIcon sx={{ fontSize: 50, color: '#3f51b5' }} />
                </Box>
                </CardMedia>
                <CardContent>
                <Typography gutterBottom variant="h5" component="div">
                    Toggle All
                </Typography>
                <Typography variant="body2" color="text.secondary">
                    The "Toggle All" button allows you to switch display modes for all results at once. For example, you can toggle between binary and color modes for all detected shorelines simultaneously.
                </Typography>
                </CardContent>
                <CardActions>
                    <Button size="small" onClick={() => handleClickOpen(toggleAllContent)}>Learn More</Button>
                </CardActions>
            </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
            <Card>
                <CardMedia>
                <Box sx={{ textAlign: 'center', p: 2 }}>
                    <DownloadIcon sx={{ fontSize: 50, color: '#3f51b5' }} />
                </Box>
                </CardMedia>
                <CardContent>
                <Typography gutterBottom variant="h5" component="div">
                    Download All
                </Typography>
                <Typography variant="body2" color="text.secondary">
                    Click on the "Download All" button to download all the results in different formats. You can choose to download binary images, color images, pixel data, or all results combined in a zip file.
                </Typography>
                </CardContent>
                <CardActions>
                    <Button size="small" onClick={() => handleClickOpen(downloadAllContent)}>Learn More</Button>
                </CardActions>
            </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
            <Card>
                <CardMedia>
                <Box sx={{ textAlign: 'center', p: 2 }}>
                    <HistoryIcon sx={{ fontSize: 50, color: '#3f51b5' }} />
                </Box>
                </CardMedia>
                <CardContent>
                <Typography gutterBottom variant="h5" component="div">
                    View Logs
                </Typography>
                <Typography variant="body2" color="text.secondary">
                    The "View Logs" button allows you to access a detailed history of all actions performed within the application. This includes uploading, processing, deleting images, and other significant events.
                </Typography>
                </CardContent>
                <CardActions>
                <Button size="small" onClick={() => handleClickOpen(viewLogsContent)}>Learn More</Button>
                </CardActions>
            </Card>
            </Grid>
            </Grid>
            

            <Dialog open={open} onClose={handleClose} aria-labelledby="alert-dialog-title" aria-describedby="alert-dialog-description">
                <DialogTitle id="alert-dialog-title">Details</DialogTitle>
                    <DialogContent>
                        <DialogContentText id="alert-dialog-description" component="div">
                            {dialogContent}
                        </DialogContentText>
                    </DialogContent>
                    <DialogActions>
                        <Button onClick={handleClose} color="primary" autoFocus>
                            Close
                        </Button>
                    </DialogActions>
                </Dialog>
        </Container>
    );
}

export default UserGuide;
