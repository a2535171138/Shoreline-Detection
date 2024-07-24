import React from 'react';
import { Snackbar, Alert } from '@mui/material';
import { useLog } from '../LogContext';

const SnackbarLog = () => {
    const { openSnackbar, currentLog, handleCloseSnackbar } = useLog();

    if (!currentLog) return null;

    return (
        <Snackbar
            anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
            open={openSnackbar}
            autoHideDuration={6000}
            onClose={handleCloseSnackbar}
        >
            <Alert onClose={handleCloseSnackbar} severity={currentLog.severity} sx={{ width: '100%' }}>
                {currentLog.message}
            </Alert>
        </Snackbar>
    );
};

export default SnackbarLog;