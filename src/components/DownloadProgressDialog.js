// DownloadProgressDialog.js
import React from 'react';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import LinearProgress from '@mui/material/LinearProgress';
import Typography from '@mui/material/Typography';

function DownloadProgressDialog({ open, modelStatus, downloadProgress }) {
    return (
        <Dialog open={open}>
            <DialogTitle>Downloading Models</DialogTitle>
            <DialogContent>
                {Object.entries(modelStatus).map(([model, status]) => (
                    <div key={model}>
                        <Typography variant="body1">{model}: {status}</Typography>
                        {status === "Downloading" && (
                            <LinearProgress variant="determinate" value={downloadProgress[model] || 0} />
                        )}
                    </div>
                ))}
            </DialogContent>
        </Dialog>
    );
}

export default DownloadProgressDialog;