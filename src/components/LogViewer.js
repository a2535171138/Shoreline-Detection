// src/components/LogViewer.js
import React from 'react';
import { useLog } from '../LogContext';
import { Dialog, DialogTitle, DialogContent, List, ListItem, ListItemText, IconButton } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

const LogViewer = ({ open, onClose }) => {
    const { getAllLogs } = useLog();
    const logs = getAllLogs();

    return (
        <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
            <DialogTitle>
                Operation Logs
                <IconButton
                    aria-label="close"
                    onClick={onClose}
                    sx={{
                        position: 'absolute',
                        right: 8,
                        top: 8,
                    }}
                >
                    <CloseIcon />
                </IconButton>
            </DialogTitle>
            <DialogContent dividers>
                <List>
                    {logs.map((log, index) => (
                        <ListItem key={index}>
                            <ListItemText
                                primary={log.message}
                                secondary={`${log.timestamp} - Severity: ${log.severity}`}
                            />
                        </ListItem>
                    ))}
                </List>
            </DialogContent>
        </Dialog>
    );
};

export default LogViewer;