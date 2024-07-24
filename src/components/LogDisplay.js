import React from 'react';
import { useLog } from '../LogContext';
import { Paper, Typography, List, ListItem, ListItemText } from '@mui/material';

const LogDisplay = () => {
    const { logs } = useLog();

    return (
        <Paper style={{ maxHeight: 200, overflow: 'auto', padding: '10px', marginTop: '20px' }}>
            <Typography variant="h6">Activity Log</Typography>
            <List dense>
                {logs.map((log, index) => (
                    <ListItem key={index}>
                        <ListItemText
                            primary={log.message}
                            secondary={log.timestamp}
                        />
                    </ListItem>
                ))}
            </List>
        </Paper>
    );
};

export default LogDisplay;