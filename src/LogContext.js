import React, { createContext, useState, useContext } from 'react';

const LogContext = createContext();

export const useLog = () => useContext(LogContext);

export const LogProvider = ({ children }) => {
    const [logs, setLogs] = useState([]);
    const [openSnackbar, setOpenSnackbar] = useState(false);
    const [currentLog, setCurrentLog] = useState(null);

    const addLog = (message, severity = 'info') => {
        const newLog = { message, severity, timestamp: new Date().toLocaleTimeString() };
        setLogs(prevLogs => [...prevLogs, newLog]);
        setCurrentLog(newLog);
        setOpenSnackbar(true);
    };

    const clearLogs = () => {
        setLogs([]);
    };

    const handleCloseSnackbar = (event, reason) => {
        if (reason === 'clickaway') {
            return;
        }
        setOpenSnackbar(false);
    };

    const getAllLogs = () => logs;

    return (
        <LogContext.Provider value={{ logs, addLog, clearLogs, openSnackbar, currentLog, handleCloseSnackbar,getAllLogs }}>
            {children}
        </LogContext.Provider>
    );
};