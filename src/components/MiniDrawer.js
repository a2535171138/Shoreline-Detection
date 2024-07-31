import React, { useState } from 'react';
import { styled, useTheme } from '@mui/material/styles';
import Box from '@mui/material/Box';
import MuiDrawer from '@mui/material/Drawer';
import MuiAppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import List from '@mui/material/List';
import CssBaseline from '@mui/material/CssBaseline';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import IconButton from '@mui/material/IconButton';
import MenuIcon from '@mui/icons-material/Menu';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import FileUploadRoundedIcon from '@mui/icons-material/FileUploadRounded';
import CheckCircleRoundedIcon from '@mui/icons-material/CheckCircleRounded';
import DeleteSweepRoundedIcon from '@mui/icons-material/DeleteSweepRounded';
import SwapHorizIcon from '@mui/icons-material/SwapHoriz';
import DownloadRoundedIcon from '@mui/icons-material/DownloadRounded';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';
import FilterIcon from '@mui/icons-material/Filter';
import VisibilityIcon from '@mui/icons-material/Visibility';
import Switch from '@mui/material/Switch';
import HistoryIcon from '@mui/icons-material/History';
import AccountCircle from '@mui/icons-material/AccountCircle';
import Tooltip from '@mui/material/Tooltip';
import { useNavigate } from 'react-router-dom'; 
import { orange } from '@mui/material/colors';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';
import Button from '@mui/material/Button';
import LinearProgress from "@mui/material/LinearProgress";

const drawerWidth = 240;

const openedMixin = (theme) => ({
    width: drawerWidth,
    transition: theme.transitions.create('width', {
        easing: theme.transitions.easing.sharp,
        duration: theme.transitions.duration.enteringScreen,
    }),
    overflowX: 'hidden',
});

const closedMixin = (theme) => ({
    transition: theme.transitions.create('width', {
        easing: theme.transitions.easing.sharp,
        duration: theme.transitions.duration.leavingScreen,
    }),
    overflowX: 'hidden',
    width: `calc(${theme.spacing(7)} + 1px)`,
    [theme.breakpoints.up('sm')]: {
        width: `calc(${theme.spacing(8)} + 1px)`,
    },
});

const DrawerHeader = styled('div')(({ theme }) => ({
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-end',
    padding: theme.spacing(0, 1),
    ...theme.mixins.toolbar,
}));

const AppBar = styled(MuiAppBar, {
    shouldForwardProp: (prop) => prop !== 'open',
})(({ theme, open }) => ({
    zIndex: theme.zIndex.drawer + 1,
    transition: theme.transitions.create(['width', 'margin'], {
        easing: theme.transitions.easing.sharp,
        duration: theme.transitions.duration.leavingScreen,
    }),
    ...(open && {
        marginLeft: drawerWidth,
        width: `calc(100% - ${drawerWidth}px)`,
        transition: theme.transitions.create(['width', 'margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
        }),
    }),
}));

const Drawer = styled(MuiDrawer, { shouldForwardProp: (prop) => prop !== 'open' })(
    ({ theme, open }) => ({
        width: drawerWidth,
        flexShrink: 0,
        whiteSpace: 'nowrap',
        boxSizing: 'border-box',
        ...(open && {
            ...openedMixin(theme),
            '& .MuiDrawer-paper': openedMixin(theme),
        }),
        ...(!open && {
            ...closedMixin(theme),
            '& .MuiDrawer-paper': closedMixin(theme),
        }),
    }),
);

function MiniDrawer({ onFileUpload, onClearImages, onGetResult, onToggleAllDisplayModes, onDownloadAll, children,  onToggleQualityCheck,qualityCheckEnabled,hasResults,onSwitchView,
                        currentView,onViewLogs, isDownloading,
                        modelStatus}) {
    const theme = useTheme();
    const [open, setOpen] = useState(false);
    const fileInputRef = React.useRef(null);
    const [anchorElDownload, setAnchorElDownload] = useState(null);
    const [anchorElAccount, setAnchorElAccount] = useState(null);
    const [anchorElResult, setAnchorElResult] = useState(null);
    const [confirmDialogOpen, setConfirmDialogOpen] = useState(false);
    const navigate = useNavigate();

    const handleDrawerOpen = () => {
        setOpen(true);
    };

    const handleDrawerClose = () => {
        setOpen(false);
    };

    // const handleUploadClick = () => {
    //     fileInputRef.current.click();
    // };

    const handleFileChange = (event) => {
        const files = Array.from(event.target.files);
        onFileUpload(files);
        event.target.value = null;
    };

    const handleDownloadAllClick = (event) => {
        setAnchorElDownload(event.currentTarget);
    };

    const handleDownloadAllClose = () => {
        setAnchorElDownload(null);
    };

    const handleAccountClick = (event) => {
        setAnchorElAccount(event.currentTarget);
    };

    const handleAccountClose = () => {
        setAnchorElAccount(null);
    };

    const handleUserGuideClick = () => {
        handleAccountClose();
        navigate('/user-guide');
    };

    const handleClearImagesClick = () => {
        setConfirmDialogOpen(true);
    };

    const handleConfirmClearImages = () => {
        onClearImages();
        setConfirmDialogOpen(false);
    };

    const handleCancelClearImages = () => {
        setConfirmDialogOpen(false);
    };

    const handleResultClick = (event) => { 
        setAnchorElResult(event.currentTarget);
    };

    const handleResultClose = () => {
        setAnchorElResult(null);
    };

    const OrangeSwitch = styled(Switch)(({ theme }) => ({
        '& .MuiSwitch-switchBase.Mui-checked': {
          color: orange[500],
          '&:hover': {
            backgroundColor: theme.palette.action.hoverOpacity,
          },
        },
        '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
          backgroundColor: orange[500],
        },
      }));

    return (
        <Box sx={{ display: 'flex' }}>
            <CssBaseline />
            <AppBar position="fixed" open={open}>
                <Toolbar>
                    <IconButton
                        color="inherit"
                        aria-label="open drawer"
                        onClick={handleDrawerOpen}
                        edge="start"
                        sx={{
                            marginRight: 5,
                            ...(open && { display: 'none' }),
                        }}
                    >
                        <MenuIcon />
                    </IconButton>
                    <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
                        AI-Driven Shoreline Detection
                    </Typography>
                    <Tooltip title="Switch between Uploaded Images and Results view" aria-label="view switch">
                        <Box sx={{ display: 'flex', alignItems: 'center', marginRight: 2 }}>
                            <VisibilityIcon sx={{ marginRight: 1 }} />
                            <Typography variant="body1" sx={{ marginRight: 1 }}>
                                View Switch
                            </Typography>
                            <Switch
                                edge="end"
                                onChange={onSwitchView}
                                checked={currentView === 'results'}
                                color="success"
                                size="small"
                            />
                        </Box>
                    </Tooltip>
                    <Tooltip title="Enabling this will increase processing time due to additional quality checks" aria-label="quality check">
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <FilterIcon color="inherit" />
                            <Typography variant="body1" sx={{ marginRight: 1 }}>
                                Quality Check
                            </Typography>
                            <OrangeSwitch
                                edge="end"
                                onChange={onToggleQualityCheck}
                                checked={qualityCheckEnabled}
                                size="small"
                            />
                        </Box>
                    </Tooltip>
                    <IconButton
                            size="large"
                            aria-label="user guide"
                            aria-controls="menu-account"
                            aria-haspopup="true"
                            onClick={handleAccountClick}
                            color="inherit"
                        >
                            <AccountCircle />
                    </IconButton>
                    <Menu
                        id="menu-appbar"
                        anchorEl={anchorElAccount}
                        anchorOrigin={{
                            vertical: 'top',
                            horizontal: 'right',
                        }}
                        keepMounted
                        transformOrigin={{
                            vertical: 'top',
                            horizontal: 'right',
                        }}
                        open={Boolean(anchorElAccount)}
                        onClose={handleAccountClose}
                    >
                        <MenuItem onClick={handleUserGuideClick}>User Guide</MenuItem>
                    </Menu>
                </Toolbar>
            </AppBar>
            <Drawer variant="permanent" open={open}>
                <DrawerHeader>
                    <IconButton onClick={handleDrawerClose}>
                        {theme.direction === 'rtl' ? <ChevronRightIcon /> : <ChevronLeftIcon />}
                    </IconButton>
                </DrawerHeader>
                <Divider />
                <List>
                    <ListItem disablePadding sx={{ display: 'block' }}>
                        <Tooltip title="Upload Image" disableHoverListener={open}>
                            <ListItemButton
                                onClick={onFileUpload}
                                sx={{
                                    minHeight: 48,
                                    justifyContent: open ? 'initial' : 'center',
                                    px: 2.5,
                                }}
                            >
                                <ListItemIcon
                                    sx={{
                                        minWidth: 0,
                                        mr: open ? 3 : 'auto',
                                        justifyContent: 'center',
                                    }}
                                >
                                    <FileUploadRoundedIcon />
                                </ListItemIcon>
                                <ListItemText primary="Upload Image" sx={{ opacity: open ? 1 : 0 }} />
                            </ListItemButton>
                        </Tooltip>
                    </ListItem>
                    <ListItem disablePadding sx={{ display: 'block' }}>
                        <Tooltip title="Get Result" disableHoverListener={open}>
                            <ListItemButton
                                onClick={handleResultClick}
                                // onClick={onGetResult}
                                sx={{
                                    minHeight: 48,
                                    justifyContent: open ? 'initial' : 'center',
                                    px: 2.5,
                                }}
                            >
                                <ListItemIcon
                                    sx={{
                                        minWidth: 0,
                                        mr: open ? 3 : 'auto',
                                        justifyContent: 'center',
                                    }}
                                >
                                    <CheckCircleRoundedIcon />
                                </ListItemIcon>
                                <ListItemText primary="Get Result" sx={{ opacity: open ? 1 : 0 }} />
                            </ListItemButton>
                        </Tooltip>
                    </ListItem>
                    <ListItem disablePadding sx={{ display: 'block' }}>
                        <Tooltip title="Clear Image" disableHoverListener={open}>
                            <ListItemButton
                                onClick={handleClearImagesClick}
                                // onClick={onClearImages}
                                sx={{
                                    minHeight: 48,
                                    justifyContent: open ? 'initial' : 'center',
                                    px: 2.5,
                                }}
                            >
                                <ListItemIcon
                                    sx={{
                                        minWidth: 0,
                                        mr: open ? 3 : 'auto',
                                        justifyContent: 'center',
                                    }}
                                >
                                    <DeleteSweepRoundedIcon />
                                </ListItemIcon>
                                <ListItemText primary="Clear Image" sx={{ opacity: open ? 1 : 0 }} />
                            </ListItemButton>
                        </Tooltip>
                    </ListItem>
                    <ListItem disablePadding sx={{ display: 'block' }}>
                        <Tooltip title="Toggle All Result Images (Between binary and color)">
                            <ListItemButton
                                onClick={onToggleAllDisplayModes}
                                sx={{
                                    minHeight: 48,
                                    justifyContent: open ? 'initial' : 'center',
                                    px: 2.5,
                                }}
                            >
                                <ListItemIcon
                                    sx={{
                                        minWidth: 0,
                                        mr: open ? 3 : 'auto',
                                        justifyContent: 'center',
                                    }}
                                >
                                    <SwapHorizIcon />
                                </ListItemIcon>
                                <ListItemText primary="Toggle All" sx={{ opacity: open ? 1 : 0 }} />
                            </ListItemButton>
                        </Tooltip>
                    </ListItem>
                    <ListItem disablePadding sx={{ display: 'block' }}>
                        <Tooltip title="Download All Results In Different Formats. (Binary images, Color images or Pixel data)" >
                            <ListItemButton
                                onClick={handleDownloadAllClick}
                                sx={{
                                    minHeight: 48,
                                    justifyContent: open ? 'initial' : 'center',
                                    px: 2.5,
                                }}
                            >
                                <ListItemIcon
                                    sx={{
                                        minWidth: 0,
                                        mr: open ? 3 : 'auto',
                                        justifyContent: 'center',
                                    }}
                                >
                                    <DownloadRoundedIcon />
                                </ListItemIcon>
                                <ListItemText primary="Download All" sx={{ opacity: open ? 1 : 0 }} />
                            </ListItemButton>
                        </Tooltip>
                    </ListItem>
                    <ListItem disablePadding sx={{ display: 'block' }}>
                        <Tooltip title="View the Action Logs and Details Of All Operations.">
                            <ListItemButton
                                onClick={onViewLogs}
                                sx={{
                                    minHeight: 48,
                                    justifyContent: open ? 'initial' : 'center',
                                    px: 2.5,
                                }}
                            >
                                <ListItemIcon
                                    sx={{
                                        minWidth: 0,
                                        mr: open ? 3 : 'auto',
                                        justifyContent: 'center',
                                    }}
                                >
                                    <HistoryIcon />
                                </ListItemIcon>
                                <ListItemText primary="View Logs" sx={{ opacity: open ? 1 : 0 }} />
                            </ListItemButton>
                        </Tooltip>
                    </ListItem>

                </List>
                <Divider />
            </Drawer>
            <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
                <DrawerHeader />
                {children}
            </Box>
            <input
                type="file"
                multiple
                ref={fileInputRef}
                onChange={handleFileChange}
                style={{ display: 'none' }}
                accept="image/*"
            />
            <Menu
                anchorEl={anchorElDownload}
                open={Boolean(anchorElDownload)}
                onClose={handleDownloadAllClose}
            >
                <MenuItem onClick={() => { onDownloadAll('binary'); handleDownloadAllClose(); }}>Download All Binary Images (ZIP)</MenuItem>
                <MenuItem onClick={() => { onDownloadAll('color'); handleDownloadAllClose(); }}>Download All Color Images (ZIP)</MenuItem>
                <MenuItem onClick={() => { onDownloadAll('pixels'); handleDownloadAllClose(); }}>Download All Pixel Data (CSV)</MenuItem>
                <MenuItem onClick={() => { onDownloadAll('all'); handleDownloadAllClose(); }}>Download All Types (ZIP)</MenuItem>
            </Menu>
            <Menu
                anchorEl={anchorElResult}
                open={Boolean(anchorElResult)}
                onClose={handleResultClose}
            >
                <MenuItem onClick={() => { onGetResult('General'); handleResultClose(); }}>Choose General Scene</MenuItem>
                <MenuItem onClick={() => { onGetResult('Narrabeen'); handleResultClose(); }}>Choose Narrabeen Scene</MenuItem>
                <MenuItem onClick={() => { onGetResult('Gold Coast'); handleResultClose(); }}>Choose Gold Coast Scene</MenuItem>
                <MenuItem onClick={() => { onGetResult('CoastSnap'); handleResultClose(); }}>Choose CoastSnap Scene</MenuItem>
            </Menu>
            <Dialog
                open={isDownloading}
                aria-labelledby="alert-dialog-title"
                sx={{
                    '& .MuiDialog-paper': {
                        marginTop: '64px',
                        width: '100%',
                        maxWidth: 'none',
                        margin: 0,
                        height: 'calc(100% - 64px)',
                        borderRadius: 0,
                    },
                }}
            >
                <DialogTitle id="alert-dialog-title">{"Downloading Models"}</DialogTitle>
                <DialogContent>
                    <DialogContentText>
                        Some models are missing and are being downloaded. Please wait...
                    </DialogContentText>
                    {modelStatus && Object.entries(modelStatus).map(([model, status]) => (
                        <div key={model}>
                            <Typography variant="body2">{model}: {status}</Typography>
                            {status === "Downloading" && <LinearProgress />}
                        </div>
                    ))}
                </DialogContent>
            </Dialog>
            <Dialog
                open={confirmDialogOpen}
                onClose={handleCancelClearImages}
                aria-labelledby="alert-dialog-title"
                aria-describedby="alert-dialog-description"
            >
                <DialogTitle id="alert-dialog-title">
                    <Typography variant="h6" component="span" style={{ color: 'red'}}>
                        Confirm Clear Images
                    </Typography>
                </DialogTitle>
                <DialogContent>
                    <DialogContentText id="alert-dialog-description" >
                        <Typography variant="body1" component="span" >
                            Are you sure you want to clear all uploaded images and processed results? This action cannot be undone.
                        </Typography>
                    </DialogContentText>
                </DialogContent>
                <DialogActions>
                    <Button 
                        onClick={handleConfirmClearImages} 
                        color="warning" 
                        autoFocus
                        style={{ color: 'red' }}
                    >
                        Yes
                    </Button>
                    <Button 
                        onClick={handleCancelClearImages} 
                        color="warning"
                        style={{ color: 'red' }}
                    >
                        Cancel
                    </Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
}

export default MiniDrawer;