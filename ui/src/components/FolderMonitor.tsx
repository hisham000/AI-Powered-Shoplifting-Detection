import React, { useState, useEffect } from 'react';
import { Form, Button, Card, Alert, Row, Col, Badge } from 'react-bootstrap';
import { FileMonitoringService } from '../services/fileMonitoringService';
import { storageService } from '../services/storageService';

interface FolderMonitorProps {
  monitorService: FileMonitoringService;
}

const FolderMonitor: React.FC<FolderMonitorProps> = ({ monitorService }) => {
  const [folderPath, setFolderPath] = useState<string>('');
  const [isWatching, setIsWatching] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [processedFiles, setProcessedFiles] = useState<number>(0);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  useEffect(() => {
    // Load saved folder path from local storage if available
    const savedPath = storageService.getFolderConfig();
    if (savedPath) {
      setFolderPath(savedPath);
    }

    // Check service status on interval
    const intervalId = setInterval(() => {
      const status = monitorService.getStatus();
      setIsWatching(status.isWatching);
      setProcessedFiles(status.processedFilesCount);
      setLastChecked(status.lastCheckedTime);
    }, 1000);

    return () => clearInterval(intervalId);
  }, [monitorService]);

  const handleStartMonitoring = () => {
    if (!folderPath) {
      setError('Please enter a valid folder path');
      return;
    }

    try {
      monitorService.setup(folderPath).startWatching();
      setIsWatching(true);
      setError(null);
      setSuccess('Monitoring started successfully');
      
      // Clear success message after 3 seconds
      setTimeout(() => {
        setSuccess(null);
      }, 3000);
    } catch (err) {
      setError('Failed to start monitoring');
      console.error('Error starting monitoring:', err);
    }
  };

  const handleStopMonitoring = () => {
    monitorService.stopWatching();
    setIsWatching(false);
    setSuccess('Monitoring stopped successfully');
    
    // Clear success message after 3 seconds
    setTimeout(() => {
      setSuccess(null);
    }, 3000);
  };

  const handleResetHistory = () => {
    if (window.confirm('Are you sure you want to clear the processed files history and notifications? This will remove all existing notifications and cause already processed videos to be processed again.')) {
      // Clear processed files history
      monitorService.clearProcessedFiles();
      
      // Clear notifications
      storageService.clearNotifications();
      
      // Update the UI immediately with the reset count
      setProcessedFiles(0);
      
      setSuccess('Processed files history and notifications have been cleared');
      setTimeout(() => {
        setSuccess(null);
      }, 3000);
    }
  };

  const formatLastChecked = () => {
    if (!lastChecked) return 'Never';
    return lastChecked.toLocaleTimeString();
  };

  return (
    <Card className="mb-4">
      <Card.Header>CCTV Folder Monitoring</Card.Header>
      <Card.Body>
        {error && <Alert variant="danger">{error}</Alert>}
        {success && <Alert variant="success">{success}</Alert>}
        
        <Form>
          <Form.Group className="mb-3">
            <Form.Label>CCTV Footage Folder Path</Form.Label>
            <Form.Control
              type="text"
              placeholder="Enter the path to your CCTV footage folder (e.g., /path/to/CCTV)"
              value={folderPath}
              onChange={(e) => setFolderPath(e.target.value)}
              disabled={isWatching}
            />
            <Form.Text className="text-muted">
              Expected format: CCTV/cam#/YYYY-MM-DD/HH/YYYY-MM-DD_HH-MM-SS_cam#.mp4
            </Form.Text>
          </Form.Group>

          <Row className="align-items-center mb-3">
            <Col>
              <div className="d-flex gap-2">
                {!isWatching ? (
                  <Button 
                    variant="primary" 
                    onClick={handleStartMonitoring}
                    disabled={!folderPath}
                  >
                    Start Monitoring
                  </Button>
                ) : (
                  <Button 
                    variant="danger" 
                    onClick={handleStopMonitoring}
                  >
                    Stop Monitoring
                  </Button>
                )}
                
                <Button 
                  variant="outline-secondary"
                  onClick={handleResetHistory}
                >
                  Reset History
                </Button>
              </div>
            </Col>
            <Col className="text-end">
              <div className="text-muted mb-1">Status: {isWatching ? 
                <Badge bg="success">Watching</Badge> : 
                <Badge bg="secondary">Not Watching</Badge>}
              </div>
              <div className="text-muted small">
                Processed Files: {processedFiles} | Last Checked: {formatLastChecked()}
              </div>
            </Col>
          </Row>
        </Form>
      </Card.Body>
    </Card>
  );
};

export default FolderMonitor;