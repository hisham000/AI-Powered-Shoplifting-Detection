import React, { useState, useEffect } from 'react';
import { Button, Card, Alert, Row, Col, Badge } from 'react-bootstrap';
import { FileMonitoringService } from '../services/fileMonitoringService';
import { storageService } from '../services/storageService';

// Fixed CCTV directory path
const CCTV_DIR = "/CCTV";

interface FolderMonitorProps {
  monitorService: FileMonitoringService;
}

const FolderMonitor: React.FC<FolderMonitorProps> = ({ monitorService }) => {
  const [isWatching, setIsWatching] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [processedFiles, setProcessedFiles] = useState<number>(0);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  useEffect(() => {
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
    try {
      monitorService.startWatching();
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
        
        <div className="mb-3">
          <h6>Monitoring Directory</h6>
          <p className="text-primary fw-bold">{CCTV_DIR}</p>
          <p className="text-muted small">
            The system is configured to automatically monitor the CCTV directory for new video files.
          </p>
        </div>

        <Row className="align-items-center mb-3">
          <Col>
            <div className="d-flex gap-2">
              {!isWatching ? (
                <Button 
                  variant="primary" 
                  onClick={handleStartMonitoring}
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
      </Card.Body>
    </Card>
  );
};

export default FolderMonitor;