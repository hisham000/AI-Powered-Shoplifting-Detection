import React, { useEffect, useState } from 'react';
import { Badge, OverlayTrigger, Tooltip } from 'react-bootstrap';
import { apiService } from '../services/apiService';

const ServerStatusIndicator: React.FC = () => {
  const [isServerAvailable, setIsServerAvailable] = useState<boolean | null>(null);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  useEffect(() => {
    // Check server status immediately
    checkServerStatus();

    // Set up a regular check every 30 seconds
    const intervalId = setInterval(checkServerStatus, 30000);
    
    return () => clearInterval(intervalId);
  }, []);

  const checkServerStatus = async () => {
    try {
      const isAvailable = await apiService.checkServerStatus();
      setIsServerAvailable(isAvailable);
      setLastChecked(new Date());
    } catch (error) {
      console.error('Error checking server status:', error);
      setIsServerAvailable(false);
      setLastChecked(new Date());
    }
  };

  // Format for the last checked time
  const formatLastChecked = () => {
    if (!lastChecked) return 'never checked';
    return `last checked ${lastChecked.toLocaleTimeString()}`;
  };

  // If we haven't checked yet, show a loading state
  if (isServerAvailable === null) {
    return <Badge bg="secondary">Checking EEP server...</Badge>;
  }

  return (
    <OverlayTrigger
      placement="bottom"
      overlay={
        <Tooltip id="server-status-tooltip">
          {isServerAvailable 
            ? 'EEP server is online and processing videos' 
            : 'EEP server is offline - using simulation mode'}
          <br />
          {formatLastChecked()}
        </Tooltip>
      }
    >
      <Badge bg={isServerAvailable ? 'success' : 'warning'} className="cursor-pointer">
        {isServerAvailable ? 'EEP Online' : 'EEP Offline (Simulation Mode)'}
      </Badge>
    </OverlayTrigger>
  );
};

export default ServerStatusIndicator;