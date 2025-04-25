import React, { useEffect, useState } from 'react';
import { Badge, OverlayTrigger, Tooltip, Stack } from 'react-bootstrap';
import { isEEPOnline } from '../utils/fileSystemUtils';

const ServerStatusIndicator: React.FC = () => {
  const [isEEPAvailable, setIsEEPAvailable] = useState<boolean | null>(null);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  useEffect(() => {
    // Check server and EEP status immediately
    checkEEPStatus();

    // Set up regular checks every 30 seconds
    const eepIntervalId = setInterval(checkEEPStatus, 30000);

    return () => {
      clearInterval(eepIntervalId);
    };
  }, []);

  const checkEEPStatus = async () => {
    try {
      const isAvailable = await isEEPOnline();
      setIsEEPAvailable(isAvailable);
      setLastChecked(new Date());
    } catch (error) {
      console.error('Error checking EEP status:', error);
      setIsEEPAvailable(false);
      setLastChecked(new Date());
    }
  };

  // Format for the last checked time
  const formatLastChecked = () => {
    if (!lastChecked) return 'never checked';
    return `last checked ${lastChecked.toLocaleTimeString()}`;
  };

  return (
    <Stack direction="horizontal" gap={2}>
      <OverlayTrigger
        placement="bottom"
        overlay={
          <Tooltip id="eep-status-tooltip">
            {isEEPAvailable
              ? 'EEP service is online and processing videos'
              : 'EEP service is offline - video processing paused'}
            <br />
            {formatLastChecked()}
          </Tooltip>
        }
      >
        <Badge
          bg={isEEPAvailable ? 'success' : 'danger'}
          className="cursor-pointer"
        >
          EEP: {isEEPAvailable ? 'Online' : 'Offline'}
        </Badge>
      </OverlayTrigger>
    </Stack>
  );
};

export default ServerStatusIndicator;
