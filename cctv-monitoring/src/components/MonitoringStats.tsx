import React, { useEffect, useState } from 'react';
import { Card, Row, Col, Badge } from 'react-bootstrap';
import { storageService } from '../services/storageService';

const MonitoringStats: React.FC = () => {
  const [startTime, setStartTime] = useState<Date | null>(null);
  const [stats, setStats] = useState({
    total: 0,
    confirmed: 0,
    falseAlarms: 0,
    unconfirmed: 0
  });

  useEffect(() => {
    // Initialize start time if not already set
    if (!startTime) {
      const savedStartTime = localStorage.getItem('monitoring_start_time');
      if (savedStartTime) {
        setStartTime(new Date(savedStartTime));
      } else {
        const now = new Date();
        localStorage.setItem('monitoring_start_time', now.toISOString());
        setStartTime(now);
      }
    }

    // Load notifications and update stats
    const loadData = () => {
      const allNotifications = storageService.getNotifications();
      
      // Calculate statistics
      const confirmed = allNotifications.filter(n => n.confirmed === true).length;
      const falseAlarms = allNotifications.filter(n => n.confirmed === false).length;
      const unconfirmed = allNotifications.filter(n => n.confirmed === null).length;
      
      setStats({
        total: allNotifications.length,
        confirmed,
        falseAlarms,
        unconfirmed
      });
    };

    loadData();
    const intervalId = setInterval(loadData, 5000);
    
    return () => clearInterval(intervalId);
  }, [startTime]);

  // Format duration since monitoring started
  const formatDuration = () => {
    if (!startTime) return "0h 0m";
    
    const now = new Date();
    const diffMs = now.getTime() - startTime.getTime();
    const diffHrs = Math.floor(diffMs / (1000 * 60 * 60));
    const diffMins = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
    
    return `${diffHrs}h ${diffMins}m`;
  };

  return (
    <Card className="mb-4">
      <Card.Header>Monitoring Statistics</Card.Header>
      <Card.Body>
        <Row>
          <Col xs={6} md={3} className="text-center mb-3">
            <div className="small text-muted">Monitoring Duration</div>
            <div className="h5">{formatDuration()}</div>
          </Col>
          <Col xs={6} md={3} className="text-center mb-3">
            <div className="small text-muted">Total Detections</div>
            <div className="h5">{stats.total}</div>
          </Col>
          <Col xs={6} md={3} className="text-center mb-3">
            <div className="small text-muted">Confirmed Shoplifting</div>
            <div className="h5">
              <Badge bg="danger">{stats.confirmed}</Badge>
            </div>
          </Col>
          <Col xs={6} md={3} className="text-center mb-3">
            <div className="small text-muted">False Alarms</div>
            <div className="h5">
              <Badge bg="success">{stats.falseAlarms}</Badge>
            </div>
          </Col>
        </Row>
        <Row>
          <Col className="text-center">
            <div className="small text-muted">Unconfirmed Incidents</div>
            <div className="h5">
              <Badge bg="warning">{stats.unconfirmed}</Badge>
            </div>
          </Col>
        </Row>
      </Card.Body>
    </Card>
  );
};

export default MonitoringStats;