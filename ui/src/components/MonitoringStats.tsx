import React, { useEffect, useState } from 'react';
import { Card, Row, Col, Badge } from 'react-bootstrap';
import { storageService } from '../services/storageService';

const MonitoringStats: React.FC = () => {
  const [stats, setStats] = useState({
    total: 0,
    confirmed: 0,
    falseAlarms: 0,
    unconfirmed: 0,
  });

  useEffect(() => {
    // Load notifications and update stats
    const loadData = () => {
      const allNotifications = storageService.getNotifications();

      // Calculate statistics
      const confirmed = allNotifications.filter(
        n => n.confirmed === true
      ).length;
      const falseAlarms = allNotifications.filter(
        n => n.confirmed === false
      ).length;
      const unconfirmed = allNotifications.filter(
        n => n.confirmed === null
      ).length;

      setStats({
        total: allNotifications.length,
        confirmed,
        falseAlarms,
        unconfirmed,
      });
    };

    loadData();
    const intervalId = setInterval(loadData, 5000);

    return () => clearInterval(intervalId);
  }, []);

  return (
    <Card className="mb-4">
      <Card.Header>Monitoring Statistics</Card.Header>
      <Card.Body>
        <Row>
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
          <Col xs={6} md={3} className="text-center">
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
