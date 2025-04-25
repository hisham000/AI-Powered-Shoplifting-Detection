import React, { useState, useEffect, useMemo } from 'react';
import {
  Card,
  ListGroup,
  Badge,
  Button,
  Modal,
  Form,
  Row,
  Col,
} from 'react-bootstrap';
import { ShopliftingNotification } from '../models/types';
import { storageService } from '../services/storageService';
import { apiService } from '../services/apiService';
import { extractCameraInfo } from '../utils/filePathUtils';

type FilterStatus = 'all' | 'unconfirmed' | 'confirmed' | 'false-alarm';

const NotificationList: React.FC = () => {
  const [notifications, setNotifications] = useState<ShopliftingNotification[]>(
    []
  );
  const [selectedNotification, setSelectedNotification] =
    useState<ShopliftingNotification | null>(null);
  const [showModal, setShowModal] = useState(false);

  // Filter state
  const [filterStatus, setFilterStatus] = useState<FilterStatus>('all');
  const [filterCamera, setFilterCamera] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');

  // Load notifications from localStorage
  useEffect(() => {
    const loadNotifications = () => {
      const savedNotifications = storageService.getNotifications();
      setNotifications(savedNotifications);
    };

    // Load notifications initially
    loadNotifications();

    // Set up interval to check for new notifications
    const intervalId = setInterval(loadNotifications, 5000);

    return () => clearInterval(intervalId);
  }, []);

  // Get unique camera IDs for filter dropdown
  const uniqueCameras = useMemo(() => {
    const cameraSet = new Set<string>();
    notifications.forEach(notification => {
      const cameraId = extractCameraInfo(notification.videoPath);
      cameraSet.add(cameraId);
    });
    return Array.from(cameraSet).sort();
  }, [notifications]);

  // Apply filters to notifications
  const filteredAndSortedNotifications = useMemo(() => {
    // First filter by status
    let filtered = [...notifications];

    if (filterStatus === 'unconfirmed') {
      filtered = filtered.filter(n => n.confirmed === null);
    } else if (filterStatus === 'confirmed') {
      filtered = filtered.filter(n => n.confirmed === true);
    } else if (filterStatus === 'false-alarm') {
      filtered = filtered.filter(n => n.confirmed === false);
    }

    // Then filter by camera
    if (filterCamera !== 'all') {
      filtered = filtered.filter(
        n => extractCameraInfo(n.videoPath) === filterCamera
      );
    }

    // Then filter by search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        n =>
          n.videoPath.toLowerCase().includes(query) ||
          extractCameraInfo(n.videoPath).toLowerCase().includes(query)
      );
    }

    // Finally sort by date (latest first)
    return filtered.sort((a, b) => {
      const dateA = new Date(a.dateDetected);
      const dateB = new Date(b.dateDetected);
      return dateB.getTime() - dateA.getTime();
    });
  }, [notifications, filterStatus, filterCamera, searchQuery]);

  // Reset all filters
  const resetFilters = () => {
    setFilterStatus('all');
    setFilterCamera('all');
    setSearchQuery('');
  };

  const handleView = (notification: ShopliftingNotification) => {
    setSelectedNotification(notification);
    setShowModal(true);
  };

  const handleConfirm = async (isShoplifting: boolean) => {
    if (!selectedNotification) return;

    try {
      // Use the video_id if available, otherwise fall back to the path
      if (selectedNotification.video_id) {
        // Send confirmation back to the EEP with the video_id
        await apiService.confirmVideo(
          selectedNotification.videoPath,
          isShoplifting
        );
      } else {
        // Fall back to using the path for backwards compatibility
        await apiService.confirmVideo(
          selectedNotification.videoPath,
          isShoplifting
        );
      }

      // Update notification in local storage
      storageService.updateNotification(selectedNotification.id, isShoplifting);

      // Update UI
      setNotifications(prevNotifications =>
        prevNotifications.map(notification =>
          notification.id === selectedNotification.id
            ? { ...notification, confirmed: isShoplifting }
            : notification
        )
      );

      setShowModal(false);
    } catch (error) {
      console.error('Error confirming notification:', error);

      // Fallback: still update local UI even if the API call fails
      storageService.updateNotification(selectedNotification.id, isShoplifting);

      setNotifications(prevNotifications =>
        prevNotifications.map(notification =>
          notification.id === selectedNotification.id
            ? { ...notification, confirmed: isShoplifting }
            : notification
        )
      );

      setShowModal(false);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  return (
    <>
      <Card>
        <Card.Header className="d-flex justify-content-between align-items-center">
          <div>Shoplifting Notifications</div>
          <div className="text-muted small">
            {filteredAndSortedNotifications.length} of {notifications.length}{' '}
            {notifications.length === 1 ? 'notification' : 'notifications'}
          </div>
        </Card.Header>

        {/* Add filter controls */}
        <Card.Body className="pb-0 pt-3">
          <Row className="mb-3">
            <Col md={4}>
              <Form.Group>
                <Form.Label className="small">Status</Form.Label>
                <Form.Select
                  size="sm"
                  value={filterStatus}
                  onChange={e =>
                    setFilterStatus(e.target.value as FilterStatus)
                  }
                >
                  <option value="all">All</option>
                  <option value="unconfirmed">Unconfirmed</option>
                  <option value="confirmed">Confirmed Shoplifting</option>
                  <option value="false-alarm">False Alarm</option>
                </Form.Select>
              </Form.Group>
            </Col>

            <Col md={4}>
              <Form.Group>
                <Form.Label className="small">Camera</Form.Label>
                <Form.Select
                  size="sm"
                  value={filterCamera}
                  onChange={e => setFilterCamera(e.target.value)}
                >
                  <option value="all">All Cameras</option>
                  {uniqueCameras.map(camera => (
                    <option key={camera} value={camera}>
                      {camera}
                    </option>
                  ))}
                </Form.Select>
              </Form.Group>
            </Col>

            <Col md={4}>
              <Form.Group>
                <Form.Label className="small">Search</Form.Label>
                <div className="d-flex">
                  <Form.Control
                    size="sm"
                    type="text"
                    placeholder="Search..."
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                  />
                  {(filterStatus !== 'all' ||
                    filterCamera !== 'all' ||
                    searchQuery) && (
                    <Button
                      variant="outline-secondary"
                      size="sm"
                      className="ms-2"
                      onClick={resetFilters}
                    >
                      Clear
                    </Button>
                  )}
                </div>
              </Form.Group>
            </Col>
          </Row>
        </Card.Body>

        <Card.Body className="p-0 pt-2">
          {filteredAndSortedNotifications.length === 0 ? (
            <p className="text-muted p-3 text-center">
              {notifications.length > 0
                ? 'No notifications match your filters'
                : 'No shoplifting notifications'}
            </p>
          ) : (
            <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
              <ListGroup variant="flush">
                {filteredAndSortedNotifications.map(notification => (
                  <ListGroup.Item
                    key={notification.id}
                    // Add onClick handler and cursor style
                    onClick={() => handleView(notification)}
                    className="d-flex justify-content-between align-items-center border-start-0 border-end-0 notification-list-item cursor-pointer"
                  >
                    <div>
                      <div>
                        <strong>
                          {extractCameraInfo(notification.videoPath)}
                        </strong>
                      </div>
                      <div>
                        Detected: {formatDate(notification.dateDetected)}
                      </div>
                    </div>
                    <div className="d-flex align-items-center">
                      {notification.confirmed === null ? (
                        <Badge bg="warning" className="me-2">
                          Unconfirmed
                        </Badge>
                      ) : notification.confirmed ? (
                        <Badge bg="danger" className="me-2">
                          Confirmed Shoplifting
                        </Badge>
                      ) : (
                        <Badge bg="success" className="me-2">
                          False Alarm
                        </Badge>
                      )}
                      {/* Button is still useful for accessibility and clear visual cue */}
                      <Button
                        variant="outline-primary"
                        size="sm"
                        // Prevent click event from bubbling up to the ListGroup.Item
                        onClick={e => {
                          e.stopPropagation();
                          handleView(notification);
                        }}
                      >
                        {notification.confirmed === null ? 'Review' : 'View'}
                      </Button>
                    </div>
                  </ListGroup.Item>
                ))}
              </ListGroup>
            </div>
          )}
        </Card.Body>
      </Card>

      {/* Video Review Modal */}
      <Modal show={showModal} onHide={() => setShowModal(false)} size="lg">
        <Modal.Header closeButton>
          <Modal.Title>
            Review Potential Shoplifting Incident -{' '}
            {selectedNotification &&
              extractCameraInfo(selectedNotification.videoPath)}
          </Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {selectedNotification && (
            <>
              {/* Video Player - Using file server URL instead of direct path */}
              <div className="ratio ratio-16x9 mb-3">
                <video controls>
                  <source
                    src={apiService.getVideoUrl(selectedNotification.videoPath)}
                    type="video/mp4"
                  />
                  Your browser does not support the video tag.
                </video>
              </div>

              <p>
                <strong>Camera:</strong>{' '}
                {extractCameraInfo(selectedNotification.videoPath)}
              </p>
              <p>
                <strong>Detection Time:</strong>{' '}
                {formatDate(selectedNotification.dateDetected)}
              </p>
              <p>
                <strong>File:</strong> {selectedNotification.videoPath}
              </p>

              {selectedNotification.confirmed === null ? (
                <p className="text-warning">
                  Please confirm if this is actually shoplifting or a false
                  alarm.
                </p>
              ) : (
                <p
                  className={
                    selectedNotification.confirmed
                      ? 'text-danger'
                      : 'text-success'
                  }
                >
                  This was marked as{' '}
                  {selectedNotification.confirmed
                    ? 'shoplifting'
                    : 'not shoplifting'}
                  .
                </p>
              )}
            </>
          )}
        </Modal.Body>
        <Modal.Footer>
          {selectedNotification && selectedNotification.confirmed === null && (
            <>
              <Button variant="success" onClick={() => handleConfirm(false)}>
                Not Shoplifting
              </Button>
              <Button variant="danger" onClick={() => handleConfirm(true)}>
                Confirm Shoplifting
              </Button>
            </>
          )}
          <Button variant="secondary" onClick={() => setShowModal(false)}>
            Close
          </Button>
        </Modal.Footer>
      </Modal>
    </>
  );
};

export default NotificationList;
