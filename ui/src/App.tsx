import { Container, Row, Col, Navbar } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

import FolderMonitor from './components/FolderMonitor';
import NotificationList from './components/NotificationList';
import MonitoringStats from './components/MonitoringStats';
import ServerStatusIndicator from './components/ServerStatusIndicator';
import { FileMonitoringService } from './services/fileMonitoringService';

// Create a singleton instance of the file monitoring service
const fileMonitoringService = new FileMonitoringService();

function App() {
  return (
    <div className="App d-flex flex-column min-vh-100">
      <Navbar bg="dark" variant="dark" expand="lg" className="mb-4">
        <Container>
          <Navbar.Brand>AI-Powered Shoplifting Detection</Navbar.Brand>
          <div className="ms-auto">
            <ServerStatusIndicator />
          </div>
        </Container>
      </Navbar>

      <Container className="flex-grow-1">
        <Row>
          <Col>
            <FolderMonitor monitorService={fileMonitoringService} />
          </Col>
        </Row>
        <Row className="mt-4">
          <Col>
            <MonitoringStats />
          </Col>
        </Row>
        <Row className="mt-4 mb-5">
          <Col>
            <NotificationList />
          </Col>
        </Row>
      </Container>

      <footer className="bg-light py-3 mt-5 border-top">
        <Container>
          <Row>
            <Col className="text-center text-muted">
              <p className="mb-0">AI-Powered Shoplifting Detection</p>
            </Col>
          </Row>
        </Container>
      </footer>
    </div>
  );
}

export default App;
