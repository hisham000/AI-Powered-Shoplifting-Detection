# CCTV Shoplifting Detection Monitor

A React application that monitors a directory containing CCTV footage, processes videos to detect shoplifting incidents, and provides an interface for reviewing and confirming detected incidents.

## Features

- Monitor a folder containing CCTV camera footage
- Automatically process new videos with the EEP (External Event Processor)
- Receive and store notifications for detected shoplifting incidents
- Review video footage of potential shoplifting
- Confirm or reject shoplifting incidents
- Send feedback to the EEP to improve detection accuracy

## Folder Structure

The application expects CCTV footage in the following format:
```
CCTV/cam{i}/YYYY-MM-DD/HH/YYYY-MM-DD_HH-MM-SS_cam{i}.mp4
```
Where:
- `{i}` is the camera ID number
- `YYYY-MM-DD` is the date
- `HH` is the hour
- `MM` is the minute
- `SS` is the second

## Getting Started

1. Install dependencies:
   ```
   npm install
   ```

2. Start the development server:
   ```
   npm start
   ```

3. Enter the path to your CCTV footage folder in the application and click "Start Monitoring"

## API Endpoints

The application communicates with an EEP (External Event Processor) running on `127.0.0.1:8000` with the following endpoints:

- `POST /eep/process-video`: Processes a video to detect shoplifting
- `POST /eep/confirm-video`: Confirms whether a detected incident was actually shoplifting

## Technologies

- React
- TypeScript
- Bootstrap for UI components
- localStorage for client-side storage
- Axios for API communication
