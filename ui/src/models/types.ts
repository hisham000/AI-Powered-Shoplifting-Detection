// Video data interface
export interface VideoData {
  path: string;
  timestamp: string;
  cameraId: string;
  processed: boolean;
}

// Shoplifting notification interface
export interface ShopliftingNotification {
  id: string;
  videoPath: string;
  timestamp: string;
  cameraId: string;
  confirmed: boolean | null;
  dateDetected: string;
  video_id?: string; // ID used by the EEP for confirming videos
}

// API response interface for process-video endpoint
export interface ProcessVideoResponse {
  shoplifting_detected: boolean;
  video_path: string;
  timestamp: string;
  camera_id: string;
  video_id?: string;
  simulation?: boolean; // Flag to indicate if this is a simulated response
}

// Folder watching service configuration
export interface FolderWatchConfig {
  folderPath: string;
  isWatching: boolean;
}