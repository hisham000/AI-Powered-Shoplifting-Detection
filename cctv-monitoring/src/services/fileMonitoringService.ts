import { apiService } from './apiService';
import { storageService } from './storageService';
import { ShopliftingNotification } from '../models/types';
import { v4 as uuidv4 } from 'uuid';
import {
  readVideoFilesFromDirectory,
  getVideoFile,
  getDirectoryChanges,
} from '../utils/fileSystemUtils';

// Service for monitoring folder for new CCTV footage
export class FileMonitoringService {
  private isWatching: boolean = false;
  private folderPath: string = '';
  private intervalId: number | null = null;
  private processedFiles: Set<string> = new Set();
  private lastCheckedTime: Date = new Date();
  private currentVideos: string[] = [];
  private previousVideos: string[] = [];

  // Server status checks
  private serverAvailable = true;
  private lastServerCheck: number = 0;
  private checkInterval: number = 5000; // Check interval in ms (default: 5 seconds)

  constructor() {
    // Load previously processed files from localStorage
    this.loadProcessedFiles();
  }

  // Load processed files from localStorage
  private loadProcessedFiles(): void {
    const storedFiles = localStorage.getItem('processed_files');
    if (storedFiles) {
      this.processedFiles = new Set(JSON.parse(storedFiles));
    }
  }

  // Save processed files to localStorage
  private saveProcessedFiles(): void {
    localStorage.setItem(
      'processed_files',
      JSON.stringify([...this.processedFiles])
    );
  }

  // Set up the folder path to monitor
  setup(folderPath: string) {
    this.folderPath = folderPath;
    
    storageService.saveFolderConfig(folderPath);
    console.log(`Set up monitoring for folder: ${folderPath} with interval: ${this.checkInterval}ms`);
    return this;
  }

  // Start watching for new files
  startWatching() {
    if (this.isWatching || !this.folderPath) return this;

    this.isWatching = true;
    this.lastCheckedTime = new Date();

    // Set monitoring start time in localStorage if not already set
    if (!localStorage.getItem('monitoring_start_time')) {
      localStorage.setItem('monitoring_start_time', new Date().toISOString());
    }

    // Check server status before starting
    this.checkServerStatus().then(available => {
      if (!available) {
        console.warn(
          'Server is not available. Some functionality may be limited.'
        );
      }
    });

    // Set up interval for checking new files
    this.intervalId = window.setInterval(
      () => this.checkForNewFiles(),
      this.checkInterval
    );

    console.log(
      `Started monitoring folder: ${this.folderPath} with interval: ${this.checkInterval}ms`
    );
    return this;
  }

  // Stop watching for new files
  stopWatching() {
    if (!this.isWatching) return this;

    if (this.intervalId !== null) {
      window.clearInterval(this.intervalId);
      this.intervalId = null;
    }

    this.isWatching = false;
    console.log(`Stopped monitoring folder: ${this.folderPath}`);
    return this;
  }

  // Check if a file is already processed
  isProcessed(filePath: string): boolean {
    return this.processedFiles.has(filePath);
  }

  // Mark a file as processed
  markAsProcessed(filePath: string): void {
    this.processedFiles.add(filePath);
    this.saveProcessedFiles();
  }

  // Clear processed files history
  clearProcessedFiles(): void {
    this.processedFiles.clear();
    localStorage.removeItem('processed_files');
    console.log('Cleared processed files history');
  }

  // Get status of the monitoring service
  getStatus() {
    return {
      isWatching: this.isWatching,
      folderPath: this.folderPath,
      processedFilesCount: this.processedFiles.size,
      lastCheckedTime: this.lastCheckedTime,
      checkInterval: this.checkInterval,
      serverAvailable: this.serverAvailable,
    };
  }

  // Check server availability periodically
  private async checkServerStatus() {
    // Only check the server once every 30 seconds
    const now = Date.now();
    if (now - this.lastServerCheck < 30000) {
      return this.serverAvailable;
    }

    this.lastServerCheck = now;

    try {
      this.serverAvailable = await apiService.checkServerStatus();
    } catch (error) {
      console.error('Error checking server status:', error);
      this.serverAvailable = false;
    }

    return this.serverAvailable;
  }

  // Get server status
  getServerStatus(): { available: boolean; lastCheck: number } {
    return {
      available: this.serverAvailable,
      lastCheck: this.lastServerCheck,
    };
  }

  // Check for new files in the monitored directory
  private async checkForNewFiles() {
    if (!this.isWatching || !this.folderPath) {
      return;
    }

    try {
      const currentTime = new Date();

      // Method 1: Check for changes since last check
      const changedFiles = await getDirectoryChanges(
        this.folderPath,
        this.lastCheckedTime
      );

      // Method 2: Get all current files and compare with previous list
      this.currentVideos = await readVideoFilesFromDirectory(this.folderPath);

      // Find new files from both methods (avoiding duplicates)
      const newFilesFromChanges = changedFiles.filter(
        file => !this.isProcessed(file)
      );
      const newFilesFromList = this.currentVideos.filter(
        file => !this.previousVideos.includes(file) && !this.isProcessed(file)
      );

      // Combine new files from both methods (avoiding duplicates)
      const allNewFiles = [
        ...new Set([...newFilesFromChanges, ...newFilesFromList]),
      ];

      if (allNewFiles.length > 0) {
        console.log(`Found ${allNewFiles.length} new files to process`);
      }

      // Process each new file
      for (const file of allNewFiles) {
        await this.processVideoFile(file);
      }

      // Update previous state
      this.previousVideos = [...this.currentVideos];
      this.lastCheckedTime = currentTime;
    } catch (error) {
      console.error('Error checking for new files:', error);
    }
  }

  // Process a video file by sending it to the API
  private async processVideoFile(filePath: string) {
    if (this.isProcessed(filePath)) {
      return;
    }

    console.log(`Processing new video: ${filePath}`);

    try {
      // Mark as processed early to prevent duplicate processing if there are multiple checks
      this.markAsProcessed(filePath);

      // Get the actual video file from the path
      const videoFile = await this.getVideoFile(filePath);

      // Check server availability before making API call
      const serverAvailable = await this.checkServerStatus();
      if (!serverAvailable) {
        console.warn('Server is not available. Skipping video processing.');
        return;
      }

      // Process with API - pass the actual file if available
      const response = await apiService.processVideo(filePath, videoFile);

      // If shoplifting was detected, create a notification
      if (response.shoplifting_detected) {
        const notification: ShopliftingNotification = {
          id: uuidv4(),
          videoPath: response.video_path || filePath,
          timestamp: response.timestamp || new Date().toISOString(),
          cameraId: response.camera_id || this.extractCameraId(filePath),
          confirmed: null,
          dateDetected: new Date().toISOString(),
          video_id: response.video_id,
        };

        storageService.saveNotification(notification);
        console.log('Shoplifting detected! Notification saved.');
      } else {
        console.log('No shoplifting detected in this video.');
      }
    } catch (error) {
      console.error('Error processing video:', error);

      // In case of an error, we might want to consider unmarking the file as processed
      // so it can be retried next time, but this could lead to infinite retries
      // this.processedFiles.delete(filePath);
    }
  }

  // Get a video file from a path
  private async getVideoFile(filePath: string): Promise<File | undefined> {
    try {
      // Use our utility function to get the actual video file via the server
      console.log(`Fetching video file: ${filePath}`);
      return await getVideoFile(filePath);
    } catch (error) {
      console.error('Error accessing video file:', error);
      return undefined;
    }
  }

  // Extract camera ID from file path
  private extractCameraId(path: string): string {
    // Try to extract from Shoplifting*_x264_N.mp4 format
    const cameraMatch = path.match(/Shoplifting\d+_x264_(\d+)\.mp4$/);
    if (cameraMatch && cameraMatch[1]) {
      return `cam${cameraMatch[1]}`;
    }

    // Try to extract from path containing cam{N}
    const dirMatch = path.match(/cam(\d+)/);
    if (dirMatch && dirMatch[1]) {
      return `cam${dirMatch[1]}`;
    }

    // Extract anything that looks like a number as camera ID
    const numericMatch = path.match(/(\d+)\.mp4$/);
    if (numericMatch && numericMatch[1]) {
      return `cam${numericMatch[1]}`;
    }

    return 'cam0'; // Default camera ID
  }
}
