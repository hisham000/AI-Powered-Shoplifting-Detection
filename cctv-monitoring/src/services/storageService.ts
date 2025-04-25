import { ShopliftingNotification } from '../models/types';

const NOTIFICATIONS_KEY = 'cctv_shoplifting_notifications';
const CONFIG_KEY = 'cctv_folder_config';

export const storageService = {
  // Save notification to localStorage
  saveNotification: (notification: ShopliftingNotification): void => {
    const existingNotifications = storageService.getNotifications();
    
    // Check if this notification already exists
    const exists = existingNotifications.some(n => n.videoPath === notification.videoPath);
    
    if (!exists) {
      const updatedNotifications = [...existingNotifications, notification];
      localStorage.setItem(NOTIFICATIONS_KEY, JSON.stringify(updatedNotifications));
    }
  },
  
  // Get all notifications from localStorage
  getNotifications: (): ShopliftingNotification[] => {
    const notifications = localStorage.getItem(NOTIFICATIONS_KEY);
    return notifications ? JSON.parse(notifications) : [];
  },
  
  // Update notification status (confirmed shoplifting or not)
  updateNotification: (id: string, confirmed: boolean): void => {
    const existingNotifications = storageService.getNotifications();
    const updatedNotifications = existingNotifications.map(notification => 
      notification.id === id 
        ? { ...notification, confirmed } 
        : notification
    );
    
    localStorage.setItem(NOTIFICATIONS_KEY, JSON.stringify(updatedNotifications));
  },
  
  // Clear all notifications from localStorage
  clearNotifications: (): void => {
    localStorage.removeItem(NOTIFICATIONS_KEY);
  },
  
  // Save folder path configuration
  saveFolderConfig: (folderPath: string): void => {
    localStorage.setItem(CONFIG_KEY, folderPath);
  },
  
  // Get folder path configuration
  getFolderConfig: (): string | null => {
    return localStorage.getItem(CONFIG_KEY);
  }
};