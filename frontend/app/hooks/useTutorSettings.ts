'use client';

import { useState, useEffect } from 'react';
import { TutorSettings, DEFAULT_TUTOR_SETTINGS } from '../types/tutor';

const STORAGE_KEY = 'tutor_settings';

/**
 * Hook for managing user preferences for tutor UI
 * Persists settings to localStorage
 */
export function useTutorSettings() {
  const [settings, setSettings] = useState<TutorSettings>(DEFAULT_TUTOR_SETTINGS);
  const [isLoaded, setIsLoaded] = useState(false);

  // Load from localStorage on mount
  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        setSettings({ ...DEFAULT_TUTOR_SETTINGS, ...parsed });
      }
    } catch (e) {
      console.warn('Failed to load tutor settings from localStorage:', e);
    }
    setIsLoaded(true);
  }, []);

  // Save to localStorage when settings change
  const updateSettings = (updates: Partial<TutorSettings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...updates };
      try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch (e) {
        console.warn('Failed to save tutor settings to localStorage:', e);
      }
      return next;
    });
  };

  const resetSettings = () => {
    setSettings(DEFAULT_TUTOR_SETTINGS);
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch (e) {
      console.warn('Failed to reset tutor settings:', e);
    }
  };

  return {
    settings,
    isLoaded,
    updateSettings,
    resetSettings,
  };
}

