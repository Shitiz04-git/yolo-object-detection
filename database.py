"""
Database Module - Store detection results for analytics
"""

import sqlite3
import json
from datetime import datetime
from contextlib import contextmanager

class DetectionDatabase:
    """SQLite database for storing detection results"""
    
    def __init__(self, db_path='detections.db'):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Detections table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    source_type TEXT,
                    source_path TEXT,
                    total_objects INTEGER,
                    detection_data TEXT,
                    processing_time_ms REAL
                )
            ''')
            
            # Analytics table for daily stats
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date DATE PRIMARY KEY,
                    total_detections INTEGER,
                    unique_sources INTEGER,
                    object_counts TEXT
                )
            ''')
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def save_detection(self, source_type: str, source_path: str, 
                       results_data: dict, processing_time: float):
        """Save detection results to database"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO detections 
                (timestamp, source_type, source_path, total_objects, 
                 detection_data, processing_time_ms)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                source_type,
                source_path,
                results_data.get('total_objects', 0),
                json.dumps(results_data),
                processing_time
            ))
            conn.commit()
    
    def get_stats(self, days: int = 7) -> dict:
        """Get statistics for last N days"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*), AVG(total_objects), AVG(processing_time_ms)
                FROM detections
                WHERE timestamp >= datetime('now', ?)
            ''', (f'-{days} days',))
            total, avg_objects, avg_time = cursor.fetchone()
            
            return {
                'total_detections': total or 0,
                'avg_objects_per_frame': avg_objects or 0,
                'avg_processing_time_ms': avg_time or 0,
                'days': days
            }