"""
Deployment Configuration for Production
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class DeploymentConfig:
    """Configuration for different deployment environments"""
    
    # Model settings
    model_path: str = 'yolov8n.pt'
    confidence_threshold: float = 0.5
    
    # Server settings
    host: str = '0.0.0.0'
    port: int = 5000
    workers: int = 4  # For Gunicorn
    
    # Database (for storing detection results)
    database_url: str = 'sqlite:///detections.db'
    
    # Redis for caching and rate limiting
    redis_url: str = 'redis://localhost:6379'
    
    # Cloud storage (AWS S3 / Azure Blob)
    cloud_storage_bucket: Optional[str] = None
    
    # Security
    secret_key: str = os.getenv('SECRET_KEY', 'your-secret-key-here')
    jwt_secret: str = os.getenv('JWT_SECRET', 'jwt-secret-here')
    
    # Logging
    log_level: str = 'INFO'
    log_file: str = 'detection.log'

# Environment-specific configs
class DevelopmentConfig(DeploymentConfig):
    debug: bool = True
    database_url: str = 'sqlite:///detections_dev.db'

class ProductionConfig(DeploymentConfig):
    debug: bool = False
    workers: int = 8  # More workers for production
    
class DockerConfig(DeploymentConfig):
    """Configuration for Docker deployment"""
    host: str = '0.0.0.0'
    port: int = 8080

# Select config based on environment
ENV = os.getenv('ENVIRONMENT', 'development')
if ENV == 'production':
    config = ProductionConfig()
elif ENV == 'docker':
    config = DockerConfig()
else:
    config = DevelopmentConfig()                    