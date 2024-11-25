import os

# flake8: noqa

class DatabaseConfig:
    USER_DATABASE = os.environ.get("USER_DATABASE", "postgres")
    PASSWORD_DATABASE = os.environ.get("PASSWORD_DATABASE", "admin123")
    HOST_DATABASE = os.environ.get("HOST_DATABASE", "localhost")
    PORT_DATABASE = os.environ.get("PORT_DATABASE", "5412")
    NAME_DATABASE = os.environ.get("NAME_DATABASE", "data-recall-system")
    URL_DATABASE = f"postgresql://{USER_DATABASE}:{PASSWORD_DATABASE}@{HOST_DATABASE}:{PORT_DATABASE}/{NAME_DATABASE}"

class MinIOConfig:
    MINIO_DOMAIN = "localhost"
    MINIO_USER = "minio"
    MINIO_PASSWORD = "12345678"

class RedisConfig:
    REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
    REDIS_PORT = os.environ.get("REDIS_PORT", 6379)
    REDIS_PASS = os.environ.get("REDIS_PASS", "")
    REDIS_DB = os.environ.get("REDIS_DB", 0)
    CELERY_RESULT_BACKEND = "redis://:{password}@{hostname}:{port}/{db}".format(
        hostname=REDIS_HOST,
        password=REDIS_PASS,
        port=REDIS_PORT,
        db=REDIS_DB
    )
    
class BrokerConfig:
    BROKER_HOST = os.environ.get("BROKER_HOST", "localhost")
    BROKER_PORT = os.environ.get("BROKER_PORT", 5672)
    BROKER_USER = os.environ.get("BROKER_USER", "guest")
    BROKER_PASS = os.environ.get("BROKER_PASS", "guest")
    BROKER_VHOST = os.environ.get("BROKER_VHOST", "")
    CELERY_BROKER_URL = "amqp://{user}:{pw}@{hostname}:{port}/{vhost}".format(
        user=BROKER_USER,
        pw=BROKER_PASS,
        hostname=BROKER_HOST,
        port=BROKER_PORT,
        vhost=BROKER_VHOST
    )

class QdrantConfig:
    QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
    QDRANT_PORT = os.environ.get("QDRANT_PORT", 6333)
    
class NuclioConfig:
    NUCLIO_SCHEME = os.environ.get("NUCLIO_SCHEME", "http")
    NUCLIO_HOST = os.environ.get("NUCLIO_HOST", "localhost")
    NUCLIO_PORT = os.environ.get("NUCLIO_PORT", 8071)
    NUCLIO_FUNCTION_NAMESPACE = os.environ.get("NUCLIO_FUNCTION_NAMESPACE", "nuclio")
    NUCLIO_DEFAULT_TIMEOUT = os.environ.get("NUCLIO_DEFAULT_TIMEOUT", 120)

class CvatConfig:
    CVAT_DOMAIN = os.environ.get("CVAT_DOMAIN", "https://www.cvat.ai/")
    CVAT_USERNAME = os.environ.get("USER_NAME", "admin")
    CVAT_PASSWORD = os.environ.get("PASSWORD", "12345678")
    
class Config:
    database = DatabaseConfig()
    minio = MinIOConfig()
    redis = RedisConfig()
    broker = BrokerConfig()
    nuclio = NuclioConfig()
    qdrant = QdrantConfig()
    cvat = CvatConfig()