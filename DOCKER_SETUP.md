# vantage Django + MongoDB Application

A Django web application integrated with MongoDB and containerized with Docker.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.13+
- Virtual environment (venv)

## Local Development Setup

### Without Docker

1. **Create and activate virtual environment:**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install and run MongoDB locally:**
   - Download MongoDB Community Edition from [mongodb.com](https://www.mongodb.com)
   - Or use MongoDB Atlas (cloud): Update `MONGO_URI` in `.env`

4. **Run Django development server:**
```bash
python manage.py runserver
```

The application will be available at `http://localhost:8000`

### With Docker (Recommended)

1. **Build and start containers:**
```bash
docker-compose up --build
```

2. **In another terminal, run migrations (if needed):**
```bash
docker-compose exec web python manage.py migrate
```

3. **Access the application:**
   - Django app: `http://localhost:8000`
   - MongoDB Express (UI): `http://localhost:8081`

## Docker Services

### MongoDB
- **Container:** vantage-mongodb
- **Port:** 27017
- **Database:** vantage
- **Connection String:** `mongodb://mongodb:27017/vantage`

### Django Web Server
- **Container:** vantage-web
- **Port:** 8000
- **Auto-reloads** on file changes (mounted volume)

### MongoDB Express (Optional)
- **Container:** vantage-mongo-express
- **Port:** 8081
- Web interface for MongoDB database management

## Environment Variables

See `.env` file for configuration:

```env
# Auth0
AUTH0_CLIENT_ID=your_client_id
AUTH0_CLIENT_SECRET=your_client_secret
AUTH0_DOMAIN=your_domain

# MongoDB
MONGO_URI=mongodb://mongodb:27017/vantage

# Django
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
```

## Project Structure

```
vantage/
├── app/
│   ├── models.py          # MongoDB models using MongoEngine
│   ├── views.py           # Django views with Auth0 integration
│   ├── urls.py            # URL routing
│   ├── settings.py        # Django configuration
│   ├── templates/         # HTML templates
│   └── wsgi.py
├── Dockerfile             # Docker image configuration
├── docker-compose.yml     # Multi-container setup
├── requirements.txt       # Python dependencies
├── manage.py
├── .env                   # Environment variables
└── .dockerignore
```

## Models

The application includes MongoDB models in `app/models.py`:

- **User:** User profile with Auth0 integration
- **Session:** Session management
- **Log:** Audit logging

## Useful Commands

### Docker Commands

```bash
# Start services
docker-compose up

# Start in background
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f web

# Access web container shell
docker-compose exec web bash

# Access MongoDB shell
docker-compose exec mongodb mongosh
```

### Django Commands

```bash
# Run server
python manage.py runserver

# Create superuser
python manage.py createsuperuser

# Run migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic

# Run tests
python manage.py test
```

### MongoDB Commands

```bash
# Access MongoDB shell
docker-compose exec mongodb mongosh

# List databases
show dbs

# Use vantage database
use vantage

# List collections
show collections

# Query users
db.users.find().pretty()
```

## MongoDB Integration

### MongoEngine ORM

The project uses [MongoEngine](https://docs.mongoengine.org/) for MongoDB integration:

```python
from app.models import User

# Create a user
user = User(name="John", email="john@example.com")
user.save()

# Query users
users = User.objects(email="john@example.com")

# Update
user.update(name="Jane")

# Delete
user.delete()
```

### Direct PyMongo Access

For raw MongoDB operations:

```python
from mongoengine import connect
import pymongo

# Connected automatically via settings.py
db = connect('vantage', host='mongodb://mongodb:27017/vantage')
```

## Production Deployment

For production, update `docker-compose.yml`:

1. Change `DEBUG=False` in environment variables
2. Use strong `SECRET_KEY` in `settings.py`
3. Set proper `ALLOWED_HOSTS`
4. Use environment-specific `.env` files
5. Consider using Gunicorn with multiple workers
6. Add Nginx as reverse proxy
7. Use proper MongoDB authentication

Example production update in `Dockerfile`:

```dockerfile
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "app.wsgi"]
```

## Troubleshooting

### MongoDB Connection Issues
- Ensure MongoDB container is healthy: `docker-compose ps`
- Check logs: `docker-compose logs mongodb`
- Verify connection string in `.env`

### Port Already in Use
- Check which service is using the port: `netstat -ano | findstr :8000`
- Change port in `docker-compose.yml` if needed

### Database Sync Issues
- Remove MongoDB volume: `docker-compose down -v`
- Rebuild: `docker-compose up --build`

## Additional Resources

- [Django Documentation](https://docs.djangoproject.com/)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [MongoEngine Documentation](https://docs.mongoengine.org/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Auth0 Django Integration](https://auth0.com/docs/quickstart/webapp/django)

## Support

For issues or questions, check the documentation or open an issue in the repository.
