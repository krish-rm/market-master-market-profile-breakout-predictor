## Docker Deployment

### Build Docker Image

```bash
docker build -t market-profile-ml:latest .
```

### Run Container

```bash
# Option 1: Direct Docker
docker run -it -p 9696:9696 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  market-profile-ml:latest

# Option 2: Docker Compose (easier)
docker-compose up -d
```

### Access API in Container

```bash
curl http://localhost:9696/health
```

### View Logs

```bash
docker logs -f market-profile-api
```

### Stop Container

```bash
docker-compose down
# OR
docker stop market-profile-api
```


