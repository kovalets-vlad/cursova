===DOCKER-COMPOSE===
How to build and run with Docker Compose
/cd cursova

BUILD AND RUN
docker-compose up --build

RUN
docker-compose up


===DOCKERFILE===

How to build and run with Docker for backend

BUILD

/cd backend
docker build -t my-backend-app .

RUN

docker run -it --rm -p 8000:8000 --env-file .env --name backend-container my-backend-app

How to build and run with Docker for frontend

BUILD

/cd frontend
docker build -t my-frontend-app .

RUN

docker run -it -p 5173:5173 --name frontend-container my-frontend-app


===MANUAL===

How to start backend

/cd backend
uvicorn main:app --reload

How to start frontend 
/cd frontend
npm run dev