# Use an official Node runtime as a parent image
FROM node:23-slim AS build

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json (or yarn.lock)
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# accept build args
ARG REACT_APP_EEP_URL
ARG REACT_APP_FILE_SERVER_URL

# expose them to the build
ENV REACT_APP_EEP_URL=${REACT_APP_EEP_URL}
ENV REACT_APP_FILE_SERVER_URL=${REACT_APP_FILE_SERVER_URL}

# Build the application
RUN npm run build

# Production Stage
FROM nginx:stable-alpine AS production

# Copy the built app from 'build' stage to nginx web server directory
COPY --from=build /app/build /usr/share/nginx/html

# Expose port 80 for the container
EXPOSE 80

# Start nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]
