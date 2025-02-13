FROM node:20-alpine AS build

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json first (for caching)
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the entire Angular project
COPY . .

RUN npm run build --configuration=production


RUN ls -l /app/dist/


FROM nginx:1.21.6-alpine


COPY --from=build /app/dist/garbage-classifier/browser /usr/share/nginx/html

# Expose port 80 for HTTP traffic
EXPOSE 80

# Start Nginx server
CMD ["nginx", "-g", "daemon off;"]