Running the Dockerized Application : Our application is now fully Dockerized, requiring only Docker to be installed.

Step 1: Install "Docker Desktop" from Docker’s official site and ensure it is running in the background.

Step 2: Navigate to the project directory where docker-compose.yml is located:

cd /path/to/project  

Step 3: Verify that docker-compose.yml is present in the directory.

Step 4: Build and start the application by running:
        a. docker-compose up --build -d  

Step 5: Ensure you are connected to the internet as necessary libraries will be downloaded. 
Once the process completes successfully, open a browser and go to http://localhost:4200/ to 
access the application. You can upload images and check the classification, which supports 12 classes.
