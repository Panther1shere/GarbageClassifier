1. Start the virtual Environment after installing python
	python3.11 -m venv tf_env

2. Activate virtual env
	mac:source tf_env/bin/activate
	windows: tf_env\Scripts\activate

3. pip install tensorflow==2.15.0 fastapi uvicorn python-multipart numpy pillow

4. pip install keras opencv-python scikit-learn


5. Command to start the API of python is :
uvicorn GarbageClassifierAPI:app --host 0.0.0.0 --port 8000 --reload

