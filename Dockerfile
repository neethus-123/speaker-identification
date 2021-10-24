FROM python:3.9.1

# Expose port you want your app on
EXPOSE 8080

# Upgrade pip and install requirements
COPY requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt

# Copy app code and set working directory
COPY text_explorer text_explorer
COPY speaker-recognition-app.py speaker-recognition-app.py
COPY references references
WORKDIR .

# Run
ENTRYPOINT [“streamlit”, “run”, “speaker-recognition-app.py”, “–server.port=8501”, “–server.address=192.168.0.95”]
CMD streamlit run--server.port 8501--serverCORS false speaker-recognition-pp.py