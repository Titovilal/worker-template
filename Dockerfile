FROM runpod/base:0.6.2-cuda12.4.1

# Install PyTorch
# RUN echo "Installing PyTorch..." && \
# pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Copy ComfyUI
COPY ComfyUI /ComfyUI

# Add src files (Worker Template)
ADD handler.py .
ADD test_input.json .

# Create start script for ComfyUI and port check
RUN echo -e '#!/bin/bash\n\
nohup python3.11 -u /ComfyUI/main.py &\n\
\n\
# Wait for ComfyUI port to be ready\n\
while ! nc -z localhost 8188; do\n\
  echo "Waiting for ComfyUI to be ready..."\n\
  sleep 1\n\
done\n\
echo "ComfyUI is ready!"' > /start-comfyui.sh && \
    chmod +x /start-comfyui.sh

# Install netcat for port checking
RUN apt-get update && apt-get install -y netcat-openbsd

# Run both ComfyUI and handler
CMD /start-comfyui.sh && python3.11 -u /handler.py
