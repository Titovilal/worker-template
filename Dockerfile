FROM runpod/base:0.6.2-cuda12.4.1

# Install PyTorch
RUN echo "Installing PyTorch..." && \
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124


# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN echo "Upgrading pip..." && \
    python3.11 -m pip install --upgrade pip && \
    echo "Installing dependencies from requirements.txt..." && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    echo "Removing requirements.txt..." && \
    rm /requirements.txt

# Copy ComfyUI
COPY ComfyUI /ComfyUI

# Add src files (Worker Template)
ADD handler.py .
ADD comfy_serverless.py .
ADD utils.py .
ADD test_input.json .

# Create start script for ComfyUI
RUN echo -e '#!/bin/bash\nnohup python3.11 -u /ComfyUI/main.py &' > /start-comfyui.sh && \
    chmod +x /start-comfyui.sh


# Run both ComfyUI and handler
CMD /start-comfyui.sh && sleep 10 && python3.11 -u /handler.py

