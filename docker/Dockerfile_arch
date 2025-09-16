# syntax=docker/dockerfile:1.2
FROM archlinux:latest

# Install base packages
RUN pacman -Sy --noconfirm sudo base-devel git

# Make a builder user
RUN useradd -m builder
RUN echo "builder ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER builder

# Set working directory
WORKDIR /home/builder/src

# Install yay for AUR packages
RUN sudo pacman -S --noconfirm go
RUN git clone https://aur.archlinux.org/yay
RUN makepkg -si --noconfirm -D yay

# Install Python interpreter
RUN yay -Sy --noconfirm python312
RUN python3.12 -m ensurepip
ENV PATH="/home/builder/.local/bin:${PATH}"

# Install CUDA toolkit
RUN sudo pacman -S --noconfirm cuda
ENV PATH="/opt/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/cuda/lib:${LD_LIBRARY_PATH}"
ENV NVCC_CCBIN="/usr/bin/gcc-14"

# Install python tools
RUN pip3.12 install ipython torch huggingface-hub transformers

# Install ziggy locally
RUN git clone https://github.com/CompendiumLabs/ziggy
RUN pip3.12 install -e ziggy

# Install text-embeddings-inference (and rust)
ENV CUDA_COMPUTE_CAP=80
RUN sudo pacman -S --noconfirm rust cargo
RUN git clone https://github.com/huggingface/text-embeddings-inference
RUN cargo install --path text-embeddings-inference/router -F candle-cuda
ENV PATH="/home/builder/.cargo/bin:${PATH}"

# Set starting directory
WORKDIR /home/builder

# Install user tools
RUN sudo pacman -S --noconfirm screen zsh xh
RUN pip3.12 install nvitop

# Install oh-my-zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
ENV SHELL="/bin/zsh"

# Keep container running
CMD ["tail", "-f", "/dev/null"]
