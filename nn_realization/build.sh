set -o xtrace

setup_root() {
    apt-get install -qq -y \
        ffmpeg             \
        git                \
        libsm6             \
        libxext6           \
        python3-pip        \
        python3-tk         \
        ;

    python3 -m pip install -qq             \
        aiohttp==3.8.5                     \
        aiosignal==1.3.1                   \
        albumentations==1.3.1              \
        albumentations-experimental==0.0.1 \
        async-timeout==4.0.3               \
        attrs==23.1.0                      \
        certifi==2023.7.22                 \
        charset-normalizer==3.2.0          \
        cmake==3.27.5                      \
        contourpy==1.1.1                   \
        cycler==0.11.0                     \
        decorator==4.4.2                   \
        exceptiongroup==1.1.3              \
        filelock==3.12.4                   \
        fonttools==4.42.1                  \
        frozenlist==1.4.0                  \
        fsspec==2023.9.1                   \
        huggingface-hub==0.17.1            \
        idna==3.4                          \
        imageio==2.31.3                    \
        imageio-ffmpeg==0.4.9              \
        imgaug==0.4.0                      \
        iniconfig==2.0.0                   \
        Jinja2==3.1.2                      \
        joblib==1.3.2                      \
        kiwisolver==1.4.5                  \
        lazy_loader==0.3                   \
        lightning-utilities==0.9.0         \
        lit==16.0.6                        \
        MarkupSafe==2.1.3                  \
        matplotlib==3.8.0                  \
        moviepy==1.0.3                     \
        mpmath==1.3.0                      \
        multidict==6.0.4                   \
        networkx==3.1                      \
        numpy==1.26.0                      \
        nvidia-cublas-cu11==11.10.3.66     \
        nvidia-cuda-cupti-cu11==11.7.101   \
        nvidia-cuda-nvrtc-cu11==11.7.99    \
        nvidia-cuda-runtime-cu11==11.7.99  \
        nvidia-cudnn-cu11==8.5.0.96        \
        nvidia-cufft-cu11==10.9.0.58       \
        nvidia-curand-cu11==10.2.10.91     \
        nvidia-cusolver-cu11==11.4.0.1     \
        nvidia-cusparse-cu11==11.7.4.91    \
        nvidia-nccl-cu11==2.14.3           \
        nvidia-nvtx-cu11==11.7.91          \
        opencv-python==4.8.0.76            \
        opencv-python-headless==4.8.0.76   \
        packaging==23.1                    \
        pandas==2.1.0                      \
        Pillow==10.0.1                     \
        pluggy==1.3.0                      \
        proglog==0.1.10                    \
        pyparsing==3.1.1                   \
        pytest==7.4.2                      \
        python-dateutil==2.8.2             \
        pytorch-lightning==2.0.9           \
        pytz==2023.3.post1                 \
        PyWavelets==1.4.1                  \
        PyYAML==6.0.1                      \
        qudida==0.0.4                      \
        requests==2.31.0                   \
        safetensors==0.3.3                 \
        scikit-image==0.21.0               \
        scikit-learn==1.3.0                \
        scipy==1.11.2                      \
        shapely==2.0.1                     \
        six==1.16.0                        \
        sympy==1.12                        \
        threadpoolctl==3.2.0               \
        tifffile==2023.8.30                \
        timm==0.9.7                        \
        tomli==2.0.1                       \
        torch==2.0.1                       \
        torchmetrics==1.1.2                \
        torchvision==0.15.2                \
        tqdm==4.66.1                       \
        triton==2.0.0                      \
        typing_extensions==4.8.0           \
        tzdata==2023.3                     \
        urllib3==2.0.4                     \
        yarl==1.9.2                        \
        ;
}

setup_checker() {
    python3 --version # Python 3.10.12
    python3 -m pip freeze # see list above
    python3 -c 'import matplotlib.pyplot'
}

"$@"