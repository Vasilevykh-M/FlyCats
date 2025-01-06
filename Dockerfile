# Используем базовый образ с Ubuntu и поддержкой OpenGL
FROM ubuntu:22.04

# Устанавливаем необходимые зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libglfw3-dev \
    libglew-dev \
    libglm-dev \
    libassimp-dev \
    libx11-dev \
    libxi-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxxf86vm-dev \
    libxcb-xkb-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    mesa-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем stb_image (если требуется)
RUN mkdir -p /usr/local/include/stb && \
    curl -o /usr/local/include/stb/stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h

# Копируем исходный код проекта в контейнер
COPY . /app
WORKDIR /app

# Создаем директорию для сборки
RUN mkdir build && cd build && cmake .. && make

RUN chmod +x run.sh

# Указываем точку входа для запуска программы
CMD ["./run.sh"]
