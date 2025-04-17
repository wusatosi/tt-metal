FROM quay.io/pypa/manylinux_2_28_x86_64

# Install Clang 17 (adapt to your favorite method)
RUN curl https://apt.llvm.org/llvm.sh | bash -s 17 \
 && ln -s /usr/bin/clang-17 /usr/local/bin/clang \
 && ln -s /usr/bin/clang++-17 /usr/local/bin/clang++

# Optional: install Python 3.10
RUN /opt/python/cp310*/bin/pip install -U pip setuptools wheel

ENV CC=clang CXX=clang++
