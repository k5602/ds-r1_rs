# syntax=docker/dockerfile:1

# Minimal, reproducible Dockerfile for building and testing ds-r1-rs

ARG RUST_VERSION=1
FROM rust:${RUST_VERSION}-bookworm AS builder

ENV CARGO_TERM_COLOR=always \
    RUST_BACKTRACE=1 \
    CARGO_INCREMENTAL=0

# Install common native deps some crates require
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Leverage Docker layer caching: first build deps with a dummy src
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src && printf 'fn main(){println!("build cache");}\n' > src/main.rs
RUN cargo build --release --locked

# Now copy the full workspace and build/tests
# (This Dockerfile lives at ds-r1_rs/, so the build context should be that dir)
COPY . .

# Build the release binary reproducibly (uses Cargo.lock)
RUN cargo build --release --locked

# Run tests (unit + integration). If you want to skip during image build,
# you can comment this line or pass BuildKit secret/env and gate it.
RUN cargo test --all --release --locked

# Runtime image with just the binary
FROM debian:bookworm-slim AS runtime

# Minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV RUST_LOG=info

# Copy the binary from the builder
COPY --from=builder /app/target/release/ds-r1-rs /usr/local/bin/ds-r1-rs

# Default entrypoint runs the CLI; override with `docker run ... <command>`
ENTRYPOINT ["ds-r1-rs"]
CMD ["version"]
