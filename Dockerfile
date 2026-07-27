FROM docker.io/library/rust:1.97-slim-bookworm@sha256:b001fed8c602fe3126bfee18c7afa14fe58dc855ce1d0cdfb4ac3ee7d6361a1c AS build

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --locked --release

FROM docker.io/library/debian:bookworm-slim@sha256:63a496b5d3b99214b39f5ed70eb71a61e590a77979c79cbee4faf991f8c0783e

COPY --from=build /src/target/release/gemma-gateway /usr/local/bin/gemma-gateway
USER 65532:65532
EXPOSE 8899
ENTRYPOINT ["/usr/local/bin/gemma-gateway"]
CMD ["--listen", "0.0.0.0:8899", "--upstream", "http://host.docker.internal:8891/v1"]
