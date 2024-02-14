# Use a base image that includes curl and shell utilities
FROM ubuntu:20.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install curl and other utilities you might need
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Run the Ollama installation script
RUN curl https://ollama.ai/install.sh | sh

# Copy the entrypoint script into the container
COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# Expose the default port Ollama might use
EXPOSE 8080

# Use the entrypoint script to start the server and install the LLM
ENTRYPOINT ["/entrypoint.sh"]
