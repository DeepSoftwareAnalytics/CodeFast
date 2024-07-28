#!/bin/bash

# Specify the Go version you want to install
VERSION="1.19.1"

# Check if Go is already installed
if command -v go &> /dev/null; then
    echo "Uninstalling existing Go installation..."
    sudo rm -rf /usr/local/go
fi

# Download and install the specified Go version
echo "Downloading and installing Go $VERSION..."
wget "https://golang.org/dl/go$VERSION.linux-amd64.tar.gz" -O go.tar.gz
sudo tar -C /usr/local -xzf go.tar.gz
rm go.tar.gz

# Add Go binary directory to PATH
if ! grep -q '/usr/local/go/bin' "$HOME/.profile"; then
    echo 'export PATH=$PATH:/usr/local/go/bin' >> "$HOME/.profile"
    source "$HOME/.profile"
fi

# Verify the installation
go version

echo "Go $VERSION has been successfully installed."
