#!/bin/bash

echo "============================= VT-Physics Init Git Submodules ============================="
git submodule init
git submodule update

# Set vcpkg installation path
VCPKG_PATH="$(dirname "$0")/Simulator/Thirdparty/vcpkg"

# Set dependencies file path
DEPENDENCIES_FILE="dependencies.txt"

# Check if vcpkg executable exists
if [ ! -f "$VCPKG_PATH/vcpkg" ]; then
    echo "Note: vcpkg not found at $VCPKG_PATH. Init vcpkg..."
    "$VCPKG_PATH/bootstrap-vcpkg.sh"
fi

# Check if dependencies file exists
if [ ! -f "$DEPENDENCIES_FILE" ]; then
    echo "ERROR: Dependencies file not found: $DEPENDENCIES_FILE"
    exit 1
fi

echo "============================= VT-Physics Vcpkg Downloading Dependencies ============================="
# Iterate through each line in the dependencies file and install the dependencies
while IFS= read -r LIB_NAME || [ -n "$LIB_NAME" ]; do
    # Skip empty lines
    if [ -n "$LIB_NAME" ]; then
        echo "Installing $LIB_NAME ..."
        "$VCPKG_PATH/vcpkg" install "$LIB_NAME"
        
        # Check if installation was successful
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to install $LIB_NAME."
            exit 1
        else
            echo "Successfully installed $LIB_NAME!"
        fi
    fi
done < "$DEPENDENCIES_FILE"

echo "All dependencies installed successfully."
exit 0