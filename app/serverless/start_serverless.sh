#!/bin/sh

echo "Starting ..."
set -eu

FUNCTIONS_DIR="/app/serverless/tasks"

export DOCKER_BUILDKIT=1

# for func_config in "$FUNCTIONS_DIR"/function.yaml
for func_config in "$FUNCTIONS_DIR"/**/function.yaml
do
    func_root="$(dirname "$func_config")"
    echo "func_config", $func_config
    echo "func_root", $func_root

    echo "Deploying $func_root function..."
    nuctl deploy --project-name default --path "$func_root" \
        --file "$func_config" --platform local --verbose
done


nuctl get function --platform local

tail -f /dev/null