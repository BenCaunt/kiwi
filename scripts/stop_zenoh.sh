#!/usr/bin/env bash

exec python3 "$(dirname "$0")/stop_zenoh.py" "$@"
