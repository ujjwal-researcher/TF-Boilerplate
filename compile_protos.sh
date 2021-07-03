#!/usr/bin/env bash

echo "Compiling Proto files"

protoc \
  --python_out=. \
  ./protos/*.proto

echo "Finished compiling"
