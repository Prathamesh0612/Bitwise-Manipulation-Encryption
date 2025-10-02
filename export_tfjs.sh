#!/bin/bash

# Convert Keras model to TensorFlow.js - Fixed version
set -e

MODEL_FILE="nec_policy_keras.h5"
OUTPUT_DIR="./web/model"

echo "🔄 Installing/upgrading tensorflowjs..."
python -m pip install --upgrade tensorflowjs

echo "📁 Creating output directory..."
mkdir -p "$OUTPUT_DIR"

if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Error: $MODEL_FILE not found. Run training first:"
    echo "   python train_nec_policy.py --synth"
    exit 1
fi

echo "🚀 Converting $MODEL_FILE to TensorFlow.js format..."
python -m tensorflowjs_converter --input_format=keras "$MODEL_FILE" "$OUTPUT_DIR"

if [ -f "$OUTPUT_DIR/model.json" ]; then
    echo "✅ Conversion successful!"
    echo "📊 Generated files:"
    ls -la "$OUTPUT_DIR/"
    echo ""
    echo "📋 Next steps:"
    echo "  1. Place the web/ folder in your web server"
    echo "  2. Open index.html in a browser"
    echo "  3. AI model will load automatically"
else
    echo "❌ Conversion failed - model.json not found"
    exit 1
fi
