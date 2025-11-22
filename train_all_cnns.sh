#!/bin/bash
# Train all 3 CNNs in parallel

echo "Training all 3 CNNs in parallel..."

python train_single_cnn.py --feature mfcc --epochs 30 --batch_size 32 > logs/mfcc_train.log 2>&1 &
PID1=$!

python train_single_cnn.py --feature melspec --epochs 30 --batch_size 32 > logs/melspec_train.log 2>&1 &
PID2=$!

python train_single_cnn.py --feature dwt --epochs 30 --batch_size 32 > logs/dwt_train.log 2>&1 &
PID3=$!

echo "Training started:"
echo "  MFCC CNN (PID: $PID1)"
echo "  MelSpec CNN (PID: $PID2)"
echo "  DWT CNN (PID: $PID3)"

# Wait for all to complete
wait $PID1 $PID2 $PID3

echo "All training complete!"