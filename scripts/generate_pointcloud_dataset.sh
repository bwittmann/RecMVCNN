#!/bin/bash
for i in {0..35708}
do
   python scripts/generate_pointcloud_dataset.py --index $i --split train
done
for i in {0..5158}
do
   python scripts/generate_pointcloud_dataset.py --index $i --split val
done
for i in {0..10261}
do
   python scripts/generate_pointcloud_dataset.py --index $i --split test
done