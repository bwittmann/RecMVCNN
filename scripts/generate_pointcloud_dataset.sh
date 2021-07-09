#!/bin/bash
for i in {0..35708}
do
   python scripts/generate_pointcloud_dataset.py --index $i
done