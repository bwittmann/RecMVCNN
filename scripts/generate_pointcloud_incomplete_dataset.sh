#!/bin/bash
for i in {0..35707}
do
   python scripts/generate_pointcloud_dataset.py --index $i --split train --incomplete
done
for i in {0..5157}
do
   python scripts/generate_pointcloud_dataset.py --index $i --split val --incomplete
done
for i in {0..10260}
do
   python scripts/generate_pointcloud_dataset.py --index $i --split test --incomplete
done