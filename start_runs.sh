#!/bin/bash


for cutoff in 0.1 1 2 3 4 5 6 7 10; do
#for cutoff in 3 ; do
	sed -i "s/cutoff=.*/cutoff=${cutoff}/g" make_networks.slurm
	sbatch make_networks.slurm
done
