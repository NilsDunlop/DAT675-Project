#!/bin/bash


for cutoff in 2 3 5 6 7; do
	sed -i 's/cutoff=.*/cutoff=${cutoff}/g' make_networks.slurm
	sbatch make_networks.slurm
done
