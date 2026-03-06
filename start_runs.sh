#!/bin/bash


for cutoff in 0.1 1 ; do
	sed -i "s/cutoff=.*/cutoff=${cutoff}/g" make_networks.slurm
	sbatch make_networks.slurm
done
