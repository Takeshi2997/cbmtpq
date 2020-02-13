#!/bin/sh

x=${1:?}
y=${x%.*}

gnuplot << EOF
    set terminal png
    set output '$y.png'
    set xlabel "inverse temperature"
    set ylabel "energy expected value"
    set xrange [0.0:3.0]
    plot '$x' u 1:3 title "numerical solution", 'energy_expected_value.txt' u 1:3 w l title "exact solution"
EOF



