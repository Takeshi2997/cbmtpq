#!/bin/sh

zip energy_data.zip energy_data.txt
zip exact_energy.zip exact_energy.txt
export LANG=ja_JP.utf8
(uuencode ./energy_data.zip energy_data.zip ; uuencode ./exact_energy.zip exact_energy.zip ; echo "energy data S = 8, B = 64" ) | mail -s "Energy Data" "takeshi6.6260@gmail.com"

