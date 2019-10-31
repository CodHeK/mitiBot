mkdir datasets
cd datasets

echo "Created ./datasets folder"

echo "Download Training set files into ./datasets"

curl https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/detailed-bidirectional-flow-labels/capture20110810.binetflow -k -o 42.csv
echo "42.csv done"

curl https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-43/detailed-bidirectional-flow-labels/capture20110811.binetflow -k -o 43.csv
echo "43.csv done"

curl https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-46/detailed-bidirectional-flow-labels/capture20110815-2.binetflow -k -o 46.csv
echo "46.csv done"

curl https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-47/detailed-bidirectional-flow-labels/capture20110816.binetflow -k -o 47.csv
echo "47.csv done"

curl https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-48/detailed-bidirectional-flow-labels/capture20110816-2.binetflow -k -o 48.csv
echo "48.csv done"

curl https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-52/detailed-bidirectional-flow-labels/capture20110818-2.binetflow -k -o 52.csv
echo "52.csv done"

curl https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-53/detailed-bidirectional-flow-labels/capture20110819.binetflow -k -o 53.csv
echo "53.csv done"

echo "Done with training set!"

echo "Download Testing set files into ./datasets"

curl https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-50/detailed-bidirectional-flow-labels/capture20110817.binetflow -k -o 50.csv
echo "50.csv done"

curl https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-51/detailed-bidirectional-flow-labels/capture20110818.binetflow -k -o 51.csv
echo "51.csv done"

echo "Done with testing set!"
