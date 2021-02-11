# downloading the dataset
wget http://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/usodi/img/projects/detection/iiit-ar-13/dataset/IIIT-AR-13K_dataset.zip

#unzipping the dataset
unzip IIIT-AR-13K_dataset.zip -d zip_data

# unzipping the train-test-validation dataset
apt-get install pv
unzip -o zip_data/training_images.zip | pv -l >/dev/null
unzip -o zip_data/training_xml.zip | pv -l >/dev/null

unzip -o zip_data/test_images.zip | pv -l >/dev/null
unzip -o zip_data/test_xml.zip | pv -l >/dev/null

unzip -o zip_data/validation_images.zip | pv -l >/dev/null
unzip -o zip_data/validation_xml.zip | pv -l >/dev/null

# removing zip data
rm IIIT-AR-13K_dataset.zip
rm -rf zip_data
